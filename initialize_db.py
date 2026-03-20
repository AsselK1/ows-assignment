#!/usr/bin/env python3
"""
ClickHouse Database Schema Initialization for Kazakhstan Procurement Agent
Executes create_tables.sql and verifies all tables and materialized views
"""

import os
import sys
import requests
from urllib.parse import urlencode
import json
from typing import Dict, List, Tuple, Optional, Any

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST")
CLICKHOUSE_PORT = os.getenv("CLICKHOUSE_PORT", "8443")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD")
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "procurement")

# Construct ClickHouse HTTP endpoint
CLICKHOUSE_URL = f"https://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}"

# Expected tables to create
EXPECTED_TABLES = [
    "subjects",
    "plans",
    "announcements",
    "lots",
    "contracts",
    "contract_acts",
    "reference_enstr",
    "reference_kato",
    "reference_mkei",
    "anomaly_results",
]

# Expected materialized views
EXPECTED_MVS = [
    "mv_spend_by_bin",
    "mv_spend_by_enstr",
]


def execute_query(query: str, database: Optional[str] = None) -> Dict[str, Any]:
    """Execute a query against ClickHouse HTTP API"""
    try:
        params = {
            "user": CLICKHOUSE_USER,
            "password": CLICKHOUSE_PASSWORD,
        }

        if database:
            params["database"] = database

        headers = {"Content-Type": "text/plain"}

        response = requests.post(
            CLICKHOUSE_URL,
            data=query,
            params=params,
            headers=headers,
            verify=False,  # SSL verification disabled for cloud
            timeout=30,
        )

        if response.status_code != 200:
            return {"success": False, "status_code": response.status_code, "error": response.text}

        return {"success": True, "status_code": response.status_code, "data": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_database() -> bool:
    """Create procurement database if not exists"""
    print(f"[1/4] Creating database '{CLICKHOUSE_DB}'...")

    query = f"CREATE DATABASE IF NOT EXISTS {CLICKHOUSE_DB}"
    result = execute_query(query)

    if result["success"]:
        print(f"[OK] Database '{CLICKHOUSE_DB}' ready")
        return True
    else:
        print(f"[FAIL] Failed to create database: {result['error']}")
        return False


def create_tables() -> bool:
    """Read and execute create_tables.sql"""
    print(f"[2/4] Creating tables from create_tables.sql...")

    sql_file = "src/database/create_tables.sql"

    if not os.path.exists(sql_file):
        print(f"[FAIL] SQL file not found: {sql_file}")
        return False

    try:
        with open(sql_file, "r", encoding="utf-8") as f:
            sql_content = f.read()

        statements = []
        current_stmt = ""
        for line in sql_content.split("\n"):
            if line.strip() == "" and current_stmt.strip() == "":
                continue
            current_stmt += line + "\n"
            if line.strip().endswith(";"):
                statements.append(current_stmt.strip())
                current_stmt = ""

        if current_stmt.strip():
            statements.append(current_stmt.strip())

        failed = False
        for i, stmt in enumerate(statements, 1):
            if not stmt.strip():
                continue

            result = execute_query(stmt, CLICKHOUSE_DB)

            if not result["success"]:
                print(f"[FAIL] Statement {i} failed: {result['error'][:100]}")
                failed = True
                break

        if not failed:
            print(f"[OK] All {len(statements)} tables and materialized views created successfully")
            return True
        else:
            return False
    except Exception as e:
        print(f"[FAIL] Error reading SQL file: {str(e)}")
        return False


def verify_tables() -> Tuple[bool, List[str]]:
    """Verify all expected tables exist"""
    print(f"[3/4] Verifying table creation...")

    query = f"SHOW TABLES FROM {CLICKHOUSE_DB}"
    result = execute_query(query)

    if not result["success"]:
        print(f"[FAIL] Failed to list tables: {result['error']}")
        return False, []

    # Parse table names from response
    existing_tables = [line.strip() for line in result["data"].strip().split("\n") if line.strip()]

    print(f"  Found {len(existing_tables)} tables/views")

    missing = []
    for table_name in EXPECTED_TABLES + EXPECTED_MVS:
        if table_name in existing_tables:
            print(f"  [OK] {table_name}")
        else:
            print(f"  [FAIL] MISSING: {table_name}")
            missing.append(table_name)

    if missing:
        print(f"\n[FAIL] Missing {len(missing)} items: {', '.join(missing)}")
        return False, missing
    else:
        print(f"\n[OK] All {len(EXPECTED_TABLES) + len(EXPECTED_MVS)} tables and views verified")
        return True, []


def verify_table_structure() -> bool:
    """Verify table structure matches schema"""
    print(f"[4/4] Verifying table structure...")

    # Check a sample table structure
    query = f"DESCRIBE TABLE {CLICKHOUSE_DB}.subjects"
    result = execute_query(query)

    if not result["success"]:
        print(f"[FAIL] Failed to describe table: {result['error']}")
        return False

    # Parse structure - ClickHouse DESCRIBE returns tab-separated values
    lines = result["data"].strip().split("\n")
    columns = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            columns.append(parts[0])

    print(f"  Subjects table has {len(columns)} columns")

    # Verify key columns exist
    key_columns = ["id", "pid", "bin", "name_ru", "name_kz"]
    for col in key_columns:
        if col in columns:
            print(f"  [OK] Column: {col}")
        else:
            print(f"  [FAIL] Missing column: {col}")
            return False

    print(f"\n[OK] Table structure verified")
    return True


def test_connection() -> bool:
    """Test connectivity to ClickHouse"""
    print(f"[0/4] Testing ClickHouse connection...")
    print(f"  Host: {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}")
    print(f"  User: {CLICKHOUSE_USER}")
    print(f"  Database: {CLICKHOUSE_DB}")

    query = "SELECT 1 as ping"
    result = execute_query(query)

    if result["success"]:
        print(f"[OK] ClickHouse connection successful (response: {result['data'].strip()})")
        return True
    else:
        print(f"[FAIL] Connection failed: {result['error']}")
        return False


def main():
    """Main initialization flow"""
    print("=" * 70)
    print("ClickHouse Database Schema Initialization")
    print("Kazakhstan Procurement Agent")
    print("=" * 70)
    print()

    # Step 0: Test connection
    if not test_connection():
        print("\n[FAIL] Cannot proceed without ClickHouse connection")
        return False

    print()

    # Step 1: Create database
    if not create_database():
        return False

    print()

    # Step 2: Create tables
    if not create_tables():
        return False

    print()

    # Step 3: Verify tables
    success, missing = verify_tables()
    if not success:
        return False

    print()

    # Step 4: Verify structure
    if not verify_table_structure():
        return False

    print()
    print("=" * 70)
    print("[OK] DATABASE INITIALIZATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Summary:")
    print(f"  - Database: {CLICKHOUSE_DB}")
    print(f"  - Tables created: {len(EXPECTED_TABLES)}")
    print(f"  - Materialized views created: {len(EXPECTED_MVS)}")
    print(f"  - Indexes: Configured per table schema")
    print()

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInitialization cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
