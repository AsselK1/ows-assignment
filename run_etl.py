#!/usr/bin/env python3
"""
ETL Execution Wrapper Script
Loads .env file and executes ETL scripts with proper environment variables.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def load_env_file(env_path: Path) -> dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}

    if not env_path.exists():
        print(f"ERROR: .env file not found at {env_path}")
        sys.exit(1)

    print(f"Loading environment from {env_path}")

    with open(env_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value and value[0] in ("'", '"') and value[-1] == value[0]:
                    value = value[1:-1]

                env_vars[key] = value
            else:
                print(f"WARNING: Skipping malformed line {line_num}: {line}")

    print(f"Loaded {len(env_vars)} environment variables")
    return env_vars


def run_script(
    script_path: Path, env_vars: dict[str, str], extra_args: list[str] | None = None
) -> int:
    """Execute a Python script with environment variables loaded."""
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return 1

    # Merge with current environment
    full_env = os.environ.copy()
    full_env.update(env_vars)

    # Build command
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'=' * 80}")
    print(f"Executing: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    # Run the script
    result = subprocess.run(cmd, env=full_env)

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute ETL scripts with .env file loaded")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--script",
        type=str,
        choices=[
            "ref_data",
            "subjects",
            "plans",
            "announcements",
            "lots",
            "contracts",
            "contract_acts",
            "all",
        ],
        default="all",
        help="Which ETL script to run (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reload (truncate tables before loading)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for inserts (default: 500)",
    )

    args = parser.parse_args()

    # Load environment variables
    env_vars = load_env_file(args.env_file)

    # Determine which scripts to run
    project_root = Path(__file__).parent
    etl_dir = project_root / "src" / "etl"

    scripts_map = {
        "ref_data": etl_dir / "load_ref_data.py",
        "subjects": etl_dir / "etl_subjects.py",
        "plans": etl_dir / "etl_plans.py",
        "announcements": etl_dir / "etl_announcements.py",
        "lots": etl_dir / "etl_lots.py",
        "contracts": etl_dir / "etl_contracts.py",
        "contract_acts": etl_dir / "etl_contract_acts.py",
    }

    # Build extra args for scripts
    extra_args = []
    if args.force:
        extra_args.append("--force")
    if args.batch_size != 500:
        extra_args.extend(["--batch-size", str(args.batch_size)])

    # Execute scripts
    if args.script == "all":
        # Run in dependency order: reference data first, then entities
        execution_order = [
            "ref_data",
            "subjects",
            "plans",
            "announcements",
            "lots",
            "contracts",
            "contract_acts",
        ]

        for script_name in execution_order:
            script_path = scripts_map[script_name]
            print(f"\n{'#' * 80}")
            print(f"# Running: {script_name}")
            print(f"{'#' * 80}\n")

            exit_code = run_script(script_path, env_vars, extra_args)

            if exit_code != 0:
                print(f"\nERROR: Script {script_name} failed with exit code {exit_code}")
                print("Stopping execution.")
                return exit_code

            print(f"\nOK: {script_name} completed successfully")
    else:
        # Run single script
        script_path = scripts_map[args.script]
        exit_code = run_script(script_path, env_vars, extra_args)

        if exit_code != 0:
            print(f"\nERROR: Script {args.script} failed with exit code {exit_code}")
            return exit_code

        print(f"\nOK: {args.script} completed successfully")

    print(f"\n{'=' * 80}")
    print("ALL ETL SCRIPTS COMPLETED SUCCESSFULLY")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
