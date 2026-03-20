#!/usr/bin/env python3
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clickhouse_connect
from src.rag.vector_store import VectorStore


def main():
    clickhouse = clickhouse_connect.get_client(
        host="HOST",
        port=8443,
        database="procurement",
        username="default",
        password="password",
        secure=True,
    )

    vector_store = VectorStore(
        qdrant_host="HOST",
        qdrant_port=6333,
        collection_name="procurement_documents",
    )
    vector_store.ensure_collection()

    # Get total count
    total = clickhouse.query("SELECT count() FROM contracts").result_rows[0][0]
    print(f"Total contracts: {total}")

    batch_size = 10
    offset = 0
    total_indexed = 0

    while offset < total:
        rows = clickhouse.query(f"""
            SELECT id FROM contracts ORDER BY id DESC LIMIT {batch_size} OFFSET {offset}
        """).result_rows

        if not rows:
            break

        for row in rows:
            contract_id = int(row[0])
            if contract_id <= 0:
                continue
            chunks = vector_store.index_contract(
                contract_id=contract_id, clickhouse_client=clickhouse
            )
            total_indexed += chunks
            print(f"Indexed contract {contract_id}, total: {total_indexed}")

        offset += batch_size
        print(f"Progress: {offset}/{total}")

    clickhouse.close()
    print(f"Done! Total indexed: {total_indexed}")


if __name__ == "__main__":
    main()
