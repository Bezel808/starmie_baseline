#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert 1218 unionable labels to Starmie UTS folder layout."
    )
    parser.add_argument("--dataset-root", required=True, help="e.g. .../santos_benchmark_1218")
    parser.add_argument("--output-root", required=True, help="e.g. /tmp/starmie_uts1218")
    parser.add_argument("--max-valid-pairs", type=int, default=-1)
    parser.add_argument("--max-test-pairs", type=int, default=-1)
    parser.add_argument("--max-query-tables", type=int, default=-1)
    parser.add_argument("--max-datalake-tables", type=int, default=-1)
    args = parser.parse_args()

    label_dir = os.path.join(args.dataset_root, "label_plus", "unionable_table_search")
    datalake_plus = os.path.join(args.dataset_root, "datalake_plus")

    valid_df = pd.read_csv(os.path.join(label_dir, "validate.csv"))
    test_df = pd.read_csv(os.path.join(label_dir, "test.csv"))

    if args.max_valid_pairs > 0:
        valid_df = valid_df.head(args.max_valid_pairs)
    if args.max_test_pairs > 0:
        test_df = test_df.head(args.max_test_pairs)

    query_tables = list(
        dict.fromkeys(list(valid_df["table_name_1"]) + list(test_df["table_name_1"]))
    )
    if args.max_query_tables > 0:
        query_tables = query_tables[: args.max_query_tables]

    valid_df = valid_df[valid_df["table_name_1"].isin(query_tables)]
    test_df = test_df[test_df["table_name_1"].isin(query_tables)]

    datalake_tables = list(
        dict.fromkeys(list(valid_df["table_name_2"]) + list(test_df["table_name_2"]) + query_tables)
    )
    if args.max_datalake_tables > 0:
        datalake_tables = datalake_tables[: args.max_datalake_tables]
    datalake_set = set(datalake_tables)
    valid_df = valid_df[valid_df["table_name_2"].isin(datalake_set)]
    test_df = test_df[test_df["table_name_2"].isin(datalake_set)]

    base = os.path.join(args.output_root, "santos")
    query_dir = os.path.join(base, "query")
    datalake_dir = os.path.join(base, "datalake")
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(datalake_dir, exist_ok=True)

    def symlink_table(table_name: str, dst_dir: str) -> bool:
        src = os.path.join(datalake_plus, table_name)
        dst = os.path.join(dst_dir, table_name)
        if not os.path.exists(src):
            return False
        if os.path.lexists(dst):
            return True
        os.symlink(src, dst)
        return True

    linked_query = 0
    for table_name in query_tables:
        if symlink_table(table_name, query_dir):
            linked_query += 1

    linked_datalake = 0
    for table_name in datalake_tables:
        if symlink_table(table_name, datalake_dir):
            linked_datalake += 1

    valid_out = os.path.join(base, "valid_pairs.csv")
    test_out = os.path.join(base, "test_pairs.csv")
    valid_df.to_csv(valid_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(
        json.dumps(
            {
                "output_root": args.output_root,
                "query_tables": linked_query,
                "datalake_tables": linked_datalake,
                "valid_pairs": len(valid_df),
                "test_pairs": len(test_df),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
