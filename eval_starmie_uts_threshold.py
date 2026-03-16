#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd
from munkres import DISALLOWED, Munkres, make_cost_matrix
from numpy.linalg import norm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = norm(v1) * norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def table_score(table1, table2, edge_threshold: float = 0.0) -> float:
    nrow = len(table1)
    ncol = len(table2)
    graph = np.zeros((nrow, ncol), dtype=float)
    for i in range(nrow):
        for j in range(ncol):
            sim = cosine_sim(table1[i], table2[j])
            if sim > edge_threshold:
                graph[i, j] = sim
    max_graph = make_cost_matrix(
        graph,
        lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED,
    )
    indexes = Munkres().compute(max_graph)
    score = 0.0
    for r, c in indexes:
        score += graph[r, c]
    return score


def compute_metrics(y_true, y_score, threshold: float) -> dict:
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = roc_auc_score(y_true, y_score)
    else:
        out["auc"] = float("nan")
    return out


def best_threshold_by_f1(y_true, y_score) -> tuple[float, dict]:
    best_t = 0.0
    best_f1 = -1.0
    best_metrics = None
    low = float(np.min(y_score))
    high = float(np.max(y_score))
    for t in np.linspace(low, high, 501):
        m = compute_metrics(y_true, y_score, float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
            best_metrics = m
    return best_t, best_metrics


def build_split_scores(df: pd.DataFrame, query_emb_map: dict, dl_emb_map: dict) -> tuple[list, list]:
    y_true, y_score = [], []
    for _, row in df.iterrows():
        q = str(row["table_name_1"])
        d = str(row["table_name_2"])
        y = int(row["label"])
        if q not in query_emb_map or d not in dl_emb_map:
            continue
        s = table_score(query_emb_map[q], dl_emb_map[d], edge_threshold=0.0)
        y_true.append(y)
        y_score.append(s)
    return y_true, y_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune threshold on valid and evaluate on test for Starmie UTS."
    )
    parser.add_argument("--query-pkl", required=True)
    parser.add_argument("--datalake-pkl", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    args = parser.parse_args()

    query_vecs = pickle.load(open(args.query_pkl, "rb"))
    dl_vecs = pickle.load(open(args.datalake_pkl, "rb"))
    query_emb_map = {k: v for k, v in query_vecs}
    dl_emb_map = {k: v for k, v in dl_vecs}

    valid_df = pd.read_csv(args.valid_csv)
    test_df = pd.read_csv(args.test_csv)

    vy, vs = build_split_scores(valid_df, query_emb_map, dl_emb_map)
    ty, ts = build_split_scores(test_df, query_emb_map, dl_emb_map)

    threshold, valid_metrics = best_threshold_by_f1(vy, vs)
    test_metrics = compute_metrics(ty, ts, threshold)

    print(
        json.dumps(
            {
                "valid_threshold": threshold,
                "valid_metrics": valid_metrics,
                "test_metrics": test_metrics,
                "valid_pairs_used": len(vy),
                "test_pairs_used": len(ty),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
