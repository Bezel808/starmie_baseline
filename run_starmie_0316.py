#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


STARMIE = Path(__file__).resolve().parent
LOCAL_CONVERT = STARMIE / "convert_1218_unionable_to_starmie_uts.py"
LOCAL_EVAL = STARMIE / "eval_starmie_uts_threshold.py"
DEFAULT_DATASET_ROOT = os.environ.get("STARMIE_DATASET_ROOT", "").strip()

DATASETS = {
    "wikidbs_1218": "wikidbs_1218",
    "santos_benchmark_1218": "santos_benchmark_1218",
    "magellan_1218": "magellan_1218",
}


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append_log(log_path: Path, dataset: str, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_ts()}] [DATASET={dataset}] {line}")


def run_cmd(
    cmd: list[str],
    dataset: str,
    log_path: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[str, float]:
    _append_log(log_path, dataset, f"[RUN] {' '.join(cmd)}\n")
    t0 = time.perf_counter()
    out_lines: list[str] = []
    with subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as p:
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            out_lines.append(line)
            _append_log(log_path, dataset, line)
        rc = p.wait()
    elapsed = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"Command failed with exit={rc}: {' '.join(cmd)}")
    _append_log(log_path, dataset, f"[DONE] elapsed={elapsed:.3f}s\n")
    return "".join(out_lines), elapsed


def parse_last_json(stdout: str) -> dict:
    starts = [i for i, ch in enumerate(stdout) if ch == "{"]
    for i in reversed(starts):
        blob = stdout[i:].strip()
        try:
            return json.loads(blob)
        except Exception:
            continue
    raise ValueError("Cannot parse JSON object from command output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Starmie UTS baseline on *_1218 datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(DEFAULT_DATASET_ROOT) if DEFAULT_DATASET_ROOT else None,
        help="Root containing *_1218 datasets. Can also be set by STARMIE_DATASET_ROOT.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=STARMIE / "runs" / "uts_1218_full",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(os.environ.get("STARMIE_PYTHON", sys.executable)),
        help="Python executable used to run sub-commands (default: current interpreter).",
    )
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--augment-op", type=str, default="drop_col")
    parser.add_argument("--sample-meth", type=str, default="tfidf_entity")
    parser.add_argument("--table-order", type=str, default="column")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_root is None:
        raise ValueError(
            "Missing dataset root. Pass --dataset-root or set STARMIE_DATASET_ROOT."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}

    base_env = os.environ.copy()
    base_env["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Avoid flaky network calls to HF Hub during repeated tokenizer/model loads.
    base_env["TRANSFORMERS_OFFLINE"] = "1"
    base_env["HF_HUB_OFFLINE"] = "1"

    for run_id, ds_name in enumerate(args.datasets):
        ds_start = time.perf_counter()
        ds_root = args.dataset_root / DATASETS[ds_name]
        run_root = args.output_root / ds_name
        log_path = logs_dir / f"{ds_name}.log"
        if log_path.exists():
            log_path.unlink()

        nvsmi_out, _ = run_cmd(["nvidia-smi"], ds_name, log_path, env=base_env)

        conv_cmd = [
            str(args.python_bin),
            str(LOCAL_CONVERT),
            "--dataset-root",
            str(ds_root),
            "--output-root",
            str(run_root),
            "--max-valid-pairs",
            "-1",
            "--max-test-pairs",
            "-1",
            "--max-query-tables",
            "-1",
            "--max-datalake-tables",
            "-1",
        ]
        conv_out, convert_sec = run_cmd(conv_cmd, ds_name, log_path, env=base_env)
        _ = parse_last_json(conv_out)

        run_env = base_env.copy()
        run_env["STARMIE_DATA_ROOT"] = str(run_root)

        pretrain_cmd = [
            str(args.python_bin),
            "run_pretrain.py",
            "--task",
            "santos",
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--lm",
            "roberta",
            "--n_epochs",
            str(args.n_epochs),
            "--max_len",
            str(args.max_len),
            "--size",
            str(args.pretrain_size),
            "--projector",
            "768",
            "--save_model",
            "--augment_op",
            args.augment_op,
            "--sample_meth",
            args.sample_meth,
            "--table_order",
            args.table_order,
            "--run_id",
            str(run_id),
        ]
        if args.fp16:
            pretrain_cmd.append("--fp16")
        _, pretrain_sec = run_cmd(pretrain_cmd, ds_name, log_path, cwd=STARMIE, env=run_env)

        extract_cmd = [
            str(args.python_bin),
            "extractVectors.py",
            "--benchmark",
            "santos",
            "--table_order",
            args.table_order,
            "--run_id",
            str(run_id),
            "--save_model",
        ]
        (run_root / "santos" / "vectors").mkdir(parents=True, exist_ok=True)
        _, extract_sec = run_cmd(extract_cmd, ds_name, log_path, cwd=STARMIE, env=run_env)

        vec_dir = run_root / "santos" / "vectors"
        eval_cmd = [
            str(args.python_bin),
            str(LOCAL_EVAL),
            "--query-pkl",
            str(vec_dir / f"cl_query_{args.augment_op}_{args.sample_meth}_{args.table_order}_{run_id}.pkl"),
            "--datalake-pkl",
            str(vec_dir / f"cl_datalake_{args.augment_op}_{args.sample_meth}_{args.table_order}_{run_id}.pkl"),
            "--valid-csv",
            str(run_root / "santos" / "valid_pairs.csv"),
            "--test-csv",
            str(run_root / "santos" / "test_pairs.csv"),
        ]
        eval_out, eval_sec = run_cmd(eval_cmd, ds_name, log_path, env=run_env)
        eval_json = parse_last_json(eval_out)

        total_sec = time.perf_counter() - ds_start
        summary[ds_name] = {
            "dataset": ds_name,
            "best_threshold": eval_json["valid_threshold"],
            "test_metrics": eval_json["test_metrics"],
            "timing_seconds": {
                "convert": round(convert_sec, 3),
                "pretrain": round(pretrain_sec, 3),
                "extract": round(extract_sec, 3),
                "eval": round(eval_sec, 3),
                "total": round(total_sec, 3),
            },
            "log_path": str(log_path),
            "nvidia_smi_head": nvsmi_out.splitlines()[:8],
        }

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[{now_ts()}] [DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
