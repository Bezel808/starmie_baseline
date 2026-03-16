# CHANGES

## 2026-03-16
- Added configurable data root support.
- Path source: `STARMIE_DATA_ROOT` (optional) with legacy fallback.
- Updated files:
  - `run_pretrain.py`: resolve pretrain table path via `STARMIE_DATA_ROOT`.
  - `sdd/pretrain.py`: resolve load/eval data paths via `STARMIE_DATA_ROOT`.
  - `sdd/baselines.py`: added AdamW import fallback for newer transformers.
  - `extractVectors.py`: resolve input/output vector paths via `STARMIE_DATA_ROOT`.
  - `test_naive_search.py`, `test_lsh.py`, `test_hnsw_search.py`: resolve query/table/index/groundtruth paths via `STARMIE_DATA_ROOT`.
  - `checkPrecisionRecall.py`: fallback to stdlib `pickle` when `pickle5` is unavailable.
- Compatibility:
  - Without `STARMIE_DATA_ROOT`, all scripts keep original `data/...` behavior.

- Evaluation update: validation-set threshold search (best F1) and fixed-threshold test metrics (F1/Precision/Recall/Accuracy/AUC) in `sdd/utils.py` and `sdd/baselines.py`.

- Added `run_starmie_0316.py` for reproducible 1218 UTS full runs (wikidbs_1218, santos_benchmark_1218, magellan_1218):
  - Enforces stage order: convert -> pretrain -> extract -> eval.
  - Uses full valid/test pairs (`-1` limits) and tunes threshold on valid then reports fixed-threshold test metrics.
  - Writes per-dataset logs with timestamp + `[DATASET=...]` prefixes.
  - Writes `summary.json` with dataset, best threshold, test metrics, phase timings (`convert/pretrain/extract/eval/total`), and log path.
  - Fixed extract stage robustness by pre-creating `santos/vectors` before `extractVectors.py` writes output pickle files.
  - Added offline-HF env guards (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`) in runner to avoid checkpoint/tokenizer fetch failures during extraction.
  - Tuned runner defaults toward original Starmie settings for better quality:
    - `n_epochs`: 1 -> 3
    - `batch_size`: 16 -> 64
    - `pretrain size`: 200 -> 10000
    - enabled `--fp16` by default
  - Exposed pretrain/search-critical knobs via CLI:
    - `--pretrain-size`, `--lr`, `--max-len`, `--augment-op`, `--sample-meth`, `--table-order`, `--[no-]fp16`
  - Made eval vector file names follow selected `augment/sample/table_order` instead of fixed `drop_col/tfidf_entity/column`.
  - Added local `convert_1218_unionable_to_starmie_uts.py` under `starmie_baseline` and switched `run_starmie_0316.py` to use this local converter (removes external `dataset_tools` converter dependency).
  - Added local `eval_starmie_uts_threshold.py` under `starmie_baseline` and switched `run_starmie_0316.py` to use this local evaluator (removes external `dataset_tools` evaluator dependency).
