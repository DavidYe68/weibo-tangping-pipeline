#!/bin/bash
set -euo pipefail

ROOT="/Users/apple/Local/fdurop/code/result"
cd "$ROOT"

COMPARE_ROOT="bert/artifacts/broad_analysis/topic_model_compare"
SHARED_CHECKPOINT_DIR="$COMPARE_ROOT/_shared_checkpoints"
LOG_PATH="$COMPARE_ROOT/autopilot.log"
REVIEW_PATH="$COMPARE_ROOT/review_log.txt"
LOCKDIR="$COMPARE_ROOT/.autopilot.lock"

RUN1_DIR="$COMPARE_ROOT/O_outlier"
RUN2_DIR="$COMPARE_ROOT/N_nr40"
RUN3_DIR="$COMPARE_ROOT/N_nr25"
RUN4_DIR="$COMPARE_ROOT/F_auto_m800_s10"

mkdir -p "$COMPARE_ROOT"

if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "Autopilot lock exists at $LOCKDIR; exiting."
  exit 0
fi
trap 'rmdir "$LOCKDIR"' EXIT

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOG_PATH"
}

inspect_run() {
  local run_dir="$1"
  local label="$2"
  .venv/bin/python - "$run_dir" "$label" <<'PY' >> "$REVIEW_PATH"
import json
import sys
from pathlib import Path

import pandas as pd

run_dir = Path(sys.argv[1])
label = sys.argv[2]
summary_path = run_dir / "topic_model_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
topic_info_path = Path(summary["topic_info_path"])
topic_overview_path = Path(summary["topic_overview_path"])

topic_info = pd.read_csv(topic_info_path)
topic_info["Topic"] = pd.to_numeric(topic_info["Topic"], errors="coerce")
topic_info["Count"] = pd.to_numeric(topic_info["Count"], errors="coerce").fillna(0).astype(int)
non = topic_info[topic_info["Topic"] >= 0].copy()
outlier_count = int(topic_info.loc[topic_info["Topic"] == -1, "Count"].sum())
doc_count = int(summary["document_count"])
top10_docs = int(non.sort_values("Count", ascending=False).head(10)["Count"].sum()) if not non.empty else 0
top10_share_clustered = float(top10_docs / non["Count"].sum()) if not non.empty else 0.0

topic_overview = pd.read_csv(topic_overview_path)
head = topic_overview.head(8)[["topic_id", "topic_count", "top_terms"]]

print(f"\n=== {label} ===")
print(f"summary_path: {summary_path}")
print(f"document_count: {doc_count}")
print(f"outlier_count: {outlier_count}")
print(f"outlier_share: {summary.get('outlier_share', 0.0):.6f}")
print(f"topic_count_excluding_outliers: {int(summary.get('topic_count_excluding_outliers', 0))}")
print(f"initial_outlier_document_count: {int(summary.get('initial_outlier_document_count', 0))}")
print(f"initial_outlier_share: {float(summary.get('initial_outlier_share', 0.0)):.6f}")
print(f"outlier_reduction_applied: {bool(summary.get('outlier_reduction_applied', False))}")
print(f"top10_share_clustered: {top10_share_clustered:.6f}")
print("top_topics:")
for row in head.itertuples(index=False):
    print(f"  topic={int(row.topic_id):>3} count={int(row.topic_count):>6} terms={row.top_terms}")
PY
}

should_run_fallback() {
  .venv/bin/python - "$RUN1_DIR" "$RUN2_DIR" "$RUN3_DIR" <<'PY'
import json
import sys
from pathlib import Path

def load_summary(run_dir: str):
    path = Path(run_dir) / "topic_model_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

good = False
for run_dir in sys.argv[1:]:
    summary = load_summary(run_dir)
    if not summary:
        continue
    outlier_share = float(summary.get("outlier_share", 1.0))
    topic_count = int(summary.get("topic_count_excluding_outliers", 0))
    if outlier_share <= 0.45 and 8 <= topic_count <= 15:
        good = True
        break

print("no" if good else "yes")
PY
}

run_topic_model() {
  local run_dir="$1"
  local label="$2"
  shift 2

  log "Starting ${label}: $*"
  caffeinate -dimsu .venv/bin/python bert/08_topic_model_bertopic.py \
    --output_dir "$run_dir" \
    --checkpoint_dir "$SHARED_CHECKPOINT_DIR" \
    --resume \
    "$@" | tee -a "$LOG_PATH"
  log "Finished ${label}"
  inspect_run "$run_dir" "$label"
}

wait_for_existing_run1() {
  local summary_path="$RUN1_DIR/topic_model_summary.json"
  local run1_pattern="bert/08_topic_model_bertopic.py.*--output_dir $RUN1_DIR"
  log "Waiting for current Run 1 to finish: $RUN1_DIR"
  while [[ ! -f "$summary_path" ]]; do
    if ! pgrep -af "$run1_pattern" >/dev/null; then
      log "Run 1 summary missing and no active Run 1 process detected; restarting Run 1"
      run_topic_model "$RUN1_DIR" "Run1_O_outlier" \
        --min_topic_size 500 \
        --hdbscan_min_samples 25 \
        --umap_n_neighbors 120 \
        --outlier_reduction_strategy c-tf-idf+distributions \
        --outlier_reduction_threshold 0.00
      return
    fi
    sleep 60
  done
  log "Detected completed Run 1 summary"
  inspect_run "$RUN1_DIR" "Run1_O_outlier"
}

log "Autopilot started"
wait_for_existing_run1

run_topic_model "$RUN2_DIR" "Run2_N_nr40" \
  --min_topic_size 500 \
  --hdbscan_min_samples 25 \
  --umap_n_neighbors 120 \
  --nr_topics 40 \
  --outlier_reduction_strategy c-tf-idf+distributions \
  --outlier_reduction_threshold 0.00

run_topic_model "$RUN3_DIR" "Run3_N_nr25" \
  --min_topic_size 500 \
  --hdbscan_min_samples 25 \
  --umap_n_neighbors 120 \
  --nr_topics 25 \
  --outlier_reduction_strategy c-tf-idf+distributions \
  --outlier_reduction_threshold 0.00

if [[ "$(should_run_fallback)" == "yes" ]]; then
  log "No run met the target window (8-15 topics with outlier_share <= 0.45); starting fallback run"
  run_topic_model "$RUN4_DIR" "Run4_F_auto_m800_s10" \
    --min_topic_size 800 \
    --hdbscan_min_samples 10 \
    --umap_n_neighbors 120 \
    --nr_topics auto \
    --outlier_reduction_strategy c-tf-idf+distributions \
    --outlier_reduction_threshold 0.05
else
  log "At least one run met the target window; skipping fallback run"
fi

log "Autopilot completed all scheduled runs"
