#!/bin/bash
set -euo pipefail

ROOT="/Users/apple/Local/fdurop/code/result"
cd "$ROOT"

COMPARE_ROOT="bert/artifacts/broad_analysis/topic_model_compare"
SHARED_CHECKPOINT_DIR="$COMPARE_ROOT/_shared_checkpoints"
LOG_PATH="$COMPARE_ROOT/continuation.log"
REVIEW_PATH="$COMPARE_ROOT/review_log.txt"
LOCKDIR="$COMPARE_ROOT/.continuation.lock"

RUN1_DIR="$COMPARE_ROOT/S_m650_s20"
RUN2_DIR="$COMPARE_ROOT/A_auto_clean"
RUN3_DIR="$COMPARE_ROOT/G_m700_u150"

mkdir -p "$COMPARE_ROOT"

if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "Continuation lock exists at $LOCKDIR; exiting."
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
print(f"outlier_share: {float(outlier_count / doc_count) if doc_count else 0.0:.6f}")
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

evaluate_run() {
  local run_dir="$1"
  .venv/bin/python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

run_dir = Path(sys.argv[1])
summary = json.loads((run_dir / "topic_model_summary.json").read_text(encoding="utf-8"))
topic_info = pd.read_csv(summary["topic_info_path"])
topic_info["Topic"] = pd.to_numeric(topic_info["Topic"], errors="coerce")
topic_info["Count"] = pd.to_numeric(topic_info["Count"], errors="coerce").fillna(0).astype(int)
non = topic_info[topic_info["Topic"] >= 0].copy()
clustered = int(non["Count"].sum())
top10_docs = int(non.sort_values("Count", ascending=False).head(10)["Count"].sum()) if clustered else 0
top10_share = float(top10_docs / clustered) if clustered else 0.0
topic_count = int(summary.get("topic_count_excluding_outliers", 0))
outlier_share = float(summary.get("outlier_share", 1.0))

if outlier_share > 0.05:
    print("too_many_outliers")
elif topic_count >= 80 and topic_count <= 220 and top10_share <= 0.35:
    print("good_stop")
elif topic_count > 220:
    print("still_fragmented")
elif topic_count < 25 or top10_share > 0.60:
    print("overmerged")
else:
    print("borderline")
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

log "Continuation started"

run_topic_model "$RUN1_DIR" "Run1_S_m650_s20" \
  --min_topic_size 650 \
  --hdbscan_min_samples 20 \
  --umap_n_neighbors 120 \
  --outlier_reduction_strategy c-tf-idf+distributions \
  --outlier_reduction_threshold 0.00

run1_status="$(evaluate_run "$RUN1_DIR")"
log "Run1_S_m650_s20 evaluation: $run1_status"

if [[ "$run1_status" == "good_stop" ]]; then
  log "Run 1 is good enough; stopping to avoid overfitting"
  exit 0
fi

if [[ "$run1_status" == "still_fragmented" || "$run1_status" == "borderline" ]]; then
  run_topic_model "$RUN2_DIR" "Run2_A_auto_clean" \
    --min_topic_size 500 \
    --hdbscan_min_samples 25 \
    --umap_n_neighbors 120 \
    --nr_topics auto \
    --outlier_reduction_strategy c-tf-idf+distributions \
    --outlier_reduction_threshold 0.00

  run2_status="$(evaluate_run "$RUN2_DIR")"
  log "Run2_A_auto_clean evaluation: $run2_status"

  if [[ "$run2_status" == "good_stop" || "$run2_status" == "borderline" ]]; then
    log "Run 2 is acceptable; stopping here"
    exit 0
  fi
fi

run_topic_model "$RUN3_DIR" "Run3_G_m700_u150" \
  --min_topic_size 700 \
  --hdbscan_min_samples 25 \
  --umap_n_neighbors 150 \
  --outlier_reduction_strategy c-tf-idf+distributions \
  --outlier_reduction_threshold 0.00

run3_status="$(evaluate_run "$RUN3_DIR")"
log "Run3_G_m700_u150 evaluation: $run3_status"
log "Continuation completed"
