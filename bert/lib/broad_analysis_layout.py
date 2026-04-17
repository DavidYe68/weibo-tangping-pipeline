from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from lib.io_utils import save_json


CANONICAL_TOPIC_MODEL_DIR = "topic_model_BAAI"
CANONICAL_SEMANTIC_DIR = "semantic_analysis"
CANONICAL_DRIFT_DIR = "drift_analysis"


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def iter_snapshot_dirs(root: Path, group: str) -> list[Path]:
    snapshot_root = root / "snapshots" / group
    if not snapshot_root.is_dir():
        return []
    return sorted([path for path in snapshot_root.iterdir() if path.is_dir()], key=lambda item: item.name)


def latest_snapshot_dir(root: Path, group: str) -> Path | None:
    snapshot_dirs = iter_snapshot_dirs(root, group)
    if not snapshot_dirs:
        return None
    return snapshot_dirs[-1]


def copy_output_bundle(source_dir: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in sorted(source_dir.iterdir(), key=lambda item: item.name):
        destination = target_dir / child.name
        if child.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)
    return target_dir


def ensure_canonical_output_from_latest_snapshot(
    root: Path,
    *,
    snapshot_group: str,
    canonical_dir_name: str,
) -> Path | None:
    canonical_dir = root / canonical_dir_name
    if canonical_dir.is_dir():
        return canonical_dir

    snapshot_dir = latest_snapshot_dir(root, snapshot_group)
    if snapshot_dir is None:
        return None
    return copy_output_bundle(snapshot_dir, canonical_dir)


def sync_semantic_output_metadata(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "semantic_analysis_summary.json"
    if not summary_path.exists():
        return {}

    summary = _safe_read_json(summary_path)
    summary.update(
        {
            "output_dir": str(output_dir.resolve()),
            "cooccurrence_path": str((output_dir / "keyword_cooccurrence.csv").resolve()),
            "semantic_neighbors_path": str((output_dir / "keyword_semantic_neighbors.csv").resolve()),
            "tokenized_analysis_base_path": str((output_dir / "tokenized_analysis_base.parquet").resolve()),
        }
    )
    save_json(summary_path, summary)
    return summary


def sync_drift_output_metadata(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "drift_analysis_summary.json"
    if not summary_path.exists():
        return {}

    summary = _safe_read_json(summary_path)
    summary.update(
        {
            "output_dir": str(output_dir.resolve()),
            "collocation_drift_path": str((output_dir / "keyword_collocation_drift.csv").resolve()),
            "neighbor_drift_path": str((output_dir / "keyword_neighbor_drift.csv").resolve()),
            "topic_drift_path": str((output_dir / "topic_drift_by_keyword.csv").resolve()),
            "topic_share_change_path": str((output_dir / "topic_share_change_by_keyword.csv").resolve()),
            "overall_topic_drift_path": str((output_dir / "topic_drift_overall.csv").resolve()),
            "overall_topic_change_path": str((output_dir / "topic_share_change_overall.csv").resolve()),
            "topic_drift_by_ip_path": str((output_dir / "topic_drift_by_ip.csv").resolve()),
            "topic_share_change_by_ip_path": str((output_dir / "topic_share_change_by_ip.csv").resolve()),
            "topic_drift_by_ip_and_keyword_path": str((output_dir / "topic_drift_by_ip_and_keyword.csv").resolve()),
            "topic_share_change_by_ip_and_keyword_path": str(
                (output_dir / "topic_share_change_by_ip_and_keyword.csv").resolve()
            ),
        }
    )
    save_json(summary_path, summary)
    return summary


def sync_all_analysis_output_metadata(root: Path) -> None:
    for path in [root / CANONICAL_SEMANTIC_DIR, *iter_snapshot_dirs(root, "semantic")]:
        if path.is_dir():
            sync_semantic_output_metadata(path)
    for path in [root / CANONICAL_DRIFT_DIR, *iter_snapshot_dirs(root, "drift")]:
        if path.is_dir():
            sync_drift_output_metadata(path)
