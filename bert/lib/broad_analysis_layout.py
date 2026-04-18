from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from lib.io_utils import save_json


CANONICAL_TOPIC_MODEL_DIR = "topic_model_BAAI"
CANONICAL_SEMANTIC_DIR = "semantic_analysis"
CANONICAL_DRIFT_DIR = "drift_analysis"
READOUTS_DIRNAME = "readouts"
VIZ_INPUTS_DIRNAME = "viz_inputs"

TOPIC_MODEL_READOUT_FILES = (
    "topic_info.csv",
    "topic_overview.csv",
    "topic_terms.csv",
    "topic_share_by_period.csv",
    "topic_share_by_period_and_keyword.csv",
)
TOPIC_MODEL_VIZ_FILES = (
    "document_topics.parquet",
    "topic_share_by_ip.csv",
    "topic_share_by_period_and_ip.csv",
    "topic_share_by_period_and_ip_and_keyword.csv",
)
SEMANTIC_VIZ_FILES = (
    "keyword_cooccurrence.csv",
    "keyword_semantic_neighbors.csv",
    "tokenized_analysis_base.parquet",
)
SEMANTIC_READOUT_FILES = (
    "semantic_bucket_override_template.csv",
    "semantic_context_bucket_summary.csv",
    "semantic_context_shift_summary.csv",
    "semantic_context_trajectory.csv",
    "semantic_keyword_overview.csv",
    "semantic_midterm_candidates.csv",
    "semantic_midterm_coding_template.csv",
    "semantic_midterm_notes.md",
    "semantic_midterm_operation_log.md",
    "semantic_midterm_summary.json",
    "semantic_noise_diagnostics.csv",
    "semantic_period_overview.csv",
    "semantic_period_shortlist.csv",
)
SEMANTIC_READOUT_SUBDIRS = {
    "semantic_midterm_notes.md": "01_start_here",
    "semantic_keyword_overview.csv": "01_start_here",
    "semantic_context_trajectory.csv": "01_start_here",
    "semantic_context_shift_summary.csv": "01_start_here",
    "semantic_period_shortlist.csv": "02_period_detail",
    "semantic_period_overview.csv": "02_period_detail",
    "semantic_context_bucket_summary.csv": "02_period_detail",
    "semantic_bucket_override_template.csv": "03_workbench",
    "semantic_midterm_candidates.csv": "03_workbench",
    "semantic_midterm_coding_template.csv": "03_workbench",
    "semantic_noise_diagnostics.csv": "03_workbench",
    "semantic_midterm_summary.json": "99_meta",
    "semantic_midterm_operation_log.md": "99_meta",
}
DRIFT_READOUT_FILES = (
    "keyword_collocation_drift.csv",
    "keyword_neighbor_drift.csv",
    "topic_drift_by_keyword.csv",
    "topic_drift_overall.csv",
)
DRIFT_VIZ_FILES = (
    "topic_share_change_by_keyword.csv",
    "topic_share_change_overall.csv",
    "topic_drift_by_ip.csv",
    "topic_share_change_by_ip.csv",
    "topic_drift_by_ip_and_keyword.csv",
    "topic_share_change_by_ip_and_keyword.csv",
)


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _existing_or_preferred(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def topic_model_output_paths(output_dir: Path) -> dict[str, Path]:
    readouts_dir = output_dir / READOUTS_DIRNAME
    viz_inputs_dir = output_dir / VIZ_INPUTS_DIRNAME
    return {
        "output_dir": output_dir,
        "readouts_dir": readouts_dir,
        "viz_inputs_dir": viz_inputs_dir,
        "checkpoint_dir": output_dir / "checkpoints",
        "summary_path": output_dir / "topic_model_summary.json",
        "documents_path": viz_inputs_dir / "document_topics.parquet",
        "topic_info_path": readouts_dir / "topic_info.csv",
        "topic_overview_path": readouts_dir / "topic_overview.csv",
        "topic_terms_path": readouts_dir / "topic_terms.csv",
        "topic_share_by_period_path": readouts_dir / "topic_share_by_period.csv",
        "topic_share_by_period_and_keyword_path": readouts_dir / "topic_share_by_period_and_keyword.csv",
        "topic_share_by_ip_path": viz_inputs_dir / "topic_share_by_ip.csv",
        "topic_share_by_period_and_ip_path": viz_inputs_dir / "topic_share_by_period_and_ip.csv",
        "topic_share_by_period_and_ip_and_keyword_path": viz_inputs_dir / "topic_share_by_period_and_ip_and_keyword.csv",
        "model_dir": viz_inputs_dir / "model",
    }


def semantic_output_paths(output_dir: Path) -> dict[str, Path]:
    readouts_dir = output_dir / READOUTS_DIRNAME
    viz_inputs_dir = output_dir / VIZ_INPUTS_DIRNAME
    return {
        "output_dir": output_dir,
        "readouts_dir": readouts_dir,
        "viz_inputs_dir": viz_inputs_dir,
        "summary_path": output_dir / "semantic_analysis_summary.json",
        "cooccurrence_path": viz_inputs_dir / "keyword_cooccurrence.csv",
        "semantic_neighbors_path": viz_inputs_dir / "keyword_semantic_neighbors.csv",
        "tokenized_analysis_base_path": viz_inputs_dir / "tokenized_analysis_base.parquet",
    }


def semantic_readout_path(output_dir: Path, name: str) -> Path:
    readouts_dir = output_dir if output_dir.name == READOUTS_DIRNAME else semantic_output_paths(output_dir)["readouts_dir"]
    subdir = SEMANTIC_READOUT_SUBDIRS.get(name)
    if subdir:
        return readouts_dir / subdir / name
    return readouts_dir / name


def drift_output_paths(output_dir: Path) -> dict[str, Path]:
    readouts_dir = output_dir / READOUTS_DIRNAME
    viz_inputs_dir = output_dir / VIZ_INPUTS_DIRNAME
    return {
        "output_dir": output_dir,
        "readouts_dir": readouts_dir,
        "viz_inputs_dir": viz_inputs_dir,
        "summary_path": output_dir / "drift_analysis_summary.json",
        "collocation_drift_path": readouts_dir / "keyword_collocation_drift.csv",
        "neighbor_drift_path": readouts_dir / "keyword_neighbor_drift.csv",
        "topic_drift_path": readouts_dir / "topic_drift_by_keyword.csv",
        "overall_topic_drift_path": readouts_dir / "topic_drift_overall.csv",
        "topic_share_change_path": viz_inputs_dir / "topic_share_change_by_keyword.csv",
        "overall_topic_change_path": viz_inputs_dir / "topic_share_change_overall.csv",
        "topic_drift_by_ip_path": viz_inputs_dir / "topic_drift_by_ip.csv",
        "topic_share_change_by_ip_path": viz_inputs_dir / "topic_share_change_by_ip.csv",
        "topic_drift_by_ip_and_keyword_path": viz_inputs_dir / "topic_drift_by_ip_and_keyword.csv",
        "topic_share_change_by_ip_and_keyword_path": viz_inputs_dir / "topic_share_change_by_ip_and_keyword.csv",
    }


def resolve_topic_model_artifact(output_dir: Path, name: str) -> Path:
    paths = topic_model_output_paths(output_dir)
    if name == "topic_model_summary.json":
        return paths["summary_path"]
    if name == "checkpoints":
        return paths["checkpoint_dir"]
    if name == "model":
        return _existing_or_preferred([paths["model_dir"], output_dir / "model"])
    if name in TOPIC_MODEL_READOUT_FILES:
        return _existing_or_preferred([paths["readouts_dir"] / name, output_dir / name])
    if name in TOPIC_MODEL_VIZ_FILES:
        return _existing_or_preferred([paths["viz_inputs_dir"] / name, output_dir / name])
    return output_dir / name


def resolve_semantic_artifact(output_dir: Path, name: str) -> Path:
    paths = semantic_output_paths(output_dir)
    if name == "semantic_analysis_summary.json":
        return paths["summary_path"]
    if name == "midterm_bundle":
        return _existing_or_preferred([paths["readouts_dir"], output_dir / "midterm_bundle"])
    if name in SEMANTIC_READOUT_FILES:
        return _existing_or_preferred(
            [
                semantic_readout_path(output_dir, name),
                paths["readouts_dir"] / name,
                paths["readouts_dir"] / "midterm_bundle" / name,
                output_dir / name,
            ]
        )
    if name in SEMANTIC_VIZ_FILES:
        return _existing_or_preferred([paths["viz_inputs_dir"] / name, output_dir / name])
    return output_dir / name


def resolve_drift_artifact(output_dir: Path, name: str) -> Path:
    paths = drift_output_paths(output_dir)
    if name == "drift_analysis_summary.json":
        return paths["summary_path"]
    if name in DRIFT_READOUT_FILES:
        return _existing_or_preferred([paths["readouts_dir"] / name, output_dir / name])
    if name in DRIFT_VIZ_FILES:
        return _existing_or_preferred([paths["viz_inputs_dir"] / name, output_dir / name])
    return output_dir / name


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
    paths = semantic_output_paths(output_dir)
    summary_path = paths["summary_path"]
    if not summary_path.exists():
        return {}

    summary = _safe_read_json(summary_path)
    summary.pop("midterm_bundle_dir", None)
    summary.update(
        {
            "output_dir": str(output_dir.resolve()),
            "readouts_dir": str(paths["readouts_dir"].resolve()),
            "viz_inputs_dir": str(paths["viz_inputs_dir"].resolve()),
            "readout_groups": {
                name: str(semantic_readout_path(output_dir, name).resolve())
                for name in SEMANTIC_READOUT_FILES
            },
            "cooccurrence_path": str(paths["cooccurrence_path"].resolve()),
            "semantic_neighbors_path": str(paths["semantic_neighbors_path"].resolve()),
            "tokenized_analysis_base_path": str(paths["tokenized_analysis_base_path"].resolve()),
        }
    )
    save_json(summary_path, summary)
    return summary


def sync_drift_output_metadata(output_dir: Path) -> dict[str, Any]:
    paths = drift_output_paths(output_dir)
    summary_path = paths["summary_path"]
    if not summary_path.exists():
        return {}

    summary = _safe_read_json(summary_path)
    summary.update(
        {
            "output_dir": str(output_dir.resolve()),
            "readouts_dir": str(paths["readouts_dir"].resolve()),
            "viz_inputs_dir": str(paths["viz_inputs_dir"].resolve()),
            "collocation_drift_path": str(paths["collocation_drift_path"].resolve()),
            "neighbor_drift_path": str(paths["neighbor_drift_path"].resolve()),
            "topic_drift_path": str(paths["topic_drift_path"].resolve()),
            "topic_share_change_path": str(paths["topic_share_change_path"].resolve()),
            "overall_topic_drift_path": str(paths["overall_topic_drift_path"].resolve()),
            "overall_topic_change_path": str(paths["overall_topic_change_path"].resolve()),
            "topic_drift_by_ip_path": str(paths["topic_drift_by_ip_path"].resolve()),
            "topic_share_change_by_ip_path": str(paths["topic_share_change_by_ip_path"].resolve()),
            "topic_drift_by_ip_and_keyword_path": str(paths["topic_drift_by_ip_and_keyword_path"].resolve()),
            "topic_share_change_by_ip_and_keyword_path": str(paths["topic_share_change_by_ip_and_keyword_path"].resolve()),
        }
    )
    save_json(summary_path, summary)
    return summary


def sync_topic_model_output_metadata(output_dir: Path) -> dict[str, Any]:
    paths = topic_model_output_paths(output_dir)
    summary_path = paths["summary_path"]
    if not summary_path.exists():
        return {}

    summary = _safe_read_json(summary_path)
    summary.update(
        {
            "output_dir": str(output_dir.resolve()),
            "readouts_dir": str(paths["readouts_dir"].resolve()),
            "viz_inputs_dir": str(paths["viz_inputs_dir"].resolve()),
            "checkpoint_dir": str(paths["checkpoint_dir"].resolve()),
            "documents_path": str(paths["documents_path"].resolve()),
            "topic_info_path": str(paths["topic_info_path"].resolve()),
            "topic_overview_path": str(paths["topic_overview_path"].resolve()),
            "topic_terms_path": str(paths["topic_terms_path"].resolve()),
            "topic_share_by_period_path": str(paths["topic_share_by_period_path"].resolve()),
            "topic_share_by_period_and_keyword_path": str(paths["topic_share_by_period_and_keyword_path"].resolve()),
            "topic_share_by_ip_path": str(paths["topic_share_by_ip_path"].resolve()),
            "topic_share_by_period_and_ip_path": str(paths["topic_share_by_period_and_ip_path"].resolve()),
            "topic_share_by_period_and_ip_and_keyword_path": str(
                paths["topic_share_by_period_and_ip_and_keyword_path"].resolve()
            ),
        }
    )
    save_json(summary_path, summary)
    return summary


def sync_all_analysis_output_metadata(root: Path) -> None:
    topic_model_dir = root / CANONICAL_TOPIC_MODEL_DIR
    if topic_model_dir.is_dir():
        sync_topic_model_output_metadata(topic_model_dir)
    for path in [root / CANONICAL_SEMANTIC_DIR, *iter_snapshot_dirs(root, "semantic")]:
        if path.is_dir():
            sync_semantic_output_metadata(path)
    for path in [root / CANONICAL_DRIFT_DIR, *iter_snapshot_dirs(root, "drift")]:
        if path.is_dir():
            sync_drift_output_metadata(path)
