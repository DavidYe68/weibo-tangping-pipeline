#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lib.broad_analysis_layout import (
    CANONICAL_DRIFT_DIR,
    CANONICAL_SEMANTIC_DIR,
    CANONICAL_TOPIC_MODEL_DIR,
    DRIFT_READOUT_FILES,
    DRIFT_VIZ_FILES,
    READOUTS_DIRNAME,
    SEMANTIC_VIZ_FILES,
    TOPIC_MODEL_READOUT_FILES,
    TOPIC_MODEL_VIZ_FILES,
    VIZ_INPUTS_DIRNAME,
    drift_output_paths,
    ensure_canonical_output_from_latest_snapshot,
    latest_snapshot_dir,
    semantic_output_paths,
    sync_all_analysis_output_metadata,
    sync_topic_model_output_metadata,
    topic_model_output_paths,
)
from lib.broad_analysis_overview import refresh_broad_analysis_overview


CANONICAL_TOP_LEVEL_DIRS = {"broad_analysis", "runs"}
DUAL_LABEL_RUN_MARKERS = {"broad", "strict", "shared", "compare", "inspect"}
SINGLE_LABEL_RUN_MARKERS = {"best_model", "metrics.json"}
LEGACY_BROAD_ANALYSIS_DIRS = (
    "topic_interpretability_BAAI",
)
PREFERRED_TOPIC_MODEL_DIRNAME = CANONICAL_TOPIC_MODEL_DIR
TOPIC_MODEL_BUNDLE_MARKERS = (
    "topic_model_summary.json",
    "topic_info.csv",
    "topic_terms.csv",
    "checkpoints",
)
SNAPSHOT_MOVE_RULES = (
    ("semantic_analysis_", "semantic"),
    ("drift_analysis_", "drift"),
    ("overnight_09_10_", "overnight"),
)
NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9._-]+")
VISUALIZATION_DIR_NAMES = (
    "topic_visualization",
    "topic_visuals",
    "topic_visuals_2",
    "topic_visuals_BAAI",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize bert/artifacts into canonical runs/ and legacy/ buckets."
    )
    parser.add_argument(
        "--artifacts_dir",
        default="bert/artifacts",
        help="Artifact root directory. Defaults to bert/artifacts.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned moves without modifying the filesystem.",
    )
    return parser.parse_args()


def emit(message: str) -> None:
    print(f"[organize-artifacts] {message}", flush=True)


def ensure_dir(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def remove_path(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        emit(f"Would remove {path}")
        return
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()


def move_path(source: Path, target: Path, *, dry_run: bool) -> None:
    if not source.exists():
        return
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {target}")

    emit(f"Move {source} -> {target}")
    if dry_run:
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def next_available_target(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.parent / f"{path.name}_{index}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find available target path for {path}")


def merge_directory_contents(source_dir: Path, target_dir: Path, *, dry_run: bool) -> None:
    ensure_dir(target_dir, dry_run=dry_run)
    for child in sorted(source_dir.iterdir(), key=lambda item: item.name):
        target_child = target_dir / child.name
        if target_child.exists():
            raise FileExistsError(f"Refusing to overwrite existing path: {target_child}")
        move_path(child, target_child, dry_run=dry_run)
    remove_path(source_dir, dry_run=dry_run)


def has_any_marker(path: Path, markers: tuple[str, ...]) -> bool:
    return any((path / marker).exists() for marker in markers)


def is_dual_label_run(path: Path) -> bool:
    if not path.is_dir():
        return False
    child_names = {child.name for child in path.iterdir()}
    return DUAL_LABEL_RUN_MARKERS.issubset(child_names)


def is_single_label_run(path: Path) -> bool:
    if not path.is_dir():
        return False
    child_names = {child.name for child in path.iterdir()}
    return SINGLE_LABEL_RUN_MARKERS.issubset(child_names)


def cleanup_ds_store(artifacts_dir: Path, *, dry_run: bool) -> None:
    for ds_store in sorted(artifacts_dir.rglob(".DS_Store")):
        emit(f"Remove {ds_store}")
        if not dry_run:
            ds_store.unlink()


def relocate_training_runs(artifacts_dir: Path, *, dry_run: bool) -> None:
    dual_root = artifacts_dir / "runs" / "dual_label"
    single_root = artifacts_dir / "runs" / "single_label"

    for child in sorted(artifacts_dir.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in CANONICAL_TOP_LEVEL_DIRS:
            continue

        if is_dual_label_run(child):
            move_path(child, dual_root / child.name, dry_run=dry_run)
            continue

        if is_single_label_run(child):
            move_path(child, single_root / child.name, dry_run=dry_run)


def archive_legacy_broad_analysis_outputs(artifacts_dir: Path, *, dry_run: bool) -> None:
    broad_analysis_dir = artifacts_dir / "broad_analysis"
    if not broad_analysis_dir.is_dir():
        return

    legacy_root = broad_analysis_dir / "legacy"
    for name in LEGACY_BROAD_ANALYSIS_DIRS:
        source = broad_analysis_dir / name
        if source.exists():
            move_path(source, next_available_target(legacy_root / name), dry_run=dry_run)


def relocate_snapshot_outputs(artifacts_dir: Path, *, dry_run: bool) -> None:
    broad_analysis_dir = artifacts_dir / "broad_analysis"
    if not broad_analysis_dir.is_dir():
        return

    snapshot_root = broad_analysis_dir / "snapshots"
    for prefix, group in SNAPSHOT_MOVE_RULES:
        for source in sorted(broad_analysis_dir.glob(f"{prefix}*"), key=lambda item: item.name):
            if not source.is_dir():
                continue
            suffix = source.name.removeprefix(prefix)
            target = next_available_target(snapshot_root / group / suffix)
            if source == target:
                continue
            move_path(source, target, dry_run=dry_run)


def flatten_nested_bundle(bundle_dir: Path, *, markers: tuple[str, ...], dry_run: bool) -> None:
    if not bundle_dir.is_dir():
        return
    if has_any_marker(bundle_dir, markers):
        return

    child_dirs = sorted([child for child in bundle_dir.iterdir() if child.is_dir()], key=lambda item: item.name)
    if len(child_dirs) != 1:
        return

    candidate = child_dirs[0]
    if not has_any_marker(candidate, markers):
        return

    emit(f"Flatten {bundle_dir.name} bundle from {candidate} into {bundle_dir}")
    merge_directory_contents(candidate, bundle_dir, dry_run=dry_run)


def restore_preferred_topic_model(artifacts_dir: Path, *, dry_run: bool) -> None:
    broad_analysis_dir = artifacts_dir / "broad_analysis"
    legacy_root = broad_analysis_dir / "legacy"
    preferred_dir = broad_analysis_dir / PREFERRED_TOPIC_MODEL_DIRNAME
    legacy_preferred_dir = legacy_root / PREFERRED_TOPIC_MODEL_DIRNAME

    if not preferred_dir.exists() and legacy_preferred_dir.exists():
        move_path(legacy_preferred_dir, preferred_dir, dry_run=dry_run)

    flatten_nested_bundle(preferred_dir, markers=TOPIC_MODEL_BUNDLE_MARKERS, dry_run=dry_run)


def read_topic_model_summary(topic_model_dir: Path) -> dict:
    summary_path = topic_model_dir / "topic_model_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict, *, dry_run: bool) -> None:
    emit(f"Write {path}")
    if dry_run:
        return
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize_slug(value: str) -> str:
    slug = NON_ALNUM_RE.sub("-", value.strip()).strip("-._")
    return slug or "legacy"


def archive_nonpreferred_topic_model(artifacts_dir: Path, *, dry_run: bool) -> None:
    broad_analysis_dir = artifacts_dir / "broad_analysis"
    legacy_root = broad_analysis_dir / "legacy"
    legacy_root.mkdir(parents=True, exist_ok=True) if not dry_run else None

    source_dir = broad_analysis_dir / "topic_model"
    preferred_dir = broad_analysis_dir / PREFERRED_TOPIC_MODEL_DIRNAME
    if not source_dir.is_dir() or source_dir == preferred_dir:
        return

    summary = read_topic_model_summary(source_dir)
    embedding_model = str(summary.get("embedding_model", "")).strip()
    target_name = "topic_model_legacy"
    if embedding_model:
        target_name = f"topic_model_{sanitize_slug(embedding_model)}"
    target_dir = legacy_root / target_name
    if target_dir.exists():
        emit(f"Skip archiving {source_dir}; target already exists: {target_dir}")
        return
    move_path(source_dir, target_dir, dry_run=dry_run)


def relocate_stage_bundle(
    stage_dir: Path,
    *,
    readout_files: tuple[str, ...] = (),
    viz_files: tuple[str, ...] = (),
    directory_moves: dict[str, Path] | None = None,
    dry_run: bool,
) -> None:
    if not stage_dir.is_dir():
        return

    readouts_dir = stage_dir / READOUTS_DIRNAME
    viz_inputs_dir = stage_dir / VIZ_INPUTS_DIRNAME
    if readout_files:
        ensure_dir(readouts_dir, dry_run=dry_run)
    if viz_files or directory_moves:
        ensure_dir(viz_inputs_dir, dry_run=dry_run)

    for name in readout_files:
        source = stage_dir / name
        if source.exists():
            move_path(source, readouts_dir / name, dry_run=dry_run)

    for name in viz_files:
        source = stage_dir / name
        if source.exists():
            move_path(source, viz_inputs_dir / name, dry_run=dry_run)

    for name, target in (directory_moves or {}).items():
        source = stage_dir / name
        if source.exists() and source != target:
            move_path(source, target, dry_run=dry_run)


def organize_canonical_stage_layouts(artifacts_dir: Path, *, dry_run: bool) -> None:
    broad_analysis_dir = artifacts_dir / "broad_analysis"
    if not broad_analysis_dir.is_dir():
        return

    topic_model_dir = broad_analysis_dir / CANONICAL_TOPIC_MODEL_DIR
    if topic_model_dir.is_dir():
        topic_paths = topic_model_output_paths(topic_model_dir)
        relocate_stage_bundle(
            topic_model_dir,
            readout_files=TOPIC_MODEL_READOUT_FILES,
            viz_files=TOPIC_MODEL_VIZ_FILES,
            directory_moves={"model": topic_paths["model_dir"]},
            dry_run=dry_run,
        )

    semantic_dir = broad_analysis_dir / CANONICAL_SEMANTIC_DIR
    if semantic_dir.is_dir():
        semantic_paths = semantic_output_paths(semantic_dir)
        relocate_stage_bundle(
            semantic_dir,
            viz_files=SEMANTIC_VIZ_FILES,
            directory_moves={"midterm_bundle": semantic_paths["midterm_bundle_dir"]},
            dry_run=dry_run,
        )

    drift_dir = broad_analysis_dir / CANONICAL_DRIFT_DIR
    if drift_dir.is_dir():
        relocate_stage_bundle(
            drift_dir,
            readout_files=DRIFT_READOUT_FILES,
            viz_files=DRIFT_VIZ_FILES,
            dry_run=dry_run,
        )


def sync_preferred_topic_model_metadata(artifacts_dir: Path, *, dry_run: bool) -> None:
    topic_model_dir = artifacts_dir / "broad_analysis" / PREFERRED_TOPIC_MODEL_DIRNAME
    if not topic_model_dir.is_dir():
        return

    topic_paths = topic_model_output_paths(topic_model_dir)
    summary_path = topic_paths["summary_path"]
    checkpoint_manifest_path = topic_model_dir / "checkpoints" / "checkpoint_manifest.json"
    output_dir = topic_model_dir.resolve()
    checkpoint_dir = (topic_model_dir / "checkpoints").resolve()

    if summary_path.exists():
        summary = read_topic_model_summary(topic_model_dir)
        if summary:
            summary.update(
                {
                    "output_dir": str(output_dir),
                    "readouts_dir": str(topic_paths["readouts_dir"].resolve()),
                    "viz_inputs_dir": str(topic_paths["viz_inputs_dir"].resolve()),
                    "checkpoint_dir": str(checkpoint_dir),
                    "embeddings_checkpoint_path": str(checkpoint_dir / "document_embeddings.npy"),
                    "reduced_embeddings_checkpoint_path": str(checkpoint_dir / "reduced_embeddings.npy"),
                    "reducer_model_checkpoint_path": str(checkpoint_dir / "dimensionality_reduction_model.pkl"),
                    "filtered_documents_checkpoint_path": str(checkpoint_dir / "filtered_documents.parquet"),
                    "checkpoint_manifest_path": str(checkpoint_dir / "checkpoint_manifest.json"),
                    "documents_path": str(topic_paths["documents_path"].resolve()),
                    "topic_info_path": str(topic_paths["topic_info_path"].resolve()),
                    "topic_overview_path": str(topic_paths["topic_overview_path"].resolve()),
                    "topic_terms_path": str(topic_paths["topic_terms_path"].resolve()),
                    "topic_share_by_period_path": str(topic_paths["topic_share_by_period_path"].resolve()),
                    "topic_share_by_period_and_keyword_path": str(
                        topic_paths["topic_share_by_period_and_keyword_path"].resolve()
                    ),
                    "topic_share_by_ip_path": str(topic_paths["topic_share_by_ip_path"].resolve()),
                    "topic_share_by_period_and_ip_path": str(topic_paths["topic_share_by_period_and_ip_path"].resolve()),
                    "topic_share_by_period_and_ip_and_keyword_path": str(
                        topic_paths["topic_share_by_period_and_ip_and_keyword_path"].resolve()
                    ),
                }
            )
            write_json(summary_path, summary, dry_run=dry_run)

    if checkpoint_manifest_path.exists():
        try:
            checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            checkpoint_manifest = {}
        if checkpoint_manifest:
            checkpoint_manifest.update(
                {
                    "embeddings_checkpoint_path": str(checkpoint_dir / "document_embeddings.npy"),
                    "reduced_embeddings_checkpoint_path": str(checkpoint_dir / "reduced_embeddings.npy"),
                    "reducer_model_checkpoint_path": str(checkpoint_dir / "dimensionality_reduction_model.pkl"),
                }
            )
            write_json(checkpoint_manifest_path, checkpoint_manifest, dry_run=dry_run)
    if not dry_run:
        sync_topic_model_output_metadata(topic_model_dir)


def ensure_canonical_analysis_outputs(artifacts_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        broad_analysis_dir = artifacts_dir / "broad_analysis"
        for snapshot_group, canonical_name in (("semantic", CANONICAL_SEMANTIC_DIR), ("drift", CANONICAL_DRIFT_DIR)):
            canonical_dir = broad_analysis_dir / canonical_name
            if canonical_dir.is_dir():
                continue
            snapshot_dir = latest_snapshot_dir(broad_analysis_dir, snapshot_group)
            if snapshot_dir is not None:
                emit(f"Would copy latest {snapshot_group} snapshot {snapshot_dir} into {canonical_dir}")
        return

    broad_analysis_dir = artifacts_dir / "broad_analysis"
    ensure_canonical_output_from_latest_snapshot(
        broad_analysis_dir,
        snapshot_group="semantic",
        canonical_dir_name=CANONICAL_SEMANTIC_DIR,
    )
    ensure_canonical_output_from_latest_snapshot(
        broad_analysis_dir,
        snapshot_group="drift",
        canonical_dir_name=CANONICAL_DRIFT_DIR,
    )
    sync_all_analysis_output_metadata(broad_analysis_dir)


def remove_visualization_outputs(artifacts_dir: Path, *, dry_run: bool) -> None:
    broad_analysis_dir = artifacts_dir / "broad_analysis"
    if not broad_analysis_dir.is_dir():
        return

    candidates = [broad_analysis_dir / name for name in VISUALIZATION_DIR_NAMES]
    candidates.extend((broad_analysis_dir / "legacy" / name) for name in VISUALIZATION_DIR_NAMES)
    for path in candidates:
        if not path.exists():
            continue
        emit(f"Remove {path}")
        if not dry_run:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def write_layout_summary(artifacts_dir: Path, *, dry_run: bool) -> None:
    summary_path = artifacts_dir / "ARTIFACT_LAYOUT.md"
    content = "\n".join(
        [
            "# Artifact Layout",
            "",
            "- `broad_analysis/README.md`: start-here guide for broad-analysis outputs.",
            "- `broad_analysis/overview/`: condensed tables and the generated manifest.",
            "- `broad_analysis/topic_model_BAAI/`: canonical BERTopic outputs (`readouts/` for direct reading, `viz_inputs/` for programmatic inputs, `checkpoints/` kept in place).",
            "- `broad_analysis/semantic_analysis/`: canonical 09 semantic-analysis outputs (`readouts/` + `viz_inputs/`).",
            "- `broad_analysis/drift_analysis/`: canonical 10 drift-analysis outputs (`readouts/` + `viz_inputs/`).",
            "- `broad_analysis/snapshots/`: dated or one-off analysis snapshots.",
            "- `broad_analysis/legacy/`: older or alternate topic-analysis outputs kept for reference.",
            "- `runs/dual_label/`: archived dual-label training runs.",
            "- `runs/single_label/`: archived single-label training runs.",
            "",
            "This file is generated by `bert/scripts/organize_artifacts.py`.",
        ]
    )
    emit(f"Write {summary_path}")
    if dry_run:
        return
    summary_path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
    if not artifacts_dir.is_dir():
        raise NotADirectoryError(f"Artifacts path is not a directory: {artifacts_dir}")

    cleanup_ds_store(artifacts_dir, dry_run=args.dry_run)
    relocate_training_runs(artifacts_dir, dry_run=args.dry_run)
    archive_legacy_broad_analysis_outputs(artifacts_dir, dry_run=args.dry_run)
    relocate_snapshot_outputs(artifacts_dir, dry_run=args.dry_run)
    restore_preferred_topic_model(artifacts_dir, dry_run=args.dry_run)
    archive_nonpreferred_topic_model(artifacts_dir, dry_run=args.dry_run)
    sync_preferred_topic_model_metadata(artifacts_dir, dry_run=args.dry_run)
    ensure_canonical_analysis_outputs(artifacts_dir, dry_run=args.dry_run)
    organize_canonical_stage_layouts(artifacts_dir, dry_run=args.dry_run)
    sync_preferred_topic_model_metadata(artifacts_dir, dry_run=args.dry_run)
    if not args.dry_run:
        sync_all_analysis_output_metadata(artifacts_dir / "broad_analysis")
    remove_visualization_outputs(artifacts_dir, dry_run=args.dry_run)
    write_layout_summary(artifacts_dir, dry_run=args.dry_run)
    if not args.dry_run:
        refresh_broad_analysis_overview(artifacts_dir / "broad_analysis")
    emit("Done")


if __name__ == "__main__":
    main()
