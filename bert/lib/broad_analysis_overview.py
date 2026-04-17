from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from lib.broad_analysis_layout import (
    CANONICAL_DRIFT_DIR,
    CANONICAL_SEMANTIC_DIR,
    CANONICAL_TOPIC_MODEL_DIR,
    resolve_drift_artifact,
    resolve_semantic_artifact,
    resolve_topic_model_artifact,
)
from lib.io_utils import ensure_parent, save_json


SNAPSHOT_GROUPS = (
    ("semantic_analysis_", "semantic"),
    ("drift_analysis_", "drift"),
    ("overnight_09_10_", "overnight"),
)


def find_broad_analysis_root(path: str | Path) -> Path | None:
    resolved = Path(path).resolve()
    candidates = [resolved]
    candidates.extend(resolved.parents)
    for candidate in candidates:
        if candidate.name == "broad_analysis":
            return candidate
    return None


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _collect_snapshot_dirs(root: Path) -> list[dict[str, str]]:
    snapshot_entries: list[dict[str, str]] = []
    snapshot_root = root / "snapshots"
    if snapshot_root.is_dir():
        for group_dir in sorted([path for path in snapshot_root.iterdir() if path.is_dir()], key=lambda item: item.name):
            for child in sorted([path for path in group_dir.iterdir() if path.is_dir()], key=lambda item: item.name, reverse=True):
                snapshot_entries.append(
                    {
                        "group": group_dir.name,
                        "name": child.name,
                        "path": _relative_to_root(child, root),
                    }
                )

    for prefix, group in SNAPSHOT_GROUPS:
        for child in sorted(root.glob(f"{prefix}*"), key=lambda item: item.name, reverse=True):
            if not child.is_dir():
                continue
            snapshot_entries.append(
                {
                    "group": group,
                    "name": child.name.removeprefix(prefix),
                    "path": _relative_to_root(child, root),
                }
            )
    return snapshot_entries


def _resolve_output_dir(root: Path, canonical_name: str, snapshot_group: str) -> tuple[Path | None, str]:
    canonical_dir = root / canonical_name
    if canonical_dir.is_dir():
        return canonical_dir, canonical_name

    snapshot_root = root / "snapshots" / snapshot_group
    if snapshot_root.is_dir():
        candidates = sorted([path for path in snapshot_root.iterdir() if path.is_dir()], key=lambda item: item.name)
        if candidates:
            return candidates[-1], f"snapshots/{snapshot_group}/{candidates[-1].name}"

    prefix = next((value for value, group in SNAPSHOT_GROUPS if group == snapshot_group), None)
    if prefix is not None:
        legacy_candidates = sorted([path for path in root.glob(f"{prefix}*") if path.is_dir()], key=lambda item: item.name)
        if legacy_candidates:
            return legacy_candidates[-1], legacy_candidates[-1].name

    return None, ""


def _build_topic_headlines(topic_model_dir: Path, output_path: Path) -> dict[str, Any]:
    topic_info = _safe_read_csv(resolve_topic_model_artifact(topic_model_dir, "topic_info.csv"))
    topic_terms = _safe_read_csv(resolve_topic_model_artifact(topic_model_dir, "topic_terms.csv"))
    share_by_period = _safe_read_csv(resolve_topic_model_artifact(topic_model_dir, "topic_share_by_period.csv"))
    share_by_period_keyword = _safe_read_csv(
        resolve_topic_model_artifact(topic_model_dir, "topic_share_by_period_and_keyword.csv")
    )

    if topic_info.empty:
        return {}

    info = topic_info.copy()
    info["topic_id"] = pd.to_numeric(info["Topic"], errors="coerce")
    info["doc_count"] = pd.to_numeric(info["Count"], errors="coerce").fillna(0).astype(int)
    info = info[info["topic_id"].ge(0)].copy()
    if info.empty:
        return {}

    total_docs = int(info["doc_count"].sum())
    info["overall_doc_share"] = info["doc_count"] / total_docs if total_docs > 0 else 0.0
    info["topic_label"] = (
        info.get("topic_label_zh", pd.Series(index=info.index))
        .fillna(info.get("Name", pd.Series(index=info.index)))
        .astype("string")
        .fillna("")
    )

    if not topic_terms.empty:
        terms = topic_terms.copy()
        terms["topic_id"] = pd.to_numeric(terms["topic_id"], errors="coerce")
        terms["term_rank"] = pd.to_numeric(terms["term_rank"], errors="coerce")
        terms = terms.sort_values(["topic_id", "term_rank"], ascending=[True, True])
        top_terms = (
            terms[terms["term_rank"].le(5)]
            .groupby("topic_id", dropna=False)["term"]
            .apply(lambda values: " / ".join(values.astype(str).tolist()))
            .to_dict()
        )
        info["top_terms"] = info["topic_id"].map(top_terms).fillna("")
    else:
        info["top_terms"] = ""

    if not share_by_period.empty:
        period_df = share_by_period.copy()
        period_df["topic_id"] = pd.to_numeric(period_df["topic_id"], errors="coerce")
        period_df["doc_share"] = pd.to_numeric(period_df["doc_share"], errors="coerce").fillna(0.0)
        period_df["doc_count"] = pd.to_numeric(period_df["doc_count"], errors="coerce").fillna(0).astype(int)
        period_df = period_df.sort_values(["topic_id", "doc_share", "doc_count", "period_label"], ascending=[True, False, False, True])
        top_period = period_df.drop_duplicates(["topic_id"], keep="first").set_index("topic_id")
        info["top_period"] = info["topic_id"].map(top_period["period_label"]).fillna("")
        info["top_period_doc_share"] = info["topic_id"].map(top_period["doc_share"]).fillna(0.0)
    else:
        info["top_period"] = ""
        info["top_period_doc_share"] = 0.0

    keyword_column = "keyword_normalized" if "keyword_normalized" in share_by_period_keyword.columns else "keyword"
    if not share_by_period_keyword.empty and keyword_column in share_by_period_keyword.columns:
        keyword_df = share_by_period_keyword.copy()
        keyword_df["topic_id"] = pd.to_numeric(keyword_df["topic_id"], errors="coerce")
        keyword_df["doc_share"] = pd.to_numeric(keyword_df["doc_share"], errors="coerce").fillna(0.0)
        keyword_df["doc_count"] = pd.to_numeric(keyword_df["doc_count"], errors="coerce").fillna(0).astype(int)
        keyword_df = keyword_df.sort_values(
            ["topic_id", "doc_share", "doc_count", keyword_column],
            ascending=[True, False, False, True],
        )
        top_keyword = keyword_df.drop_duplicates(["topic_id"], keep="first").set_index("topic_id")
        info["top_keyword"] = info["topic_id"].map(top_keyword[keyword_column]).fillna("")
        info["top_keyword_doc_share"] = info["topic_id"].map(top_keyword["doc_share"]).fillna(0.0)
    else:
        info["top_keyword"] = ""
        info["top_keyword_doc_share"] = 0.0

    output = info[
        [
            "topic_id",
            "topic_label",
            "doc_count",
            "overall_doc_share",
            "top_terms",
            "top_period",
            "top_period_doc_share",
            "top_keyword",
            "top_keyword_doc_share",
        ]
    ].sort_values(["doc_count", "topic_id"], ascending=[False, True]).head(20)
    ensure_parent(output_path)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "path": str(output_path.resolve()),
        "row_count": int(len(output)),
    }


def _build_semantic_headlines(semantic_dir: Path, output_path: Path) -> dict[str, Any]:
    cooccurrence = _safe_read_csv(resolve_semantic_artifact(semantic_dir, "keyword_cooccurrence.csv"))
    if cooccurrence.empty:
        return {}

    working = cooccurrence.copy()
    if "period_label" in working.columns and (working["period_label"].astype(str) == "ALL").any():
        working = working[working["period_label"].astype(str) == "ALL"].copy()

    working["term_rank"] = pd.to_numeric(working["term_rank"], errors="coerce").fillna(999999)
    working["term_doc_freq"] = pd.to_numeric(working["term_doc_freq"], errors="coerce").fillna(0).astype(int)
    working["term_doc_rate"] = pd.to_numeric(working["term_doc_rate"], errors="coerce").fillna(0.0)
    working["pmi"] = pd.to_numeric(working["pmi"], errors="coerce").fillna(0.0)
    working["lift"] = pd.to_numeric(working["lift"], errors="coerce").fillna(0.0)
    working = working.sort_values(["keyword", "term_rank", "pmi", "term_doc_freq"], ascending=[True, True, False, False])
    working = working.groupby("keyword", dropna=False).head(12).reset_index(drop=True)
    output = working[["keyword", "term_rank", "term", "term_doc_freq", "term_doc_rate", "pmi", "lift"]]

    ensure_parent(output_path)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "path": str(output_path.resolve()),
        "row_count": int(len(output)),
    }


def _max_drift_rows(
    df: pd.DataFrame,
    *,
    group_col: str,
    score_col: str,
    drift_type: str,
) -> pd.DataFrame:
    if df.empty or group_col not in df.columns or score_col not in df.columns:
        return pd.DataFrame()

    working = df.copy()
    working[group_col] = working[group_col].astype("string").fillna("ALL")
    working[score_col] = pd.to_numeric(working[score_col], errors="coerce").fillna(0.0)
    working = working.sort_values([group_col, score_col], ascending=[True, False])
    working = working.groupby(group_col, dropna=False).head(1).copy()
    working = working.rename(columns={group_col: "focus_area"})
    working["drift_type"] = drift_type
    working["score"] = working[score_col]
    return working


def _build_drift_watchlist(drift_dir: Path, output_path: Path) -> dict[str, Any]:
    collocation = _safe_read_csv(resolve_drift_artifact(drift_dir, "keyword_collocation_drift.csv"))
    neighbor = _safe_read_csv(resolve_drift_artifact(drift_dir, "keyword_neighbor_drift.csv"))
    topic_keyword = _safe_read_csv(resolve_drift_artifact(drift_dir, "topic_drift_by_keyword.csv"))
    topic_overall = _safe_read_csv(resolve_drift_artifact(drift_dir, "topic_drift_overall.csv"))

    tables = [
        _max_drift_rows(
            collocation,
            group_col="keyword",
            score_col="js_divergence",
            drift_type="collocation_js_divergence",
        ),
        _max_drift_rows(
            neighbor,
            group_col="keyword",
            score_col="js_divergence",
            drift_type="semantic_neighbor_js_divergence",
        ),
        _max_drift_rows(
            topic_keyword,
            group_col=("keyword_normalized" if "keyword_normalized" in topic_keyword.columns else "keyword"),
            score_col="topic_js_divergence",
            drift_type="topic_share_js_divergence",
        ),
        _max_drift_rows(
            topic_overall.assign(scope="ALL") if not topic_overall.empty else topic_overall,
            group_col="scope",
            score_col="topic_js_divergence",
            drift_type="overall_topic_js_divergence",
        ),
    ]
    output = pd.concat([table for table in tables if not table.empty], ignore_index=True) if any(
        not table.empty for table in tables
    ) else pd.DataFrame()
    if output.empty:
        return {}

    keep_columns = [
        "focus_area",
        "drift_type",
        "previous_period",
        "current_period",
        "score",
        "overlap_count",
        "jaccard_top_terms",
        "topic_count",
        "overlap_terms",
        "added_terms",
        "removed_terms",
    ]
    available_columns = [column for column in keep_columns if column in output.columns]
    output = output[available_columns].sort_values(["drift_type", "score", "focus_area"], ascending=[True, False, True])
    ensure_parent(output_path)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "path": str(output_path.resolve()),
        "row_count": int(len(output)),
    }


def _write_readme(root: Path, manifest: dict[str, Any]) -> Path:
    readme_path = root / "README.md"
    overview = manifest["overview"]
    lines = [
        "# Broad Analysis Overview",
        "",
        "## First Look",
    ]

    first_look = []
    if overview.get("topic_headlines_path"):
        first_look.append(
            f"1. 先看 `overview/{Path(overview['topic_headlines_path']).name}`：主题主表，只保留最值得先看的头部主题。"
        )
    if overview.get("semantic_headlines_path"):
        first_look.append(
            f"{len(first_look) + 1}. 再看 `overview/{Path(overview['semantic_headlines_path']).name}`：每个关键词的 ALL 时段高价值搭配词。"
        )
    if overview.get("drift_watchlist_path"):
        first_look.append(
            f"{len(first_look) + 1}. 最后看 `overview/{Path(overview['drift_watchlist_path']).name}`：只保留最明显的漂移告警。"
        )

    if first_look:
        lines.extend(first_look)
    else:
        lines.append("1. 当前还没有可汇总的标准输出。先跑 07-10 或整理现有产物。")

    lines.extend(
        [
            "",
            "## Canonical Outputs",
            f"- `analysis_base.parquet` / `analysis_base_report.json`: 分析底表和样本规模说明。",
            f"- `{manifest['topic_model'].get('path_label', CANONICAL_TOPIC_MODEL_DIR)}`: BERTopic 主结果。",
            f"- `{manifest['semantic'].get('path_label', CANONICAL_SEMANTIC_DIR)}`: 语义搭配分析主结果。",
            f"- `{manifest['drift'].get('path_label', CANONICAL_DRIFT_DIR)}`: 概念漂移主结果。",
        ]
    )

    snapshots = manifest.get("snapshots", [])
    lines.extend(["", "## Snapshots"])
    if snapshots:
        for snapshot in snapshots[:12]:
            lines.append(f"- `{snapshot['path']}`")
        if len(snapshots) > 12:
            lines.append(f"- 其余 {len(snapshots) - 12} 个快照已省略，详见 `overview/manifest.json`。")
    else:
        lines.append("- 当前没有额外快照目录；如果需要旧快照，请看 `../_unused/`。")

    lines.extend(
        [
            "",
            "## Notes",
            "- `overview/` 只放入口和浓缩表，不替代原始明细。",
            "- `topic_model_BAAI/`、`semantic_analysis/`、`drift_analysis/` 内部按 `readouts/` 和 `viz_inputs/` 分层：前者给人直接看，后者给程序或可视化调用。",
            "- 当前主目录只保留正在使用的分析结果；旧版结果和训练产物统一放在 `../_unused/`。",
        ]
    )
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def refresh_broad_analysis_overview(root: str | Path) -> dict[str, Any]:
    broad_analysis_root = find_broad_analysis_root(root)
    if broad_analysis_root is None:
        raise ValueError(f"Could not locate broad_analysis root from: {root}")

    overview_dir = broad_analysis_root / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)

    analysis_report = _safe_read_json(broad_analysis_root / "analysis_base_report.json")
    topic_model_dir, topic_model_label = _resolve_output_dir(
        broad_analysis_root,
        CANONICAL_TOPIC_MODEL_DIR,
        "topic_model",
    )
    semantic_dir, semantic_label = _resolve_output_dir(
        broad_analysis_root,
        CANONICAL_SEMANTIC_DIR,
        "semantic",
    )
    drift_dir, drift_label = _resolve_output_dir(
        broad_analysis_root,
        CANONICAL_DRIFT_DIR,
        "drift",
    )
    topic_headlines = _build_topic_headlines(topic_model_dir, overview_dir / "topic_headlines.csv") if topic_model_dir else {}
    semantic_headlines = _build_semantic_headlines(semantic_dir, overview_dir / "semantic_headlines.csv") if semantic_dir else {}
    drift_watchlist = _build_drift_watchlist(drift_dir, overview_dir / "drift_watchlist.csv") if drift_dir else {}

    snapshots = _collect_snapshot_dirs(broad_analysis_root)

    manifest: dict[str, Any] = {
        "root": str(broad_analysis_root.resolve()),
        "analysis_base": {
            "report_path": _relative_to_root(broad_analysis_root / "analysis_base_report.json", broad_analysis_root),
            "dataset_path": _relative_to_root(broad_analysis_root / "analysis_base.parquet", broad_analysis_root),
            "selected_keywords": analysis_report.get("selected_keywords", []),
            "rows_by_keyword": analysis_report.get("rows_by_keyword", {}),
            "rows_by_period_count": len(analysis_report.get("rows_by_period", {})),
        },
        "topic_model": {
            "path_label": topic_model_label,
            "path": _relative_to_root(topic_model_dir, broad_analysis_root) if topic_model_dir else "",
        },
        "semantic": {
            "path_label": semantic_label,
            "path": _relative_to_root(semantic_dir, broad_analysis_root) if semantic_dir else "",
        },
        "drift": {
            "path_label": drift_label,
            "path": _relative_to_root(drift_dir, broad_analysis_root) if drift_dir else "",
        },
        "overview": {
            "topic_headlines_path": topic_headlines.get("path", ""),
            "semantic_headlines_path": semantic_headlines.get("path", ""),
            "drift_watchlist_path": drift_watchlist.get("path", ""),
        },
        "snapshots": snapshots,
    }
    readme_path = _write_readme(broad_analysis_root, manifest)
    manifest["overview"]["readme_path"] = str(readme_path.resolve())
    manifest_path = overview_dir / "manifest.json"
    save_json(manifest_path, manifest)
    manifest["overview"]["manifest_path"] = str(manifest_path.resolve())
    return manifest
