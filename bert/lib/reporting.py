from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").fillna("false").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def _text_preview(series: pd.Series, limit: int = 120) -> pd.Series:
    text = series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return np.where(text.str.len() > limit, text.str.slice(0, limit) + "...", text)


def build_metric_rows(
    label_name: str,
    metrics_payload: Dict[str, Any],
    *,
    experiment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split_name, metric_key in (("validation", "validation_metrics"), ("test", "test_metrics")):
        metrics = metrics_payload[metric_key]
        split_sizes = metrics_payload.get("split_sizes", {})
        label_distribution = metrics_payload.get("label_distribution", {})
        distribution = label_distribution.get("val" if split_name == "validation" else "test", {})
        confusion = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        tn, fp = confusion[0]
        fn, tp = confusion[1]
        row = {
            "experiment_name": experiment_name or "",
            "label_standard": label_name,
            "split": split_name,
            "accuracy": metrics.get("accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0),
            "loss": metrics.get("loss", 0.0),
            "support": metrics.get("support", 0),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "train_rows": split_sizes.get("train", 0),
            "val_rows": split_sizes.get("val", 0),
            "test_rows": split_sizes.get("test", 0),
            "negative_rows": distribution.get("0", 0),
            "positive_rows": distribution.get("1", 0),
        }
        rows.append(row)
    return rows


def build_metrics_snapshot(metrics_payload: Dict[str, Any]) -> Dict[str, Any]:
    validation = metrics_payload.get("validation_metrics", {})
    test = metrics_payload.get("test_metrics", {})
    split_sizes = metrics_payload.get("split_sizes", {})
    return {
        "split_sizes": split_sizes,
        "validation": {
            "accuracy": validation.get("accuracy", 0.0),
            "precision": validation.get("precision", 0.0),
            "recall": validation.get("recall", 0.0),
            "f1": validation.get("f1", 0.0),
            "loss": validation.get("loss", 0.0),
        },
        "test": {
            "accuracy": test.get("accuracy", 0.0),
            "precision": test.get("precision", 0.0),
            "recall": test.get("recall", 0.0),
            "f1": test.get("f1", 0.0),
            "loss": test.get("loss", 0.0),
        },
    }


def _build_label_error_summary(df: pd.DataFrame, label_name: str, source_col: Optional[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    working = df.copy()
    working["is_error_bool"] = _bool_series(working["is_error"])

    error_only = working[working["is_error_bool"]].copy()
    for error_type, frame in error_only.groupby("error_type", dropna=False):
        rows.append(
            {
                "label_standard": label_name,
                "source_name": "",
                "error_type": str(error_type or ""),
                "count": int(len(frame)),
                "mean_confidence": float(frame["pred_confidence"].mean()) if "pred_confidence" in frame.columns else 0.0,
            }
        )

    if source_col and source_col in working.columns:
        grouped = (
            error_only.groupby([source_col, "error_type"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values([source_col, "error_type"])
        )
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "label_standard": label_name,
                    "source_name": str(row[source_col]),
                    "error_type": str(row["error_type"] or ""),
                    "count": int(row["count"]),
                    "mean_confidence": None,
                }
            )
    return rows


def _write_ranked_errors(
    errors: pd.DataFrame,
    output_path: Path,
    *,
    text_col: str,
    source_col: Optional[str],
) -> str:
    if errors.empty:
        pd.DataFrame(columns=["error_type", "pred_confidence", "text_preview"]).to_csv(
            output_path, index=False, encoding="utf-8-sig"
        )
        return str(output_path.resolve())

    ranked = errors.copy()
    ranked["text_preview"] = _text_preview(ranked[text_col])
    if "pred_confidence" in ranked.columns:
        ranked = ranked.sort_values(["pred_confidence"], ascending=[False]).reset_index(drop=True)

    ordered_columns = [
        "__eval_row_id",
        "__dual_row_id",
        "__source_name" if source_col else None,
        "id",
        text_col,
        "text_preview",
        "gold_label_text",
        "pred_label_text",
        "pred_confidence",
        "pred_prob_1",
        "pred_prob_0",
        "error_type",
    ]
    existing_columns = [column for column in ordered_columns if column and column in ranked.columns]
    extra_columns = [column for column in ranked.columns if column not in existing_columns and column != "is_error_bool"]
    ranked[existing_columns + extra_columns].head(200).to_csv(output_path, index=False, encoding="utf-8-sig")
    return str(output_path.resolve())


def _write_error_focus_files(
    df: pd.DataFrame,
    label_name: str,
    inspect_dir: Path,
    *,
    text_col: str,
    source_col: Optional[str],
) -> Dict[str, str]:
    working = df.copy()
    working["is_error_bool"] = _bool_series(working["is_error"])
    errors = working[working["is_error_bool"]].copy()

    return {
        f"top_errors_{label_name}_path": _write_ranked_errors(
            errors,
            inspect_dir / f"top_errors_{label_name}.csv",
            text_col=text_col,
            source_col=source_col,
        ),
        f"top_fp_{label_name}_path": _write_ranked_errors(
            errors[errors["error_type"] == "FP"].copy(),
            inspect_dir / f"top_fp_{label_name}.csv",
            text_col=text_col,
            source_col=source_col,
        ),
        f"top_fn_{label_name}_path": _write_ranked_errors(
            errors[errors["error_type"] == "FN"].copy(),
            inspect_dir / f"top_fn_{label_name}.csv",
            text_col=text_col,
            source_col=source_col,
        ),
    }


def _focus_from_metrics(precision: float, recall: float, f1: float) -> str:
    if f1 >= 0.85:
        return "overall_ok"
    if recall + 0.08 < precision:
        return "recall_low"
    if precision + 0.08 < recall:
        return "precision_low"
    return "mixed_errors"


def _focus_note(focus: str) -> str:
    mapping = {
        "overall_ok": "整体还可以，优先检查高置信度错例和边界样本。",
        "recall_low": "召回偏低，优先看 FN 和漏判样本。",
        "precision_low": "精度偏低，优先看 FP 和误报样本。",
        "mixed_errors": "精度和召回都需要看，先看 side-by-side 和高置信度错例。",
    }
    return mapping.get(focus, "优先看高置信度错例。")


def _build_label_diagnosis(
    metrics_overview: pd.DataFrame,
    error_summary: pd.DataFrame,
    *,
    experiment_name: Optional[str] = None,
) -> pd.DataFrame:
    if metrics_overview.empty:
        return pd.DataFrame(
            columns=[
                "experiment_name",
                "label_standard",
                "test_f1",
                "test_precision",
                "test_recall",
                "val_f1",
                "dominant_error_type",
                "dominant_error_count",
                "focus",
                "suggested_review_file",
                "note",
            ]
        )

    test_rows = metrics_overview[metrics_overview["split"] == "test"].copy()
    val_rows = (
        metrics_overview[metrics_overview["split"] == "validation"][["label_standard", "f1"]]
        .rename(columns={"f1": "val_f1"})
        .reset_index(drop=True)
    )
    merged = test_rows.merge(val_rows, on="label_standard", how="left")

    overall_errors = error_summary[error_summary["source_name"].fillna("") == ""].copy() if not error_summary.empty else error_summary
    diagnosis_rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        label_name = str(row["label_standard"])
        label_errors = overall_errors[overall_errors["label_standard"] == label_name].copy() if not overall_errors.empty else pd.DataFrame()
        dominant_error_type = ""
        dominant_error_count = 0
        if not label_errors.empty:
            label_errors = label_errors.sort_values(["count", "error_type"], ascending=[False, True]).reset_index(drop=True)
            dominant_error_type = str(label_errors.iloc[0]["error_type"])
            dominant_error_count = int(label_errors.iloc[0]["count"])

        focus = _focus_from_metrics(float(row["precision"]), float(row["recall"]), float(row["f1"]))
        suggested_review_file = f"top_errors_{label_name}.csv"
        if focus == "recall_low":
            suggested_review_file = f"top_fn_{label_name}.csv"
        elif focus == "precision_low":
            suggested_review_file = f"top_fp_{label_name}.csv"

        diagnosis_rows.append(
            {
                "experiment_name": experiment_name or "",
                "label_standard": label_name,
                "test_f1": float(row["f1"]),
                "test_precision": float(row["precision"]),
                "test_recall": float(row["recall"]),
                "val_f1": float(row.get("val_f1", 0.0) or 0.0),
                "dominant_error_type": dominant_error_type,
                "dominant_error_count": dominant_error_count,
                "focus": focus,
                "suggested_review_file": suggested_review_file,
                "note": _focus_note(focus),
            }
        )

    return pd.DataFrame(diagnosis_rows).sort_values(["label_standard"]).reset_index(drop=True)


def _write_side_by_side_diagnostics(
    side_by_side_path: Path,
    inspect_dir: Path,
    *,
    text_col: str,
    source_col: Optional[str],
) -> Dict[str, str]:
    side_by_side = pd.read_csv(side_by_side_path)
    broad_error = _bool_series(side_by_side["broad_is_error"])
    strict_error = _bool_series(side_by_side["strict_is_error"])

    category = np.select(
        [
            broad_error & strict_error,
            broad_error & ~strict_error,
            ~broad_error & strict_error,
        ],
        [
            "both_error",
            "broad_only_error",
            "strict_only_error",
        ],
        default="both_correct",
    )
    side_by_side["error_bucket"] = category
    summary = (
        side_by_side["error_bucket"]
        .value_counts()
        .rename_axis("error_bucket")
        .reset_index(name="count")
        .sort_values("error_bucket")
    )
    summary_path = inspect_dir / "side_by_side_error_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    error_rows = side_by_side[side_by_side["error_bucket"] != "both_correct"].copy()
    if not error_rows.empty:
        error_rows["text_preview"] = _text_preview(error_rows[text_col])
        error_rows["max_confidence"] = error_rows[["broad_pred_confidence", "strict_pred_confidence"]].max(axis=1)
        error_rows = error_rows.sort_values(["error_bucket", "max_confidence"], ascending=[True, False]).reset_index(drop=True)
    review_columns = [
        "__eval_row_id" if "__eval_row_id" in error_rows.columns else None,
        "__dual_row_id" if "__dual_row_id" in error_rows.columns else None,
        "__source_name" if source_col and "__source_name" in error_rows.columns else None,
        "__eval_split" if "__eval_split" in error_rows.columns else None,
        "__dual_split" if "__dual_split" in error_rows.columns else None,
        "id" if "id" in error_rows.columns else None,
        text_col if text_col in error_rows.columns else None,
        "text_preview" if "text_preview" in error_rows.columns else None,
        "error_bucket" if "error_bucket" in error_rows.columns else None,
        "broad_gold_label_text",
        "broad_pred_label_text",
        "broad_pred_confidence",
        "broad_error_type",
        "strict_gold_label_text",
        "strict_pred_label_text",
        "strict_pred_confidence",
        "strict_error_type",
    ]
    existing_review_columns = [column for column in review_columns if column]
    review_path = inspect_dir / "side_by_side_errors_review.csv"
    error_rows[existing_review_columns].head(300).to_csv(review_path, index=False, encoding="utf-8-sig")

    return {
        "side_by_side_error_summary_path": str(summary_path.resolve()),
        "side_by_side_errors_review_path": str(review_path.resolve()),
    }


def write_dual_run_inspect_artifacts(
    base_output_dir: Path,
    *,
    experiment_name: Optional[str] = None,
    text_col: str,
    source_col: Optional[str] = None,
) -> Dict[str, str]:
    inspect_dir = base_output_dir / "inspect"
    inspect_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    output_paths: Dict[str, str] = {}

    for label_name in ("broad", "strict"):
        metrics_payload = _load_json(base_output_dir / label_name / "metrics.json")
        metrics_rows.extend(build_metric_rows(label_name, metrics_payload, experiment_name=experiment_name))

        predictions = pd.read_csv(base_output_dir / label_name / "test_predictions.csv")
        error_rows.extend(_build_label_error_summary(predictions, label_name, source_col))
        output_paths.update(
            _write_error_focus_files(
                predictions,
                label_name,
                inspect_dir,
                text_col=text_col,
                source_col=source_col,
            )
        )

    metrics_overview = pd.DataFrame(metrics_rows).sort_values(["label_standard", "split"]).reset_index(drop=True)
    metrics_overview_path = inspect_dir / "metrics_overview.csv"
    metrics_overview.to_csv(metrics_overview_path, index=False, encoding="utf-8-sig")
    output_paths["metrics_overview_path"] = str(metrics_overview_path.resolve())

    error_summary = pd.DataFrame(error_rows)
    if not error_summary.empty:
        error_summary = error_summary.sort_values(["label_standard", "source_name", "error_type"]).reset_index(drop=True)
    error_summary_path = inspect_dir / "error_summary.csv"
    error_summary.to_csv(error_summary_path, index=False, encoding="utf-8-sig")
    output_paths["error_summary_path"] = str(error_summary_path.resolve())

    label_diagnosis = _build_label_diagnosis(metrics_overview, error_summary, experiment_name=experiment_name)
    label_diagnosis_path = inspect_dir / "label_diagnosis.csv"
    label_diagnosis.to_csv(label_diagnosis_path, index=False, encoding="utf-8-sig")
    output_paths["label_diagnosis_path"] = str(label_diagnosis_path.resolve())

    side_by_side_paths = _write_side_by_side_diagnostics(
        base_output_dir / "test_predictions_side_by_side.csv",
        inspect_dir,
        text_col=text_col,
        source_col=source_col,
    )
    output_paths.update(side_by_side_paths)

    metrics_text = metrics_overview.to_string(index=False) if not metrics_overview.empty else "_No metrics rows found._"
    diagnosis_text = label_diagnosis.to_string(index=False) if not label_diagnosis.empty else "_No diagnosis rows found._"
    summary_lines = [
        f"# Inspect Summary: {experiment_name or base_output_dir.name}",
        "",
        "## First Look",
        "1. 先看 label_diagnosis.csv，判断先查 FP 还是 FN。",
        "2. 再看 side_by_side_error_summary.csv，判断是 broad/strict 哪边更不稳。",
        "3. 最后按 diagnosis 建议打开 top_fp_*.csv 或 top_fn_*.csv。",
        "",
        "## Label Diagnosis",
        diagnosis_text,
        "",
        "## Metrics Overview",
        metrics_text,
        "",
        "## Quick Files",
        f"- label_diagnosis.csv: {label_diagnosis_path.name}",
        f"- metrics_overview.csv: {metrics_overview_path.name}",
        f"- error_summary.csv: {error_summary_path.name}",
        f"- side_by_side_error_summary.csv: {Path(side_by_side_paths['side_by_side_error_summary_path']).name}",
        f"- side_by_side_errors_review.csv: {Path(side_by_side_paths['side_by_side_errors_review_path']).name}",
        f"- top_errors_broad.csv: {Path(output_paths['top_errors_broad_path']).name}",
        f"- top_fp_broad.csv: {Path(output_paths['top_fp_broad_path']).name}",
        f"- top_fn_broad.csv: {Path(output_paths['top_fn_broad_path']).name}",
        f"- top_errors_strict.csv: {Path(output_paths['top_errors_strict_path']).name}",
        f"- top_fp_strict.csv: {Path(output_paths['top_fp_strict_path']).name}",
        f"- top_fn_strict.csv: {Path(output_paths['top_fn_strict_path']).name}",
    ]
    summary_md_path = inspect_dir / "summary.md"
    summary_md_path.write_text("\n".join(summary_lines), encoding="utf-8")
    output_paths["summary_md_path"] = str(summary_md_path.resolve())
    return output_paths


def write_eval_collection_inspect_artifacts(base_output_dir: Path, overall_summary: Dict[str, Any]) -> Dict[str, str]:
    inspect_dir = base_output_dir / "inspect"
    inspect_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    triage_rows: List[Dict[str, Any]] = []
    for experiment_name, experiment_summary in overall_summary.get("experiments", {}).items():
        for label_name, run in experiment_summary.get("runs", {}).items():
            metrics_payload = run.get("metrics")
            if not metrics_payload:
                metrics_path = run.get("metrics_path")
                metrics_payload = _load_json(Path(metrics_path)) if metrics_path else {}
            rows.extend(build_metric_rows(label_name, metrics_payload, experiment_name=experiment_name))
            snapshot = build_metrics_snapshot(metrics_payload)
            test_snapshot = snapshot.get("test", {})
            focus = _focus_from_metrics(
                float(test_snapshot.get("precision", 0.0)),
                float(test_snapshot.get("recall", 0.0)),
                float(test_snapshot.get("f1", 0.0)),
            )
            suggested_file = f"{experiment_name}/inspect/top_errors_{label_name}.csv"
            if focus == "recall_low":
                suggested_file = f"{experiment_name}/inspect/top_fn_{label_name}.csv"
            elif focus == "precision_low":
                suggested_file = f"{experiment_name}/inspect/top_fp_{label_name}.csv"
            triage_rows.append(
                {
                    "experiment_name": experiment_name,
                    "label_standard": label_name,
                    "test_f1": float(test_snapshot.get("f1", 0.0)),
                    "test_precision": float(test_snapshot.get("precision", 0.0)),
                    "test_recall": float(test_snapshot.get("recall", 0.0)),
                    "focus": focus,
                    "suggested_review_file": suggested_file,
                    "note": _focus_note(focus),
                }
            )

    scorecard = pd.DataFrame(rows)
    if not scorecard.empty:
        scorecard = scorecard.sort_values(["experiment_name", "label_standard", "split"]).reset_index(drop=True)
    scorecard_path = inspect_dir / "experiment_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False, encoding="utf-8-sig")

    triage = pd.DataFrame(triage_rows)
    if not triage.empty:
        triage = triage.sort_values(["experiment_name", "label_standard"]).reset_index(drop=True)
    triage_path = inspect_dir / "experiment_triage.csv"
    triage.to_csv(triage_path, index=False, encoding="utf-8-sig")

    test_only = scorecard[scorecard["split"] == "test"].copy() if not scorecard.empty else scorecard
    test_text = test_only.to_string(index=False) if not test_only.empty else "_No test rows found._"
    triage_text = triage.to_string(index=False) if not triage.empty else "_No triage rows found._"
    summary_lines = [
        f"# Eval By Source Scorecard: {base_output_dir.name}",
        "",
        "## First Look",
        "1. 先看 experiment_triage.csv，决定每个实验先查 FP 还是 FN。",
        "2. 再进对应实验目录看 inspect/summary.md 和 top_fp/top_fn 文件。",
        "",
        "## Experiment Triage",
        triage_text,
        "",
        "## Test Split",
        test_text,
    ]
    scorecard_md_path = inspect_dir / "experiment_scorecard.md"
    scorecard_md_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "experiment_scorecard_path": str(scorecard_path.resolve()),
        "experiment_scorecard_md_path": str(scorecard_md_path.resolve()),
        "experiment_triage_path": str(triage_path.resolve()),
    }
