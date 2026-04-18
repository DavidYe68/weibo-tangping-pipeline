#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from lib.analysis_utils import DEFAULT_ANALYSIS_KEYWORDS, normalize_cli_keywords, save_dataframe
from lib.broad_analysis_layout import (
    resolve_semantic_artifact,
    semantic_output_paths,
    semantic_readout_path,
    sync_semantic_output_metadata,
)
from lib.broad_analysis_overview import refresh_broad_analysis_overview
from lib.io_utils import save_json

DEFAULT_SEMANTIC_DIR = "bert/artifacts/broad_analysis/semantic_analysis"
DEFAULT_NOISE_TERMS_PATH = "bert/config/semantic_midterm_noise_terms.txt"
DEFAULT_BUCKET_RULES_PATH = "bert/config/semantic_bucket_rules.json"
DEFAULT_BUCKET_OVERRIDES_PATH = "bert/config/semantic_bucket_overrides.csv"
TOKENIZED_BASE_COLUMNS = (
    "keyword_normalized",
    "period_label",
    "analysis_text",
    "tokens",
    "点赞数",
    "评论数",
    "转发数",
)
PURE_ASCII_TERM_RE = re.compile(r"^[A-Za-z0-9._-]+$")
HAS_ASCII_RE = re.compile(r"[A-Za-z]")
HAS_DIGIT_RE = re.compile(r"\d")
NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
NOISE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("platform_recruitment", re.compile(r"(进团|代招|招代招|回官博|官博|单秒|男店)")),
    ("keyword_self_variant", re.compile(r"(佛系|摆烂|躺平)")),
)

@dataclass(frozen=True)
class MarkerBucketRule:
    bucket: str
    markers: tuple[str, ...]


@dataclass(frozen=True)
class BucketRuleSet:
    theme_rules: tuple[MarkerBucketRule, ...]
    context_rules: tuple[MarkerBucketRule, ...]


@dataclass(frozen=True)
class BucketOverride:
    keyword: str
    period_label: str
    term: str
    context_bucket: str
    theme_bucket: str
    note: str

    def specificity(self) -> tuple[int, int, int]:
        return (
            int(bool(self.keyword)),
            int(bool(self.period_label)),
            int(bool(self.term)),
        )


@dataclass
class ExampleRow:
    score: float
    text: str
    period_label: str
    likes: int
    comments: int
    reposts: int


@dataclass
class MatchStats:
    match_count: int
    distinct_text_count: int
    top_text_share: float


class OperationLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"- {timestamp} {message}"
        print(f"[semantic-midterm] {message}", flush=True)
        self.lines.append(line)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = "# 09 Midterm Operation Log\n\n" + "\n".join(self.lines) + "\n"
        self.path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn 09 semantic-analysis outputs into report-ready midterm tables."
    )
    parser.add_argument(
        "--semantic_dir",
        default=DEFAULT_SEMANTIC_DIR,
        help="Directory containing 09 semantic-analysis outputs.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for the report-ready 09 readouts. Defaults to <semantic_dir>/readouts.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=list(DEFAULT_ANALYSIS_KEYWORDS),
        help="Canonical keywords to keep. Defaults to 躺平 摆烂 佛系.",
    )
    parser.add_argument(
        "--noise_terms_path",
        default=DEFAULT_NOISE_TERMS_PATH,
        help="Optional newline-delimited exact noise terms for report cleaning.",
    )
    parser.add_argument(
        "--bucket_rules_path",
        default=DEFAULT_BUCKET_RULES_PATH,
        help="JSON file defining theme/context bucket names and marker lists.",
    )
    parser.add_argument(
        "--bucket_overrides_path",
        default=DEFAULT_BUCKET_OVERRIDES_PATH,
        help="CSV file for manual bucket overrides. Missing file is ignored.",
    )
    parser.add_argument(
        "--top_n_all",
        type=int,
        default=18,
        help="Top cleaned ALL-period terms to keep per keyword.",
    )
    parser.add_argument(
        "--top_n_period",
        type=int,
        default=6,
        help="Top cleaned terms to keep per keyword and period.",
    )
    parser.add_argument(
        "--min_doc_freq_all",
        type=int,
        default=60,
        help="Minimum document frequency for ALL-period overview rows.",
    )
    parser.add_argument(
        "--min_doc_freq_period",
        type=int,
        default=20,
        help="Minimum document frequency for period-level shortlist rows.",
    )
    parser.add_argument(
        "--example_count",
        type=int,
        default=2,
        help="Number of representative text examples to keep for each shortlisted term.",
    )
    return parser.parse_args()


def load_noise_terms(path: str | None) -> set[str]:
    if not path:
        return set()
    resolved = Path(path)
    if not resolved.exists():
        return set()
    terms: set[str] = set()
    for line in resolved.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        terms.add(value)
    return terms


def _normalize_marker_rules(payload: object, field_name: str) -> tuple[MarkerBucketRule, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{field_name} must be a list of objects.")
    rules: list[MarkerBucketRule] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"{field_name} entries must be objects.")
        bucket = str(item.get("bucket", "")).strip()
        markers_raw = item.get("markers", [])
        if not bucket:
            raise ValueError(f"{field_name} entries require a non-empty 'bucket'.")
        if not isinstance(markers_raw, list):
            raise ValueError(f"{field_name}.{bucket}.markers must be a list.")
        markers = tuple(str(marker).strip() for marker in markers_raw if str(marker).strip())
        rules.append(MarkerBucketRule(bucket=bucket, markers=markers))
    return tuple(rules)


def load_bucket_rules(path: str | None) -> BucketRuleSet:
    if not path:
        raise ValueError("bucket_rules_path is required.")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Bucket rules file not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Bucket rules JSON must be an object.")
    return BucketRuleSet(
        theme_rules=_normalize_marker_rules(payload.get("theme_buckets", []), "theme_buckets"),
        context_rules=_normalize_marker_rules(payload.get("context_buckets", []), "context_buckets"),
    )


def load_bucket_overrides(path: str | None) -> list[BucketOverride]:
    if not path:
        return []
    resolved = Path(path)
    if not resolved.exists():
        return []
    overrides: list[BucketOverride] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            term = str(row.get("term", "")).strip()
            context_bucket = str(row.get("override_context_bucket", "")).strip()
            theme_bucket = str(row.get("override_theme_bucket", "")).strip()
            enabled_text = str(row.get("enabled", "1")).strip().lower()
            enabled = enabled_text not in {"0", "false", "no", "off"}
            if not enabled or (not term and not str(row.get("keyword", "")).strip()):
                continue
            if not context_bucket and not theme_bucket:
                continue
            overrides.append(
                BucketOverride(
                    keyword=str(row.get("keyword", "")).strip(),
                    period_label=str(row.get("period_label", "")).strip(),
                    term=term,
                    context_bucket=context_bucket,
                    theme_bucket=theme_bucket,
                    note=str(row.get("note", "")).strip(),
                )
            )
    overrides.sort(key=lambda item: item.specificity(), reverse=True)
    return overrides


def resolve_tokenized_analysis_base_path(summary: dict, semantic_dir: Path) -> Path:
    candidates: list[Path] = []
    raw_path = summary.get("tokenized_analysis_base_path")
    if raw_path:
        candidate = Path(str(raw_path))
        if not candidate.is_absolute():
            candidate = (semantic_dir / candidate).resolve()
        candidates.append(candidate)
    candidates.append(resolve_semantic_artifact(semantic_dir, "tokenized_analysis_base.parquet").resolve())
    candidates.append((semantic_dir / "tokenized_analysis_base.parquet").resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find tokenized_analysis_base.parquet. Checked: {searched}")


def truncate_text(value: object, limit: int = 140) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def infer_bucket_from_markers(text: str, rules: Iterable[MarkerBucketRule], default_bucket: str = "待人工判断") -> str:
    lowered = str(text or "").lower()
    for rule in rules:
        if any(marker.lower() in lowered for marker in rule.markers):
            return rule.bucket
    return default_bucket


def infer_theme_bucket(term: str, reasons: list[str], bucket_rules: BucketRuleSet) -> str:
    if "keyword_self_variant" in reasons:
        return "关键词自变体"
    if any(reason in {"ascii_or_id_term", "alpha_numeric_noise", "custom_noise_term"} for reason in reasons):
        return "账号/代码噪声"
    return infer_bucket_from_markers(term, bucket_rules.theme_rules)


def classify_term(term: str, keyword: str, exact_noise_terms: set[str], bucket_rules: BucketRuleSet) -> tuple[list[str], str]:
    reasons: list[str] = []
    if keyword in term:
        reasons.append("keyword_self_variant")
    if PURE_ASCII_TERM_RE.fullmatch(term):
        reasons.append("ascii_or_id_term")
    elif HAS_ASCII_RE.search(term) and HAS_DIGIT_RE.search(term):
        reasons.append("alpha_numeric_noise")
    if term in exact_noise_terms:
        reasons.append("custom_noise_term")
    for reason, pattern in NOISE_PATTERNS:
        if reason == "keyword_self_variant":
            continue
        if pattern.search(term):
            reasons.append(reason)
    theme_bucket = infer_theme_bucket(term, reasons, bucket_rules)
    return sorted(set(reasons)), theme_bucket


def compute_midterm_score(row: pd.Series) -> float:
    similarity = float(row["embedding_similarity"]) if pd.notna(row["embedding_similarity"]) else 0.0
    return math.log1p(float(row["term_doc_freq"])) * float(row["lift"]) * (1.0 + max(similarity, 0.0))


def flag_low_specificity_candidate(
    row: pd.Series,
    *,
    min_term_doc_freq: int = 500,
    max_lift: float = 1.05,
) -> bool:
    """Catch very frequent period terms that are barely more specific than background.

    This is intentionally narrow: it only targets report-facing shortlist noise such as
    generic verbs/nouns surfacing in a highly homogeneous period, while leaving normal
    mid-frequency semantic neighbors alone.
    """
    try:
        period_label = str(row.get("period_label", ""))
        term_doc_freq = float(row.get("term_doc_freq", 0) or 0)
        lift = float(row.get("lift", 0) or 0)
    except (TypeError, ValueError):
        return False

    return period_label != "ALL" and term_doc_freq >= min_term_doc_freq and lift <= max_lift


def prepare_candidate_frame(
    cooccurrence_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    selected_keywords: list[str],
    exact_noise_terms: set[str],
    bucket_rules: BucketRuleSet,
) -> pd.DataFrame:
    merged_neighbors = neighbor_df.rename(columns={"neighbor_term": "term"})
    neighbor_columns = ["keyword", "period_label", "term", "embedding_similarity", "neighbor_rank"]
    merged = cooccurrence_df.merge(
        merged_neighbors[neighbor_columns],
        on=["keyword", "period_label", "term"],
        how="left",
    )
    merged = merged[merged["keyword"].isin(selected_keywords)].copy()

    reason_payload = merged.apply(
        lambda row: classify_term(str(row["term"]), str(row["keyword"]), exact_noise_terms, bucket_rules),
        axis=1,
    )
    merged["auto_drop_reasons"] = reason_payload.map(lambda item: " | ".join(item[0]))
    merged["auto_theme_bucket"] = reason_payload.map(lambda item: item[1])
    merged["auto_noise_flag"] = merged["auto_drop_reasons"].ne("")
    merged["semantic_supported"] = merged["embedding_similarity"].notna()
    merged["midterm_score"] = merged.apply(compute_midterm_score, axis=1)
    merged["low_specificity_flag"] = merged.apply(flag_low_specificity_candidate, axis=1)
    merged["auto_keep_for_midterm"] = ~merged["auto_noise_flag"]
    merged["theme_bucket"] = merged["auto_theme_bucket"]
    merged["theme_bucket_source"] = "rule"
    return merged


def rank_shortlists(
    candidates: pd.DataFrame,
    *,
    top_n_all: int,
    top_n_period: int,
    min_doc_freq_all: int,
    min_doc_freq_period: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = candidates[
        (candidates["period_label"] == "ALL")
        & candidates["auto_keep_for_midterm"]
        & (candidates["term_doc_freq"] >= min_doc_freq_all)
    ].copy()
    overall = overall.sort_values(
        ["keyword", "midterm_score", "term_doc_freq", "lift"],
        ascending=[True, False, False, False],
    )
    overall["midterm_rank"] = overall.groupby("keyword").cumcount() + 1
    overall = overall.groupby("keyword", as_index=False, group_keys=False).head(top_n_all).reset_index(drop=True)

    per_period = candidates[
        (candidates["period_label"] != "ALL")
        & candidates["auto_keep_for_midterm"]
        & ~candidates["low_specificity_flag"]
        & (candidates["term_doc_freq"] >= min_doc_freq_period)
    ].copy()
    per_period = per_period.sort_values(
        ["keyword", "period_label", "midterm_score", "term_doc_freq", "lift"],
        ascending=[True, True, False, False, False],
    )
    per_period["midterm_rank"] = per_period.groupby(["keyword", "period_label"]).cumcount() + 1
    per_period = (
        per_period.groupby(["keyword", "period_label"], as_index=False, group_keys=False)
        .head(top_n_period)
        .reset_index(drop=True)
    )
    return overall, per_period


def rerank_overall_with_diversity(
    overall_candidates: pd.DataFrame,
    *,
    top_n_all: int,
    min_distinct_text_count: int = 3,
    max_top_text_share: float = 0.5,
) -> pd.DataFrame:
    if overall_candidates.empty:
        return overall_candidates.copy()

    preferred = overall_candidates[
        (overall_candidates["distinct_text_count"] >= min_distinct_text_count)
        & (overall_candidates["top_text_share"] <= max_top_text_share)
    ].copy()
    fallback = overall_candidates.drop(preferred.index).copy()

    preferred = preferred.sort_values(
        ["keyword", "midterm_score", "term_doc_freq", "lift"],
        ascending=[True, False, False, False],
    )
    fallback = fallback.sort_values(
        ["keyword", "midterm_score", "term_doc_freq", "lift"],
        ascending=[True, False, False, False],
    )

    rows: list[pd.DataFrame] = []
    for keyword, frame in overall_candidates.groupby("keyword", dropna=False):
        preferred_rows = preferred[preferred["keyword"] == keyword].head(top_n_all)
        missing = max(top_n_all - len(preferred_rows), 0)
        fallback_rows = fallback[fallback["keyword"] == keyword].head(missing)
        final_rows = pd.concat([preferred_rows, fallback_rows], ignore_index=True)
        final_rows["midterm_rank"] = range(1, len(final_rows) + 1)
        rows.append(final_rows)
    if not rows:
        return overall_candidates.head(0).copy()
    return pd.concat(rows, ignore_index=True)


def build_period_overview(period_shortlist: pd.DataFrame) -> pd.DataFrame:
    if period_shortlist.empty:
        return pd.DataFrame(
            columns=["keyword", "period_label", "doc_count_in_keyword", "lead_terms_for_midterm", "lead_theme_buckets"]
        )
    rows: list[dict[str, object]] = []
    for (keyword, period_label), frame in period_shortlist.groupby(["keyword", "period_label"], dropna=False):
        ordered = frame.sort_values("midterm_rank")
        rows.append(
            {
                "keyword": keyword,
                "period_label": period_label,
                "doc_count_in_keyword": int(ordered["doc_count_in_keyword"].iloc[0]),
                "lead_terms_for_midterm": " / ".join(
                    f"{term}({int(freq)})" for term, freq in zip(ordered["term"], ordered["term_doc_freq"])
                ),
                "lead_theme_buckets": " / ".join(ordered["theme_bucket"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["keyword", "period_label"]).reset_index(drop=True)


def build_context_trajectory(period_candidates: pd.DataFrame, *, lead_term_limit: int = 4) -> pd.DataFrame:
    if period_candidates.empty:
        return pd.DataFrame(
            columns=[
                "keyword",
                "period_label",
                "context_bucket",
                "doc_count_in_keyword",
                "context_term_count",
                "context_term_doc_freq_sum",
                "context_midterm_score_sum",
                "context_doc_freq_share",
                "context_score_share",
                "lead_terms",
            ]
        )

    base = period_candidates.copy()
    base["context_term_doc_freq_sum"] = base.groupby(["keyword", "period_label", "context_bucket"], dropna=False)[
        "term_doc_freq"
    ].transform("sum")
    base["context_midterm_score_sum"] = base.groupby(["keyword", "period_label", "context_bucket"], dropna=False)[
        "midterm_score"
    ].transform("sum")
    base["total_period_doc_freq"] = base.groupby(["keyword", "period_label"], dropna=False)["term_doc_freq"].transform("sum")
    base["total_period_midterm_score"] = base.groupby(["keyword", "period_label"], dropna=False)["midterm_score"].transform("sum")

    rows: list[dict[str, object]] = []
    for (keyword, period_label, context_bucket), frame in base.groupby(
        ["keyword", "period_label", "context_bucket"], dropna=False
    ):
        ordered = frame.sort_values(
            ["midterm_score", "term_doc_freq", "lift"],
            ascending=[False, False, False],
        )
        context_doc_freq = float(ordered["context_term_doc_freq_sum"].iloc[0])
        context_score = float(ordered["context_midterm_score_sum"].iloc[0])
        total_doc_freq = float(ordered["total_period_doc_freq"].iloc[0])
        total_score = float(ordered["total_period_midterm_score"].iloc[0])
        rows.append(
            {
                "keyword": keyword,
                "period_label": period_label,
                "context_bucket": context_bucket,
                "doc_count_in_keyword": int(ordered["doc_count_in_keyword"].iloc[0]),
                "context_term_count": int(len(ordered)),
                "context_term_doc_freq_sum": int(context_doc_freq),
                "context_midterm_score_sum": context_score,
                "context_doc_freq_share": float(context_doc_freq / total_doc_freq) if total_doc_freq else 0.0,
                "context_score_share": float(context_score / total_score) if total_score else 0.0,
                "lead_terms": " / ".join(ordered["term"].astype(str).head(lead_term_limit).tolist()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["keyword", "period_label", "context_score_share", "context_term_doc_freq_sum"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def build_context_shift_summary(context_trajectory: pd.DataFrame) -> pd.DataFrame:
    if context_trajectory.empty:
        return pd.DataFrame(
            columns=[
                "keyword",
                "context_bucket",
                "period_count",
                "first_period",
                "first_context_score_share",
                "latest_period",
                "latest_context_score_share",
                "score_share_change",
                "peak_period",
                "peak_context_score_share",
                "avg_context_score_share",
                "representative_terms_over_time",
            ]
        )

    rows: list[dict[str, object]] = []
    for (keyword, context_bucket), frame in context_trajectory.groupby(["keyword", "context_bucket"], dropna=False):
        ordered = frame.sort_values("period_label")
        peak_row = ordered.loc[ordered["context_score_share"].idxmax()]
        first_row = ordered.iloc[0]
        latest_row = ordered.iloc[-1]
        representative_terms: list[str] = []
        for term in ordered["lead_terms"].astype(str):
            for piece in [item.strip() for item in term.split("/")]:
                if piece and piece not in representative_terms:
                    representative_terms.append(piece)
                if len(representative_terms) >= 6:
                    break
            if len(representative_terms) >= 6:
                break
        rows.append(
            {
                "keyword": keyword,
                "context_bucket": context_bucket,
                "period_count": int(len(ordered)),
                "first_period": str(first_row["period_label"]),
                "first_context_score_share": float(first_row["context_score_share"]),
                "latest_period": str(latest_row["period_label"]),
                "latest_context_score_share": float(latest_row["context_score_share"]),
                "score_share_change": float(latest_row["context_score_share"] - first_row["context_score_share"]),
                "peak_period": str(peak_row["period_label"]),
                "peak_context_score_share": float(peak_row["context_score_share"]),
                "avg_context_score_share": float(ordered["context_score_share"].mean()),
                "representative_terms_over_time": " / ".join(representative_terms),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["keyword", "avg_context_score_share", "peak_context_score_share"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_noise_diagnostics(candidates: pd.DataFrame) -> pd.DataFrame:
    exploded = candidates[["keyword", "term", "auto_drop_reasons"]].copy()
    exploded = exploded[exploded["auto_drop_reasons"].ne("")]
    if exploded.empty:
        return pd.DataFrame(columns=["keyword", "drop_reason", "term_count", "example_terms"])
    exploded["drop_reason"] = exploded["auto_drop_reasons"].str.split(r"\s+\|\s+")
    exploded = exploded.explode("drop_reason")
    grouped = (
        exploded.groupby(["keyword", "drop_reason"], dropna=False)["term"]
        .agg(["count", lambda values: " / ".join(sorted(set(list(values)[:5])))])
        .reset_index()
        .rename(columns={"count": "term_count", "<lambda_0>": "example_terms"})
        .sort_values(["keyword", "term_count", "drop_reason"], ascending=[True, False, True])
    )
    return grouped.reset_index(drop=True)


def normalize_tokens(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
        if isinstance(converted, tuple):
            return [str(item) for item in converted]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body:
            return []
        return [part.strip().strip("'\"") for part in body.split(",") if part.strip()]
    return [text]


def coerce_engagement_value(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    match = NUMBER_RE.search(text.replace(",", ""))
    if not match:
        return 0
    number = float(match.group(1))
    if "万" in text:
        number *= 10000
    return int(number)


def build_example_lookup(
    tokenized_base: pd.DataFrame,
    requested_rows: pd.DataFrame,
    *,
    example_count: int,
    logger: OperationLogger,
) -> tuple[dict[tuple[str, str, str], list[ExampleRow]], dict[tuple[str, str, str], MatchStats]]:
    if requested_rows.empty:
        return {}, {}

    overall_terms: dict[str, set[str]] = defaultdict(set)
    period_terms: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in requested_rows[["keyword", "period_label", "term"]].itertuples(index=False):
        keyword = str(row.keyword)
        period_label = str(row.period_label)
        term = str(row.term)
        if period_label == "ALL":
            overall_terms[keyword].add(term)
        else:
            period_terms[(keyword, period_label)].add(term)

    example_map: dict[tuple[str, str, str], list[ExampleRow]] = defaultdict(list)
    text_counter_map: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    logger.log(
        "Scanning tokenized base once to collect representative text examples "
        f"for {len(requested_rows)} shortlisted term rows"
    )
    scan_start = time.perf_counter()
    missing_columns = [column for column in TOKENIZED_BASE_COLUMNS if column not in tokenized_base.columns]
    if missing_columns:
        raise ValueError(
            "tokenized_analysis_base.parquet is missing required columns for 09_prepare: "
            + ", ".join(missing_columns)
        )
    for row in tokenized_base.loc[:, TOKENIZED_BASE_COLUMNS].itertuples(index=False, name=None):
        keyword = str(row[0])
        period_label = str(row[1])
        text = row[2]
        tokens = row[3]
        likes = coerce_engagement_value(row[4])
        comments = coerce_engagement_value(row[5])
        reposts = coerce_engagement_value(row[6])

        token_values = set(normalize_tokens(tokens))
        if not token_values:
            continue

        matched_keys: list[tuple[str, str, str]] = []
        for term in token_values.intersection(overall_terms.get(keyword, set())):
            matched_keys.append((keyword, "ALL", term))
        for term in token_values.intersection(period_terms.get((keyword, period_label), set())):
            matched_keys.append((keyword, period_label, term))
        if not matched_keys:
            continue

        score = float(likes + comments * 2 + reposts * 3)
        preview = truncate_text(text)
        if not preview:
            continue

        payload = ExampleRow(
            score=score,
            text=preview,
            period_label=period_label,
            likes=likes,
            comments=comments,
            reposts=reposts,
        )
        for key in matched_keys:
            text_counter_map[key][preview] += 1
            existing = example_map[key]
            if any(item.text == payload.text for item in existing):
                continue
            existing.append(payload)
            existing.sort(key=lambda item: (item.score, item.likes, item.comments, item.reposts), reverse=True)
            del existing[example_count:]
    logger.log(f"Example scan finished in {time.perf_counter() - scan_start:.2f}s")
    stats_map: dict[tuple[str, str, str], MatchStats] = {}
    for key, counter in text_counter_map.items():
        total = sum(counter.values())
        top_share = (max(counter.values()) / total) if total else 0.0
        stats_map[key] = MatchStats(
            match_count=int(total),
            distinct_text_count=int(len(counter)),
            top_text_share=float(top_share),
        )
    return example_map, stats_map


def attach_examples(df: pd.DataFrame, example_lookup: dict[tuple[str, str, str], list[ExampleRow]], example_count: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rows: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        key = (str(row["keyword"]), str(row["period_label"]), str(row["term"]))
        examples = example_lookup.get(key, [])
        for index in range(example_count):
            prefix = f"example_{index + 1}"
            if index < len(examples):
                row[f"{prefix}_text"] = examples[index].text
                row[f"{prefix}_period"] = examples[index].period_label
                row[f"{prefix}_engagement"] = int(examples[index].score)
            else:
                row[f"{prefix}_text"] = ""
                row[f"{prefix}_period"] = ""
                row[f"{prefix}_engagement"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def attach_match_stats(df: pd.DataFrame, stats_lookup: dict[tuple[str, str, str], MatchStats]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rows: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        key = (str(row["keyword"]), str(row["period_label"]), str(row["term"]))
        stats = stats_lookup.get(key)
        row["matched_doc_count"] = int(stats.match_count) if stats else 0
        row["distinct_text_count"] = int(stats.distinct_text_count) if stats else 0
        row["top_text_share"] = float(stats.top_text_share) if stats else 0.0
        row["template_like_flag"] = bool(stats and stats.top_text_share > 0.5 and stats.distinct_text_count < 3)
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_bucket_override(
    *,
    keyword: str,
    period_label: str,
    term: str,
    overrides: list[BucketOverride],
) -> BucketOverride | None:
    for override in overrides:
        if override.keyword and override.keyword != keyword:
            continue
        if override.period_label and override.period_label != period_label:
            continue
        if override.term and override.term != term:
            continue
        return override
    return None


def attach_context_buckets(df: pd.DataFrame, bucket_rules: BucketRuleSet, overrides: list[BucketOverride]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    result = df.copy()
    result["auto_context_bucket"] = result["example_1_text"].map(
        lambda text: infer_bucket_from_markers(text, bucket_rules.context_rules)
    )
    result["context_bucket"] = result["auto_context_bucket"]
    result["context_bucket_source"] = "rule"
    result["bucket_override_note"] = ""

    resolved_overrides = result.apply(
        lambda row: resolve_bucket_override(
            keyword=str(row.get("keyword", "")),
            period_label=str(row.get("period_label", "")),
            term=str(row.get("term", "")),
            overrides=overrides,
        ),
        axis=1,
    )
    result["_bucket_override"] = resolved_overrides
    has_override = result["_bucket_override"].notna()
    if has_override.any():
        for index, override in result.loc[has_override, "_bucket_override"].items():
            if override.context_bucket:
                result.at[index, "context_bucket"] = override.context_bucket
                result.at[index, "context_bucket_source"] = "override"
            if override.theme_bucket:
                result.at[index, "theme_bucket"] = override.theme_bucket
                result.at[index, "theme_bucket_source"] = "override"
            result.at[index, "bucket_override_note"] = override.note
    result = result.drop(columns=["_bucket_override"])
    return result


def build_context_bucket_summary(overall_shortlist: pd.DataFrame) -> pd.DataFrame:
    if overall_shortlist.empty:
        return pd.DataFrame(columns=["keyword", "context_bucket", "term_count", "example_terms"])
    grouped = (
        overall_shortlist.groupby(["keyword", "context_bucket"], dropna=False)["term"]
        .agg(["count", lambda values: " / ".join(list(values)[:5])])
        .reset_index()
        .rename(columns={"count": "term_count", "<lambda_0>": "example_terms"})
        .sort_values(["keyword", "term_count", "context_bucket"], ascending=[True, False, True])
    )
    return grouped.reset_index(drop=True)


def build_coding_template(overall: pd.DataFrame, per_period: pd.DataFrame) -> pd.DataFrame:
    template = pd.concat([overall, per_period], ignore_index=True)
    if template.empty:
        return pd.DataFrame(
            columns=[
                "keyword",
                "period_label",
                "term",
                "midterm_rank",
                "term_doc_freq",
                "term_doc_rate",
                "lift",
                "embedding_similarity",
                "auto_theme_bucket",
                "theme_bucket",
                "theme_bucket_source",
                "auto_context_bucket",
                "context_bucket",
                "context_bucket_source",
                "bucket_override_note",
                "manual_theme_bucket",
                "manual_context_bucket",
                "manual_keep_for_midterm",
                "manual_quote_candidate",
                "manual_note",
            ]
        )
    template = template.drop_duplicates(subset=["keyword", "period_label", "term"]).copy()
    template["manual_theme_bucket"] = ""
    template["manual_context_bucket"] = ""
    template["manual_keep_for_midterm"] = ""
    template["manual_quote_candidate"] = ""
    template["manual_note"] = ""
    return template.reset_index(drop=True)


def build_bucket_override_template(overall: pd.DataFrame, per_period: pd.DataFrame) -> pd.DataFrame:
    template = pd.concat([overall, per_period], ignore_index=True)
    if template.empty:
        return pd.DataFrame(
            columns=[
                "keyword",
                "period_label",
                "term",
                "auto_context_bucket",
                "context_bucket",
                "override_context_bucket",
                "auto_theme_bucket",
                "theme_bucket",
                "override_theme_bucket",
                "enabled",
                "note",
                "example_1_text",
            ]
        )
    columns = [
        "keyword",
        "period_label",
        "term",
        "auto_context_bucket",
        "context_bucket",
        "auto_theme_bucket",
        "theme_bucket",
        "example_1_text",
    ]
    available = [column for column in columns if column in template.columns]
    result = template[available].drop_duplicates(subset=["keyword", "period_label", "term"]).copy()
    result["override_context_bucket"] = ""
    result["override_theme_bucket"] = ""
    result["enabled"] = ""
    result["note"] = ""
    ordered_columns = [
        "keyword",
        "period_label",
        "term",
        "auto_context_bucket",
        "context_bucket",
        "override_context_bucket",
        "auto_theme_bucket",
        "theme_bucket",
        "override_theme_bucket",
        "enabled",
        "note",
        "example_1_text",
    ]
    return result[[column for column in ordered_columns if column in result.columns]].reset_index(drop=True)


def render_markdown_summary(
    *,
    candidates: pd.DataFrame,
    overall_shortlist: pd.DataFrame,
    context_trajectory: pd.DataFrame,
    context_shift_summary: pd.DataFrame,
    noise_diagnostics: pd.DataFrame,
    context_bucket_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = ["# 09 Semantic Midterm Notes", ""]
    lines.append(
        f"- 原始候选词行数：{len(candidates)}；自动保留用于中期整理的候选词行数：{int(candidates['auto_keep_for_midterm'].sum())}。"
    )
    lines.append("- `keyword_cooccurrence.csv` 作为候选词池，`readouts/` 目录下的表用于阅读、修正和汇报。")
    lines.append("- 语义簇规则默认来自 `bert/config/semantic_bucket_rules.json`；如果要手工改桶，可以在 `bert/config/semantic_bucket_overrides.csv` 里写覆盖项。")
    lines.append("")

    lines.append("## 文件结构")
    lines.append("- `01_start_here/`：总体 shortlist、轨迹表、变化摘要和导读。")
    lines.append("- `02_period_detail/`：分期 shortlist 和 period 级汇总。")
    lines.append("- `03_workbench/`：候选总表、噪声诊断和人工改桶模板。")
    lines.append("- `99_meta/`：运行摘要和生成日志。")
    lines.append("")

    lines.append("## 生成逻辑")
    lines.append("- `keyword_cooccurrence.csv` 提供候选词池，`keyword_semantic_neighbors.csv` 提供语义邻居支持。")
    lines.append("- 整理阶段会回查 `tokenized_analysis_base.parquet`，补充代表文本、命中统计和文本重复度。")
    lines.append("- 自动分桶规则来自 `bert/config/semantic_bucket_rules.json`，人工覆盖来自 `bert/config/semantic_bucket_overrides.csv`。")
    lines.append("- 最终结果会分为总体 shortlist、分期 shortlist、语义轨迹、变化摘要和人工修正工作表。")
    lines.append("")

    if not noise_diagnostics.empty:
        lines.append("## 主要噪声来源")
        for _, row in noise_diagnostics.head(12).iterrows():
            lines.append(
                f"- `{row['keyword']}` / `{row['drop_reason']}`: {int(row['term_count'])} 个词，例子：{row['example_terms']}"
            )
        lines.append("")

    lines.append("## 关键词总体摘要")
    for keyword in normalize_cli_keywords(candidates["keyword"].unique().tolist()):
        subset = overall_shortlist[overall_shortlist["keyword"] == keyword].copy()
        if subset.empty:
            lines.append(f"- `{keyword}`: 自动清洗后没有留下足够稳定的总体词。")
            continue
        bucket_subset = context_bucket_summary[context_bucket_summary["keyword"] == keyword].copy()
        bucket_text = " / ".join(
            f"{bucket}({int(count)})"
            for bucket, count in zip(bucket_subset["context_bucket"], bucket_subset["term_count"])
        )
        lead_terms = " / ".join(subset["term"].head(5).tolist())
        lines.append(f"- `{keyword}`: 主要上下文桶 = {bucket_text}；代表词 = {lead_terms}")
    lines.append("")

    if not context_shift_summary.empty:
        lines.append("## 语义簇时间变化")
        for keyword in normalize_cli_keywords(context_shift_summary["keyword"].unique().tolist()):
            subset = context_shift_summary[context_shift_summary["keyword"] == keyword].head(3)
            if subset.empty:
                continue
            parts = []
            for _, row in subset.iterrows():
                parts.append(
                    f"{row['context_bucket']}({int(row['period_count'])}期, 峰值 {row['peak_period']}, "
                    f"最新占比 {float(row['latest_context_score_share']):.2f})"
                )
            lines.append(f"- `{keyword}`: {' / '.join(parts)}")
    lines.append("")

    lines.append("## 建议阅读顺序")
    lines.append("- `01_start_here/semantic_midterm_notes.md`")
    lines.append("- `01_start_here/semantic_keyword_overview.csv`")
    lines.append("- `01_start_here/semantic_context_trajectory.csv`")
    lines.append("- `01_start_here/semantic_context_shift_summary.csv`")
    lines.append("- `02_period_detail/semantic_period_shortlist.csv`")
    lines.append("- `03_workbench/semantic_bucket_override_template.csv` 和 `03_workbench/semantic_noise_diagnostics.csv` 仅在需要人工修正时继续查看。")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_readouts_index(output_dir: Path) -> None:
    lines = [
        "# 09 Readouts",
        "",
        "09 的整理结果分为四组：总体阅读、分期细表、人工修正工作台和运行元信息。",
        "",
        "## 01_start_here",
        "- `semantic_midterm_notes.md`：本轮整理的简要摘要。",
        "- `semantic_keyword_overview.csv`：总体 shortlist。",
        "- `semantic_context_trajectory.csv`：语义桶的时间轨迹。",
        "- `semantic_context_shift_summary.csv`：轨迹摘要。",
        "",
        "## 02_period_detail",
        "- `semantic_period_shortlist.csv`：分期 shortlist。",
        "- `semantic_period_overview.csv`：period 级别的汇总表。",
        "- `semantic_context_bucket_summary.csv`：总体桶分布摘要。",
        "",
        "## 03_workbench",
        "- `semantic_bucket_override_template.csv`：人工改桶候选。",
        "- `semantic_midterm_candidates.csv`：最全候选大表。",
        "- `semantic_midterm_coding_template.csv`：人工筛选工作表。",
        "- `semantic_noise_diagnostics.csv`：噪声来源统计。",
        "",
        "## 99_meta",
        "- `semantic_midterm_summary.json`：这次运行的统计摘要。",
        "- `semantic_midterm_operation_log.md`：本轮生成日志。",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    semantic_dir = Path(args.semantic_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = semantic_output_paths(semantic_dir)["readouts_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = OperationLogger(semantic_readout_path(output_dir, "semantic_midterm_operation_log.md"))
    logger.log(f"Start 09 midterm preparation from {semantic_dir}")
    bucket_rules = load_bucket_rules(args.bucket_rules_path)
    bucket_overrides = load_bucket_overrides(args.bucket_overrides_path)
    logger.log(
        f"Loaded bucket rules (theme={len(bucket_rules.theme_rules)}, context={len(bucket_rules.context_rules)}) "
        f"and manual overrides={len(bucket_overrides)}"
    )

    cooccurrence_path = resolve_semantic_artifact(semantic_dir, "keyword_cooccurrence.csv")
    neighbors_path = resolve_semantic_artifact(semantic_dir, "keyword_semantic_neighbors.csv")
    summary_path = semantic_dir / "semantic_analysis_summary.json"
    if not cooccurrence_path.exists() or not neighbors_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Semantic-analysis inputs are incomplete under the provided semantic_dir.")

    logger.log("Load semantic-analysis summary and raw 09 outputs")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cooccurrence_df = pd.read_csv(cooccurrence_path)
    neighbor_df = pd.read_csv(neighbors_path)
    selected_keywords = normalize_cli_keywords(args.keywords)
    exact_noise_terms = load_noise_terms(args.noise_terms_path)
    logger.log(
        f"Loaded cooccurrence={len(cooccurrence_df)}, neighbors={len(neighbor_df)}, "
        f"keywords={selected_keywords}, exact_noise_terms={len(exact_noise_terms)}"
    )

    logger.log("Build cleaned candidate table with auto flags, theme buckets, and report score")
    candidates = prepare_candidate_frame(cooccurrence_df, neighbor_df, selected_keywords, exact_noise_terms, bucket_rules)
    save_dataframe(candidates, semantic_readout_path(output_dir, "semantic_midterm_candidates.csv"))

    logger.log("Select report-ready shortlists for overall keywords and per-period reading")
    overall_shortlist, period_shortlist = rank_shortlists(
        candidates,
        top_n_all=args.top_n_all,
        top_n_period=args.top_n_period,
        min_doc_freq_all=args.min_doc_freq_all,
        min_doc_freq_period=args.min_doc_freq_period,
    )
    noise_diagnostics = build_noise_diagnostics(candidates)
    overall_candidates = candidates[
        (candidates["period_label"] == "ALL")
        & candidates["auto_keep_for_midterm"]
        & (candidates["term_doc_freq"] >= args.min_doc_freq_all)
    ].copy()
    period_context_candidates = candidates[
        (candidates["period_label"] != "ALL")
        & candidates["auto_keep_for_midterm"]
        & (candidates["term_doc_freq"] >= args.min_doc_freq_period)
    ].copy()

    tokenized_path = resolve_tokenized_analysis_base_path(summary, semantic_dir)
    logger.log(f"Load tokenized analysis base from {tokenized_path}")
    tokenized_base = pd.read_parquet(tokenized_path, columns=list(TOKENIZED_BASE_COLUMNS))

    requested_rows = pd.concat([overall_candidates, period_context_candidates, period_shortlist], ignore_index=True)
    requested_rows = requested_rows.drop_duplicates(subset=["keyword", "period_label", "term"]).reset_index(drop=True)
    example_lookup, stats_lookup = build_example_lookup(
        tokenized_base,
        requested_rows,
        example_count=args.example_count,
        logger=logger,
    )
    overall_candidates = attach_match_stats(overall_candidates, stats_lookup)
    overall_shortlist = rerank_overall_with_diversity(overall_candidates, top_n_all=args.top_n_all)
    period_context_candidates = attach_match_stats(period_context_candidates, stats_lookup)
    period_shortlist = attach_match_stats(period_shortlist, stats_lookup)
    overall_shortlist = attach_examples(overall_shortlist, example_lookup, args.example_count)
    period_context_candidates = attach_examples(period_context_candidates, example_lookup, args.example_count)
    period_shortlist = attach_examples(period_shortlist, example_lookup, args.example_count)
    coding_template = build_coding_template(overall_shortlist, period_shortlist)

    overall_shortlist = attach_context_buckets(overall_shortlist, bucket_rules, bucket_overrides)
    period_context_candidates = attach_context_buckets(period_context_candidates, bucket_rules, bucket_overrides)
    period_shortlist = attach_context_buckets(period_shortlist, bucket_rules, bucket_overrides)
    period_overview = build_period_overview(period_shortlist)
    context_trajectory = build_context_trajectory(period_context_candidates)
    context_shift_summary = build_context_shift_summary(context_trajectory)
    context_bucket_summary = build_context_bucket_summary(overall_shortlist)
    coding_template = attach_context_buckets(coding_template, bucket_rules, bucket_overrides)
    bucket_override_template = build_bucket_override_template(overall_shortlist, period_shortlist)

    logger.log("Write report-facing CSV/Markdown/JSON bundle for 09")
    save_dataframe(overall_shortlist, semantic_readout_path(output_dir, "semantic_keyword_overview.csv"))
    save_dataframe(period_shortlist, semantic_readout_path(output_dir, "semantic_period_shortlist.csv"))
    save_dataframe(period_overview, semantic_readout_path(output_dir, "semantic_period_overview.csv"))
    save_dataframe(context_trajectory, semantic_readout_path(output_dir, "semantic_context_trajectory.csv"))
    save_dataframe(context_shift_summary, semantic_readout_path(output_dir, "semantic_context_shift_summary.csv"))
    save_dataframe(noise_diagnostics, semantic_readout_path(output_dir, "semantic_noise_diagnostics.csv"))
    save_dataframe(coding_template, semantic_readout_path(output_dir, "semantic_midterm_coding_template.csv"))
    save_dataframe(bucket_override_template, semantic_readout_path(output_dir, "semantic_bucket_override_template.csv"))

    render_markdown_summary(
        candidates=candidates,
        overall_shortlist=overall_shortlist,
        context_trajectory=context_trajectory,
        context_shift_summary=context_shift_summary,
        noise_diagnostics=noise_diagnostics,
        context_bucket_summary=context_bucket_summary,
        output_path=semantic_readout_path(output_dir, "semantic_midterm_notes.md"),
    )

    summary_payload = {
        "semantic_dir": str(semantic_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selected_keywords": selected_keywords,
        "bucket_rules_path": str(Path(args.bucket_rules_path).resolve()),
        "bucket_override_path": str(Path(args.bucket_overrides_path).resolve()),
        "candidate_row_count": int(len(candidates)),
        "auto_keep_row_count": int(candidates["auto_keep_for_midterm"].sum()),
        "overall_shortlist_row_count": int(len(overall_shortlist)),
        "period_shortlist_row_count": int(len(period_shortlist)),
        "context_trajectory_row_count": int(len(context_trajectory)),
        "context_shift_summary_row_count": int(len(context_shift_summary)),
        "noise_diagnostic_row_count": int(len(noise_diagnostics)),
        "context_bucket_summary_row_count": int(len(context_bucket_summary)),
        "coding_template_row_count": int(len(coding_template)),
        "bucket_override_template_row_count": int(len(bucket_override_template)),
        "exact_noise_term_count": int(len(exact_noise_terms)),
        "bucket_override_count": int(len(bucket_overrides)),
    }
    save_json(semantic_readout_path(output_dir, "semantic_midterm_summary.json"), summary_payload)
    save_dataframe(context_bucket_summary, semantic_readout_path(output_dir, "semantic_context_bucket_summary.csv"))
    render_readouts_index(output_dir)
    logger.log(
        "Finished 09 midterm preparation "
        f"(overall_shortlist={len(overall_shortlist)}, period_shortlist={len(period_shortlist)})"
    )
    logger.save()
    sync_semantic_output_metadata(semantic_dir)
    try:
        refresh_broad_analysis_overview(semantic_dir)
    except Exception as exc:
        logger.log(f"Skipped broad-analysis overview refresh after 09_prepare: {exc}")
        logger.save()


if __name__ == "__main__":
    main()
