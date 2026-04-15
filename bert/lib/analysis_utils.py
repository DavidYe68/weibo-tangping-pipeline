from __future__ import annotations

import glob
import math
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence
from urllib.parse import unquote

import pandas as pd

from lib.data_utils import detect_text_column
from lib.io_utils import ensure_parent


DEFAULT_ANALYSIS_KEYWORDS = ["躺平", "摆烂", "佛系"]
TIME_CANDIDATES = ["publish_time", "发布时间", "created_at", "publish_time", "timestamp"]
KEYWORD_CANDIDATES = ["keyword_normalized", "keyword", "hit_keyword", "query_keyword"]
IP_CANDIDATES = ["ip_normalized", "ip", "IP", "ip_location", "IP属地"]
MISSING_IP_LABEL = "UNKNOWN_IP"

_WHITESPACE_RE = re.compile(r"\s+")
_EDGE_MARK_RE = re.compile(r"^[#＃﹟%]+|[#＃﹟%]+$")
_IP_PREFIX_RE = re.compile(r"^(?:ip|IP)\s*(?:属地|定位)?\s*[:：]?\s*")
_PERIOD_FREQ_MAP = {
    "month": "M",
    "quarter": "Q",
    "year": "Y",
}


def emit_progress(prefix: str, message: str) -> None:
    print(f"[{prefix}] {message}", file=sys.stderr, flush=True)


def resolve_emit(prefix: str, emit: Optional[Callable[[str], None]]) -> Callable[[str], None]:
    if emit is not None:
        return emit
    return lambda message: emit_progress(prefix, message)


def resolve_input_files(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern))
    if not files and pattern.startswith("/"):
        files = sorted(glob.glob(pattern.lstrip("/")))
    if not files:
        raise FileNotFoundError(f"No files matched input pattern: {pattern}")
    return files


def load_tabular_files(
    pattern: str,
    *,
    emit: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    emit_fn = emit or (lambda message: None)
    files = resolve_input_files(pattern)
    frames: list[pd.DataFrame] = []

    for file_path in files:
        path = Path(file_path)
        emit_fn(f"Loading {path}")
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix} ({path})")

        current = df.copy()
        current["__input_file"] = str(path.resolve())
        frames.append(current)

    return pd.concat(frames, ignore_index=True), files


def normalize_cli_keywords(values: Sequence[str] | None) -> list[str]:
    raw_values = list(values) if values else list(DEFAULT_ANALYSIS_KEYWORDS)
    resolved: list[str] = []
    for value in raw_values:
        canonical = canonicalize_keyword(value, DEFAULT_ANALYSIS_KEYWORDS)
        if canonical and canonical not in resolved:
            resolved.append(canonical)
    if not resolved:
        raise ValueError("No usable keywords were provided.")
    return resolved


def normalize_keyword_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = unquote(str(value))
    text = text.strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"nan", "none", "null", "na"}:
        return None

    text = text.replace("＃", "#").replace("﹟", "#")
    text = _WHITESPACE_RE.sub("", text)
    text = _EDGE_MARK_RE.sub("", text)
    text = text.strip("[]【】()（）<>《》\"'“”‘’")
    text = _EDGE_MARK_RE.sub("", text)
    text = text.strip()
    return text or None


def canonicalize_keyword(value: object, allowed_keywords: Sequence[str]) -> str | None:
    normalized = normalize_keyword_text(value)
    if not normalized:
        return None

    allowed = [keyword.strip() for keyword in allowed_keywords if str(keyword).strip()]
    if normalized in allowed:
        return normalized

    matches = [keyword for keyword in allowed if keyword and keyword in normalized]
    if len(matches) == 1:
        return matches[0]

    return normalized


def normalize_ip_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"nan", "none", "null", "na", "n/a", "unknown", "未知", "未显示", "-", "--"}:
        return None

    text = _IP_PREFIX_RE.sub("", text)
    text = text.strip(" ：:;；,，[]【】()（）<>《》\"'“”‘’")
    return text or None


def build_keyword_mask(series: pd.Series, allowed_keywords: Sequence[str]) -> pd.Series:
    allowed = set(normalize_cli_keywords(allowed_keywords))
    normalized = series.map(lambda value: canonicalize_keyword(value, DEFAULT_ANALYSIS_KEYWORDS))
    return normalized.isin(allowed)


def detect_existing_column(
    df: pd.DataFrame,
    forced: Optional[str],
    *,
    candidates: Sequence[str],
    label: str,
) -> str:
    if forced:
        if forced not in df.columns:
            raise ValueError(f"{label} column '{forced}' not found.")
        return forced

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(f"Could not detect {label} column automatically.")


def detect_keyword_column(df: pd.DataFrame, forced: Optional[str]) -> str:
    return detect_existing_column(df, forced, candidates=KEYWORD_CANDIDATES, label="keyword")


def detect_time_column(df: pd.DataFrame, forced: Optional[str]) -> str:
    return detect_existing_column(df, forced, candidates=TIME_CANDIDATES, label="time")


def detect_ip_column(
    df: pd.DataFrame,
    forced: Optional[str],
    *,
    required: bool = False,
) -> str | None:
    if forced:
        if forced not in df.columns:
            raise ValueError(f"IP column '{forced}' not found.")
        return forced

    for column in IP_CANDIDATES:
        if column in df.columns:
            return column

    if required:
        raise ValueError("Could not detect IP column automatically.")
    return None


def attach_ip_columns(df: pd.DataFrame, *, ip_col: Optional[str]) -> pd.DataFrame:
    result = df.copy()
    if ip_col and ip_col in result.columns:
        result["ip_raw"] = result[ip_col].astype("string")
    else:
        result["ip_raw"] = pd.Series(pd.NA, index=result.index, dtype="string")

    result["ip_normalized"] = result["ip_raw"].map(normalize_ip_text).fillna(MISSING_IP_LABEL)
    result["ip_missing"] = result["ip_normalized"] == MISSING_IP_LABEL
    return result


def attach_time_columns(
    df: pd.DataFrame,
    *,
    time_col: str,
    prefix: str = "publish",
) -> pd.DataFrame:
    result = df.copy()
    parsed = pd.to_datetime(result[time_col], errors="coerce")
    result[f"{prefix}_time"] = parsed
    result["year"] = parsed.dt.year.astype("Int64")
    result["year_month"] = parsed.dt.to_period("M").astype("string")
    result["year_quarter"] = parsed.dt.to_period("Q").astype("string")
    result["year_only"] = parsed.dt.to_period("Y").astype("string")
    return result


def period_column_name(granularity: str) -> str:
    if granularity == "month":
        return "year_month"
    if granularity == "quarter":
        return "year_quarter"
    if granularity == "year":
        return "year_only"
    raise ValueError(f"Unsupported time granularity: {granularity}")


def coerce_period_series(series: pd.Series, granularity: str) -> pd.Series:
    freq = _PERIOD_FREQ_MAP.get(granularity)
    if freq is None:
        raise ValueError(f"Unsupported time granularity: {granularity}")

    timestamps = pd.to_datetime(series, errors="coerce")
    periods = timestamps.dt.to_period(freq).astype("string")
    return periods.fillna("NA")


def sort_period_labels(labels: Iterable[str], granularity: str) -> list[str]:
    freq = _PERIOD_FREQ_MAP.get(granularity)
    if freq is None:
        raise ValueError(f"Unsupported time granularity: {granularity}")

    valid = [label for label in labels if label and label != "NA"]
    parsed = pd.PeriodIndex(valid, freq=freq)
    ordered = [str(period) for period in parsed.sort_values()]
    if any(label == "NA" for label in labels):
        ordered.append("NA")
    return ordered


def flatten_topic_terms(topic_terms: dict[int, list[tuple[str, float]]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for topic_id, pairs in topic_terms.items():
        for rank, (term, weight) in enumerate(pairs, start=1):
            rows.append(
                {
                    "topic_id": topic_id,
                    "term_rank": rank,
                    "term": term,
                    "term_weight": float(weight),
                }
            )
    return pd.DataFrame(rows)


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q):
        raise ValueError("Distributions must have the same length.")

    p_sum = sum(p)
    q_sum = sum(q)
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")

    p_norm = [value / p_sum for value in p]
    q_norm = [value / q_sum for value in q]
    m_norm = [(left + right) / 2.0 for left, right in zip(p_norm, q_norm)]

    def _kl_divergence(left: Sequence[float], right: Sequence[float]) -> float:
        total = 0.0
        for l_value, r_value in zip(left, right):
            if l_value <= 0:
                continue
            total += l_value * math.log(l_value / r_value, 2)
        return total

    return 0.5 * _kl_divergence(p_norm, m_norm) + 0.5 * _kl_divergence(q_norm, m_norm)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    raise ValueError(f"Unsupported output format: {path.suffix}")


def prepare_analysis_frame(
    df: pd.DataFrame,
    *,
    text_col: Optional[str],
    time_col: Optional[str],
    keyword_col: Optional[str],
    ip_col: Optional[str],
    keywords: Sequence[str],
    positive_label_col: str,
    positive_only: bool,
    min_confidence: Optional[float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    working = df.copy()

    resolved_text_col = detect_text_column(working, text_col, source_name="analysis input")
    resolved_keyword_col = detect_keyword_column(working, keyword_col)
    resolved_time_col = detect_time_column(working, time_col)
    resolved_ip_col = detect_ip_column(working, ip_col, required=False)

    working = attach_time_columns(working, time_col=resolved_time_col)
    working = attach_ip_columns(working, ip_col=resolved_ip_col)
    working["analysis_text"] = working[resolved_text_col].fillna("").astype("string").str.strip()
    working["keyword_raw"] = working[resolved_keyword_col].astype("string")
    working["keyword_normalized"] = working["keyword_raw"].map(
        lambda value: canonicalize_keyword(value, DEFAULT_ANALYSIS_KEYWORDS)
    )
    working["keyword_in_scope"] = working["keyword_normalized"].isin(set(normalize_cli_keywords(keywords)))

    if positive_label_col not in working.columns:
        raise ValueError(f"Prediction label column '{positive_label_col}' not found.")

    positive_series = pd.to_numeric(working[positive_label_col], errors="coerce").fillna(0).astype(int)
    working["is_positive_prediction"] = positive_series == 1

    if min_confidence is not None:
        confidence_col = "pred_prob_1" if "pred_prob_1" in working.columns else "pred_confidence"
        if confidence_col not in working.columns:
            raise ValueError(
                "--min_confidence was provided, but neither 'pred_prob_1' nor 'pred_confidence' exists."
            )
        confidence = pd.to_numeric(working[confidence_col], errors="coerce")
        working["passes_min_confidence"] = confidence >= min_confidence
    else:
        working["passes_min_confidence"] = True

    filtered = working[
        working["analysis_text"].ne("")
        & working["keyword_in_scope"]
        & working["passes_min_confidence"]
    ].copy()

    if positive_only:
        filtered = filtered[filtered["is_positive_prediction"]].copy()

    metadata = {
        "resolved_text_col": resolved_text_col,
        "resolved_keyword_col": resolved_keyword_col,
        "resolved_time_col": resolved_time_col,
        "resolved_ip_col": resolved_ip_col,
        "selected_keywords": normalize_cli_keywords(keywords),
        "positive_label_col": positive_label_col,
        "positive_only": bool(positive_only),
        "min_confidence": min_confidence,
        "rows_before_filter": int(len(working)),
        "rows_after_filter": int(len(filtered)),
        "missing_ip_rows_after_filter": int(filtered["ip_missing"].sum()) if "ip_missing" in filtered.columns else 0,
        "missing_ip_rate_after_filter": (
            float(filtered["ip_missing"].mean()) if len(filtered) > 0 and "ip_missing" in filtered.columns else 0.0
        ),
        "unique_ip_count_excluding_missing": (
            int(filtered.loc[~filtered["ip_missing"], "ip_normalized"].nunique()) if "ip_missing" in filtered.columns else 0
        ),
    }
    return filtered.reset_index(drop=True), metadata
