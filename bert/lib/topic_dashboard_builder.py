#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import html
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from lib.analysis_utils import resolve_emit, sort_period_labels
from lib.io_utils import save_json

ECHARTS_JS_URL = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"
ECHARTS_CHINA_MAP_JS_URL = "https://cdn.jsdelivr.net/npm/echarts@5/map/js/china.js"
DEFAULT_TOPIC_MODEL_DIR = Path("bert/artifacts/broad_analysis/topic_model_BAAI")
DEFAULT_TOPIC_VISUALS_DIR = Path("bert/artifacts/broad_analysis/topic_visuals")
LEADING_TOPIC_ID_RE = re.compile(r"^-?\d+_")
KEYWORD_CANDIDATES = ["keyword_normalized", "keyword", "hit_keyword", "query_keyword"]
SHORT_REGION_TO_FULL = {
    "北京": "北京市",
    "天津": "天津市",
    "上海": "上海市",
    "重庆": "重庆市",
    "河北": "河北省",
    "山西": "山西省",
    "辽宁": "辽宁省",
    "吉林": "吉林省",
    "黑龙江": "黑龙江省",
    "江苏": "江苏省",
    "浙江": "浙江省",
    "安徽": "安徽省",
    "福建": "福建省",
    "江西": "江西省",
    "山东": "山东省",
    "河南": "河南省",
    "湖北": "湖北省",
    "湖南": "湖南省",
    "广东": "广东省",
    "海南": "海南省",
    "四川": "四川省",
    "贵州": "贵州省",
    "云南": "云南省",
    "陕西": "陕西省",
    "甘肃": "甘肃省",
    "青海": "青海省",
    "台湾": "台湾省",
    "内蒙古": "内蒙古自治区",
    "广西": "广西壮族自治区",
    "西藏": "西藏自治区",
    "宁夏": "宁夏回族自治区",
    "新疆": "新疆维吾尔自治区",
    "香港": "香港特别行政区",
    "澳门": "澳门特别行政区",
}
FULL_REGION_TO_SHORT = {value: key for key, value in SHORT_REGION_TO_FULL.items()}
KEYWORD_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#0f766e",
    "#d97706",
    "#7c3aed",
    "#0891b2",
]
OTHER_COLOR = "#94a3b8"


def format_elapsed(start_time: float) -> str:
    return f"{time.perf_counter() - start_time:.2f}s"


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def shorten_label(text: str, *, max_length: int = 44) -> str:
    compact = " ".join(clean_text(text).split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def infer_period_granularity(labels: list[str]) -> str:
    usable = [label for label in labels if label]
    if not usable:
        return "month"
    if any("Q" in label for label in usable):
        return "quarter"
    if all(re.fullmatch(r"\d{4}", label) for label in usable):
        return "year"
    return "month"


def parse_representative_docs(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    docs: list[str] = []
    for item in parsed:
        candidate = clean_text(item)
        if candidate:
            docs.append(candidate)
    return docs


def normalize_topic_name(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = LEADING_TOPIC_ID_RE.sub("", text)
    return text.replace("_", " / ")


def resolve_candidate_path(base_dir: Path, raw_value: object) -> Path | None:
    text = clean_text(raw_value)
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def default_output_dir_from_summary(summary_path: Path) -> Path:
    summary_dir = summary_path.parent
    if summary_dir.parent.name == "topic_model":
        return summary_dir.parent.parent / "topic_visuals" / summary_dir.name
    return summary_dir.parent / "topic_visuals"


def normalize_region_name(value: object) -> str | None:
    text = clean_text(value)
    if not text or text.upper() == "UNKNOWN_IP":
        return None

    normalized = (
        text.replace("省", "")
        .replace("市", "")
        .replace("壮族自治区", "")
        .replace("回族自治区", "")
        .replace("维吾尔自治区", "")
        .replace("自治区", "")
        .replace("特别行政区", "")
        .replace(" ", "")
    )
    return SHORT_REGION_TO_FULL.get(normalized)


def short_region_label(full_name: str) -> str:
    return FULL_REGION_TO_SHORT.get(full_name, full_name)


def detect_keyword_column(frame: pd.DataFrame) -> str | None:
    for candidate in KEYWORD_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    return None


def compile_noise_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            raise ValueError(f"Invalid --noise_pattern regex: {pattern!r} ({exc})") from exc
    return compiled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a single-tabbed topic dashboard from BERTopic outputs."
    )
    parser.add_argument(
        "--from_summary",
        help="Path to 08_topic_model_bertopic.py's topic_model_summary.json.",
    )
    parser.add_argument(
        "--topic_info_path",
        help="Path to topic_info.csv. Ignored when --from_summary supplies it.",
    )
    parser.add_argument(
        "--topic_terms_path",
        help="Path to topic_terms.csv. Ignored when --from_summary supplies it.",
    )
    parser.add_argument(
        "--topic_share_by_period_path",
        help="Path to topic_share_by_period.csv. Ignored when --from_summary supplies it.",
    )
    parser.add_argument(
        "--topic_share_by_period_keyword_path",
        help="Path to topic_share_by_period_and_keyword.csv. Ignored when --from_summary supplies it.",
    )
    parser.add_argument(
        "--topic_share_by_ip_path",
        help="Path to topic_share_by_ip.csv. Ignored when --from_summary supplies it.",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory for generated dashboard outputs. When omitted with --from_summary, defaults to a sibling topic_visuals/ directory.",
    )
    parser.add_argument(
        "--top_n_topics",
        type=int,
        default=30,
        help="Number of head topics to foreground in overview and timeline charts.",
    )
    parser.add_argument(
        "--top_n_terms",
        type=int,
        default=10,
        help="Number of top terms to display per topic.",
    )
    parser.add_argument(
        "--min_period_docs",
        type=int,
        default=1000,
        help="Drop very sparse periods from timeline charts.",
    )
    parser.add_argument(
        "--min_topic_size",
        type=int,
        default=0,
        help="Display-only filter: hide topics with fewer than this many documents.",
    )
    parser.add_argument(
        "--noise_pattern",
        action="append",
        default=[],
        help="Display-only regex filter for noisy topics. May be passed multiple times.",
    )
    parser.add_argument(
        "--echarts_js_url",
        default=ECHARTS_JS_URL,
        help="URL for echarts.min.js.",
    )
    parser.add_argument(
        "--echarts_china_map_js_url",
        default=ECHARTS_CHINA_MAP_JS_URL,
        help="URL for the ECharts China map plugin.",
    )
    parser.add_argument("--wordcloud_top_n_topics", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--coding_template_top_n", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--echarts_wordcloud_js_url", help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_input_config(args: argparse.Namespace) -> dict[str, Any]:
    summary_payload: dict[str, Any] = {}
    summary_path: Path | None = None
    if args.from_summary:
        summary_path = Path(args.from_summary).expanduser().resolve()
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    summary_base_dir = summary_path.parent if summary_path else Path.cwd()
    topic_model_dir = resolve_candidate_path(summary_base_dir, summary_payload.get("output_dir")) or DEFAULT_TOPIC_MODEL_DIR.resolve()

    topic_info_path = (
        resolve_candidate_path(summary_base_dir, args.topic_info_path)
        or resolve_candidate_path(summary_base_dir, summary_payload.get("topic_info_path"))
        or (topic_model_dir / "topic_info.csv").resolve()
    )
    topic_terms_path = (
        resolve_candidate_path(summary_base_dir, args.topic_terms_path)
        or resolve_candidate_path(summary_base_dir, summary_payload.get("topic_terms_path"))
        or (topic_model_dir / "topic_terms.csv").resolve()
    )
    share_by_period_path = (
        resolve_candidate_path(summary_base_dir, args.topic_share_by_period_path)
        or resolve_candidate_path(summary_base_dir, summary_payload.get("topic_share_by_period_path"))
        or (topic_model_dir / "topic_share_by_period.csv").resolve()
    )
    share_by_period_keyword_path = (
        resolve_candidate_path(summary_base_dir, args.topic_share_by_period_keyword_path)
        or resolve_candidate_path(summary_base_dir, summary_payload.get("topic_share_by_period_and_keyword_path"))
        or (topic_model_dir / "topic_share_by_period_and_keyword.csv").resolve()
    )
    share_by_ip_path = (
        resolve_candidate_path(summary_base_dir, args.topic_share_by_ip_path)
        or resolve_candidate_path(summary_base_dir, summary_payload.get("topic_share_by_ip_path"))
        or (topic_model_dir / "topic_share_by_ip.csv").resolve()
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir_from_summary(summary_path).resolve()
        if summary_path
        else DEFAULT_TOPIC_VISUALS_DIR.resolve()
    )

    required_paths = {
        "topic_info_path": topic_info_path,
        "topic_terms_path": topic_terms_path,
        "topic_share_by_period_path": share_by_period_path,
        "topic_share_by_period_keyword_path": share_by_period_keyword_path,
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        missing_details = ", ".join(f"{name}={required_paths[name]}" for name in missing)
        raise FileNotFoundError(f"Missing required topic inputs: {missing_details}")

    return {
        "summary_path": summary_path,
        "summary_payload": summary_payload,
        "topic_model_dir": topic_model_dir,
        "topic_info_path": topic_info_path,
        "topic_terms_path": topic_terms_path,
        "topic_share_by_period_path": share_by_period_path,
        "topic_share_by_period_keyword_path": share_by_period_keyword_path,
        "topic_share_by_ip_path": share_by_ip_path,
        "output_dir": output_dir,
    }


def build_topic_term_lookup(topic_terms: pd.DataFrame) -> dict[int, list[dict[str, object]]]:
    if topic_terms.empty:
        return {}

    working = topic_terms.copy()
    working["topic_id"] = pd.to_numeric(working["topic_id"], errors="coerce").astype("Int64")
    working["term_rank"] = pd.to_numeric(working["term_rank"], errors="coerce")
    working["term_weight"] = pd.to_numeric(working["term_weight"], errors="coerce")
    working = working.dropna(subset=["topic_id", "term_rank", "term", "term_weight"]).copy()
    working = working.sort_values(["topic_id", "term_rank"])

    payload: dict[int, list[dict[str, object]]] = {}
    for topic_id, frame in working.groupby("topic_id"):
        topic_id_int = int(topic_id)
        payload[topic_id_int] = [
            {
                "term": clean_text(row.term),
                "term_rank": int(row.term_rank),
                "term_weight": float(row.term_weight),
            }
            for row in frame.itertuples(index=False)
            if clean_text(row.term)
        ]
    return payload


def build_keyword_count_lookup(share_by_period_keyword: pd.DataFrame) -> tuple[list[str], dict[int, dict[str, int]], dict[str, int]]:
    if share_by_period_keyword.empty:
        return [], {}, {}

    keyword_col = detect_keyword_column(share_by_period_keyword)
    if keyword_col is None:
        return [], {}, {}

    working = share_by_period_keyword.copy()
    working["topic_id"] = pd.to_numeric(working["topic_id"], errors="coerce").astype("Int64")
    working["doc_count"] = pd.to_numeric(working["doc_count"], errors="coerce").fillna(0).astype(int)
    working[keyword_col] = working[keyword_col].astype(str)
    working = working.dropna(subset=["topic_id"]).copy()
    working = working[working["topic_id"] >= 0].copy()

    grouped = (
        working.groupby(["topic_id", keyword_col], as_index=False)["doc_count"]
        .sum()
        .sort_values(["topic_id", "doc_count", keyword_col], ascending=[True, False, True])
    )
    keywords = sorted(grouped[keyword_col].dropna().astype(str).unique().tolist())

    topic_lookup: dict[int, dict[str, int]] = {}
    for row in grouped.itertuples(index=False):
        topic_lookup.setdefault(int(row.topic_id), {})[str(getattr(row, keyword_col))] = int(row.doc_count)

    keyword_totals = (
        grouped.groupby(keyword_col)["doc_count"]
        .sum()
        .sort_index()
        .to_dict()
    )
    return keywords, topic_lookup, {str(key): int(value) for key, value in keyword_totals.items()}


def build_period_lookup(
    share_by_period: pd.DataFrame,
    *,
    topic_ids: list[int],
    min_period_docs: int,
) -> tuple[list[str], dict[str, int], dict[int, dict[str, float]], dict[int, dict[str, int]]]:
    working = share_by_period.copy()
    working["topic_id"] = pd.to_numeric(working["topic_id"], errors="coerce").astype("Int64")
    working["doc_count"] = pd.to_numeric(working["doc_count"], errors="coerce").fillna(0).astype(int)
    working["doc_share"] = pd.to_numeric(working["doc_share"], errors="coerce").fillna(0.0)
    working["period_label"] = working["period_label"].astype(str)
    working = working.dropna(subset=["topic_id"]).copy()

    period_totals_series = working.groupby("period_label", as_index=True)["doc_count"].sum()
    period_totals = {str(period): int(total) for period, total in period_totals_series.items()}
    ordered_periods = sort_period_labels(list(period_totals.keys()), infer_period_granularity(list(period_totals.keys())))
    retained_periods = [period for period in ordered_periods if period_totals.get(period, 0) >= min_period_docs]

    share_lookup: dict[int, dict[str, float]] = {topic_id: {} for topic_id in topic_ids}
    count_lookup: dict[int, dict[str, int]] = {topic_id: {} for topic_id in topic_ids}

    subset = working[working["topic_id"].isin(topic_ids) & working["period_label"].isin(retained_periods)].copy()
    for row in subset.itertuples(index=False):
        topic_id = int(row.topic_id)
        period = str(row.period_label)
        share_lookup.setdefault(topic_id, {})[period] = float(row.doc_share) * 100.0
        count_lookup.setdefault(topic_id, {})[period] = int(row.doc_count)

    return retained_periods, {period: period_totals.get(period, 0) for period in retained_periods}, share_lookup, count_lookup


def build_peak_lookup(share_by_period: pd.DataFrame) -> dict[int, dict[str, object]]:
    if share_by_period.empty:
        return {}

    working = share_by_period.copy()
    working["topic_id"] = pd.to_numeric(working["topic_id"], errors="coerce").astype("Int64")
    working["doc_count"] = pd.to_numeric(working["doc_count"], errors="coerce").fillna(0).astype(int)
    working["doc_share"] = pd.to_numeric(working["doc_share"], errors="coerce").fillna(0.0)
    working = working.dropna(subset=["topic_id"]).copy()
    working = working[working["topic_id"] >= 0].copy()

    payload: dict[int, dict[str, object]] = {}
    for topic_id, frame in working.groupby("topic_id"):
        peak_row = frame.sort_values(["doc_count", "doc_share"], ascending=[False, False]).iloc[0]
        payload[int(topic_id)] = {
            "peak_period": clean_text(peak_row["period_label"]),
            "peak_doc_count": int(peak_row["doc_count"]),
            "peak_doc_share_pct": float(peak_row["doc_share"]) * 100.0,
        }
    return payload


def topic_matches_noise(
    *,
    machine_label: str,
    zh_label: str,
    top_terms: list[str],
    compiled_patterns: list[re.Pattern[str]],
) -> str | None:
    if not compiled_patterns:
        return None
    haystack = " | ".join(part for part in [machine_label, zh_label, " ".join(top_terms)] if part)
    for pattern in compiled_patterns:
        if pattern.search(haystack):
            return pattern.pattern
    return None


def build_topic_rows(
    topic_info: pd.DataFrame,
    *,
    topic_term_lookup: dict[int, list[dict[str, object]]],
    peak_lookup: dict[int, dict[str, object]],
    keyword_lookup: dict[int, dict[str, int]],
    keywords: list[str],
    keyword_colors: dict[str, str],
    min_topic_size: int,
    noise_patterns: list[re.Pattern[str]],
    top_n_terms: int,
) -> tuple[list[dict[str, object]], int, int]:
    info = topic_info.copy()
    info["Topic"] = pd.to_numeric(info["Topic"], errors="coerce").astype("Int64")
    info["Count"] = pd.to_numeric(info["Count"], errors="coerce").fillna(0).astype(int)

    outlier_count = int(info.loc[info["Topic"] == -1, "Count"].sum())
    clustered_total = int(info.loc[info["Topic"] >= 0, "Count"].sum())

    rows: list[dict[str, object]] = []
    for row in info.itertuples(index=False):
        if pd.isna(row.Topic):
            continue
        topic_id = int(row.Topic)
        if topic_id < 0:
            continue

        term_payload = topic_term_lookup.get(topic_id, [])[:top_n_terms]
        term_labels = [str(item["term"]) for item in term_payload if clean_text(item["term"])]
        if not term_labels:
            term_labels = [term for term in parse_representative_docs(getattr(row, "Representation", ""))[:top_n_terms] if term]

        machine_label = clean_text(getattr(row, "topic_label_machine", "")) or clean_text(getattr(row, "Name", ""))
        zh_label = clean_text(getattr(row, "topic_label_zh", ""))
        normalized_machine_label = normalize_topic_name(machine_label)

        if zh_label:
            label_body = zh_label
        elif term_labels:
            label_body = " / ".join(term_labels[:4])
        elif normalized_machine_label:
            label_body = normalized_machine_label
        else:
            label_body = f"Topic {topic_id}"

        topic_count = int(row.Count)
        peak = peak_lookup.get(topic_id, {})
        keyword_counts = keyword_lookup.get(topic_id, {})
        keyword_total = max(topic_count, sum(keyword_counts.values()))
        keyword_segments = []
        dominant_keyword = "其他"
        dominant_keyword_share_pct = 0.0
        for keyword in keywords:
            keyword_count = int(keyword_counts.get(keyword, 0))
            share_pct = (100.0 * keyword_count / keyword_total) if keyword_total else 0.0
            keyword_segments.append(
                {
                    "keyword": keyword,
                    "count": keyword_count,
                    "share_pct": round(share_pct, 4),
                    "color": keyword_colors[keyword],
                }
            )
            if share_pct > dominant_keyword_share_pct:
                dominant_keyword = keyword
                dominant_keyword_share_pct = share_pct

        other_count = max(keyword_total - sum(item["count"] for item in keyword_segments), 0)
        if other_count > 0:
            other_share = (100.0 * other_count / keyword_total) if keyword_total else 0.0
            keyword_segments.append(
                {
                    "keyword": "其他",
                    "count": other_count,
                    "share_pct": round(other_share, 4),
                    "color": OTHER_COLOR,
                }
            )
            if other_share > dominant_keyword_share_pct:
                dominant_keyword = "其他"
                dominant_keyword_share_pct = other_share

        keyword_share_summary = " | ".join(
            f"{segment['keyword']} {segment['share_pct']:.1f}%"
            for segment in keyword_segments
            if float(segment["share_pct"]) > 0
        )

        representative_docs = [
            shorten_label(doc, max_length=120)
            for doc in parse_representative_docs(getattr(row, "Representative_Docs", ""))[:2]
        ]

        exclusion_reason = ""
        if min_topic_size > 0 and topic_count < min_topic_size:
            exclusion_reason = f"count<{min_topic_size}"
        elif noise_patterns:
            matched_pattern = topic_matches_noise(
                machine_label=normalized_machine_label,
                zh_label=zh_label,
                top_terms=term_labels,
                compiled_patterns=noise_patterns,
            )
            if matched_pattern:
                exclusion_reason = f"noise:{matched_pattern}"

        rows.append(
            {
                "topic_id": topic_id,
                "topic_count": topic_count,
                "share_of_clustered_docs_pct": (100.0 * topic_count / clustered_total) if clustered_total else 0.0,
                "topic_label": f"T{topic_id} {shorten_label(label_body, max_length=34)}",
                "topic_label_full": f"T{topic_id} {label_body}",
                "topic_label_zh": zh_label,
                "topic_label_machine": normalized_machine_label,
                "top_terms": term_labels,
                "top_terms_display": term_labels[: min(6, len(term_labels))],
                "peak_period": clean_text(peak.get("peak_period", "")),
                "peak_doc_count": int(peak.get("peak_doc_count", 0)),
                "peak_doc_share_pct": float(peak.get("peak_doc_share_pct", 0.0)),
                "representative_docs": representative_docs,
                "keyword_segments": keyword_segments,
                "keyword_share_summary": keyword_share_summary,
                "dominant_keyword": dominant_keyword,
                "dominant_keyword_share_pct": float(dominant_keyword_share_pct),
                "exclusion_reason": exclusion_reason,
                "term_payload": term_payload,
            }
        )

    rows.sort(key=lambda item: (-int(item["topic_count"]), int(item["topic_id"])))
    return rows, clustered_total, outlier_count


def attach_period_series(
    topic_rows: list[dict[str, object]],
    *,
    periods: list[str],
    share_lookup: dict[int, dict[str, float]],
    count_lookup: dict[int, dict[str, int]],
) -> None:
    for row in topic_rows:
        topic_id = int(row["topic_id"])
        series_share = [round(float(share_lookup.get(topic_id, {}).get(period, 0.0)), 4) for period in periods]
        series_count = [int(count_lookup.get(topic_id, {}).get(period, 0)) for period in periods]
        row["series_share_pct"] = series_share
        row["series_doc_count"] = series_count
        row["sparkline"] = series_share
        row["search_text"] = " ".join(
            [
                str(row["topic_label_full"]),
                " ".join(str(term) for term in row["top_terms"]),
                str(row["dominant_keyword"]),
            ]
        ).lower()


def build_overview_topic_bars(topic_rows: list[dict[str, object]], *, top_n_topics: int) -> list[dict[str, object]]:
    head_rows = topic_rows[:top_n_topics]
    payload = [
        {
            "label": row["topic_label"],
            "label_full": row["topic_label_full"],
            "topic_count": int(row["topic_count"]),
            "share_clustered_pct": round(float(row["share_of_clustered_docs_pct"]), 4),
            "peak_period": row["peak_period"],
            "peak_doc_count": int(row["peak_doc_count"]),
            "top_terms": row["top_terms_display"],
            "kind": "topic",
        }
        for row in head_rows
    ]
    if len(topic_rows) > top_n_topics:
        tail_rows = topic_rows[top_n_topics:]
        payload.append(
            {
                "label": f"长尾剩余 ({len(tail_rows)} 个)",
                "label_full": f"长尾剩余 ({len(tail_rows)} 个主题)",
                "topic_count": int(sum(int(row["topic_count"]) for row in tail_rows)),
                "share_clustered_pct": round(sum(float(row["share_of_clustered_docs_pct"]) for row in tail_rows), 4),
                "peak_period": "",
                "peak_doc_count": 0,
                "top_terms": [],
                "kind": "tail",
            }
        )
    return payload


def build_keyword_pie(
    *,
    keyword_totals: dict[str, int],
    clustered_total: int,
) -> list[dict[str, object]]:
    if not keyword_totals:
        return []

    total_keywords = sum(keyword_totals.values())
    denominator = max(clustered_total, total_keywords)
    payload = [
        {
            "name": keyword,
            "value": int(count),
            "share_pct": round((100.0 * int(count) / denominator) if denominator else 0.0, 4),
        }
        for keyword, count in sorted(keyword_totals.items(), key=lambda item: (-item[1], item[0]))
    ]
    if clustered_total > total_keywords:
        other_count = clustered_total - total_keywords
        payload.append(
            {
                "name": "其他",
                "value": other_count,
                "share_pct": round((100.0 * other_count / denominator) if denominator else 0.0, 4),
            }
        )
    return payload


def build_keyword_profile_payload(
    share_by_period_keyword: pd.DataFrame,
    *,
    selected_rows: list[dict[str, object]],
) -> dict[str, object]:
    if share_by_period_keyword.empty or not selected_rows:
        return {
            "keywords": [],
            "stacked_topic_labels": [],
            "stacked_series": [],
            "radar_indicators": [],
            "radar_series": [],
            "summary_rows": [],
        }

    keyword_col = detect_keyword_column(share_by_period_keyword)
    if keyword_col is None:
        return {
            "keywords": [],
            "stacked_topic_labels": [],
            "stacked_series": [],
            "radar_indicators": [],
            "radar_series": [],
            "summary_rows": [],
        }

    selected_ids = [int(row["topic_id"]) for row in selected_rows[:8]]
    topic_label_lookup = {int(row["topic_id"]): str(row["topic_label"]) for row in selected_rows}

    working = share_by_period_keyword.copy()
    working["topic_id"] = pd.to_numeric(working["topic_id"], errors="coerce").astype("Int64")
    working["doc_count"] = pd.to_numeric(working["doc_count"], errors="coerce").fillna(0).astype(int)
    working = working.dropna(subset=["topic_id"]).copy()
    working = working[working["topic_id"] >= 0].copy()

    grouped = (
        working.groupby([keyword_col, "topic_id"], as_index=False)["doc_count"]
        .sum()
        .sort_values([keyword_col, "doc_count", "topic_id"], ascending=[True, False, True])
    )
    if grouped.empty:
        return {
            "keywords": [],
            "stacked_topic_labels": [],
            "stacked_series": [],
            "radar_indicators": [],
            "radar_series": [],
            "summary_rows": [],
        }

    grouped["keyword_total"] = grouped.groupby(keyword_col)["doc_count"].transform("sum")
    grouped["share_pct"] = grouped["doc_count"] / grouped["keyword_total"] * 100.0
    keywords = sorted(grouped[keyword_col].dropna().astype(str).unique().tolist())

    stacked_series = []
    for topic_id in selected_ids:
        topic_values = []
        for keyword in keywords:
            match = grouped[(grouped[keyword_col] == keyword) & (grouped["topic_id"] == topic_id)]
            topic_values.append(round(float(match["share_pct"].iloc[0]), 4) if not match.empty else 0.0)
        stacked_series.append(
            {
                "name": topic_label_lookup.get(topic_id, f"T{topic_id}"),
                "topic_id": topic_id,
                "data": topic_values,
            }
        )

    other_values = []
    for keyword in keywords:
        total_share = sum(float(series["data"][keywords.index(keyword)]) for series in stacked_series)
        other_values.append(round(max(100.0 - total_share, 0.0), 4))
    if any(value > 0 for value in other_values):
        stacked_series.append({"name": "其他", "topic_id": -999, "data": other_values})

    radar_topic_ids = selected_ids[: min(6, len(selected_ids))]
    radar_indicators = []
    for topic_id in radar_topic_ids:
        max_share = 0.0
        for keyword in keywords:
            match = grouped[(grouped[keyword_col] == keyword) & (grouped["topic_id"] == topic_id)]
            if not match.empty:
                max_share = max(max_share, float(match["share_pct"].iloc[0]))
        radar_indicators.append(
            {
                "name": shorten_label(topic_label_lookup.get(topic_id, f"T{topic_id}"), max_length=20),
                "max": max(10.0, round(max_share * 1.15, 2)),
                "topic_id": topic_id,
            }
        )

    radar_series = []
    for keyword in keywords:
        values = []
        for topic_id in radar_topic_ids:
            match = grouped[(grouped[keyword_col] == keyword) & (grouped["topic_id"] == topic_id)]
            values.append(round(float(match["share_pct"].iloc[0]), 4) if not match.empty else 0.0)
        radar_series.append({"name": keyword, "value": values})

    summary_rows = []
    for keyword in keywords:
        keyword_frame = grouped[grouped[keyword_col] == keyword].copy()
        if keyword_frame.empty:
            continue
        top_row = keyword_frame.iloc[0]
        summary_rows.append(
            {
                "keyword": keyword,
                "doc_count": int(keyword_frame["doc_count"].sum()),
                "top_topic_label": topic_label_lookup.get(int(top_row["topic_id"]), f"T{int(top_row['topic_id'])}"),
                "top_topic_share_pct": round(float(top_row["share_pct"]), 4),
                "top3_share_pct": round(float(keyword_frame["share_pct"].head(3).sum()), 4),
            }
        )

    return {
        "keywords": keywords,
        "stacked_topic_labels": [series["name"] for series in stacked_series],
        "stacked_series": stacked_series,
        "radar_indicators": radar_indicators,
        "radar_series": radar_series,
        "summary_rows": summary_rows,
    }


def build_evolution_payload(
    *,
    periods: list[str],
    selected_rows: list[dict[str, object]],
) -> dict[str, object]:
    line_rows = selected_rows[: min(8, len(selected_rows))]
    heatmap_rows = selected_rows[: min(20, len(selected_rows))]

    line_series = [
        {
            "name": row["topic_label"],
            "topic_id": int(row["topic_id"]),
            "data": row["series_share_pct"],
            "topic_count": int(row["topic_count"]),
        }
        for row in line_rows
    ]

    heatmap_topics = [row["topic_label"] for row in reversed(heatmap_rows)]
    heatmap_data: list[list[object]] = []
    heatmap_max = 0.0
    for y_index, row in enumerate(reversed(heatmap_rows)):
        for x_index, period in enumerate(periods):
            value = float(row["series_share_pct"][x_index]) if x_index < len(row["series_share_pct"]) else 0.0
            doc_count = int(row["series_doc_count"][x_index]) if x_index < len(row["series_doc_count"]) else 0
            heatmap_data.append([x_index, y_index, round(value, 4), doc_count])
            heatmap_max = max(heatmap_max, value)

    return {
        "periods": periods,
        "line_series": line_series,
        "heatmap_topics": heatmap_topics,
        "heatmap_data": heatmap_data,
        "heatmap_max": round(heatmap_max, 4),
    }


def build_geography_payload(
    share_by_ip: pd.DataFrame,
    *,
    selected_rows: list[dict[str, object]],
) -> dict[str, object]:
    empty_payload = {
        "has_data": False,
        "scopes": [],
        "map_series": {},
        "bar_series": {},
        "mapped_region_count": 0,
        "unmapped_doc_count": 0,
    }
    if share_by_ip.empty:
        return empty_payload

    working = share_by_ip.copy()
    if "ip_normalized" not in working.columns:
        return empty_payload
    working["topic_id"] = pd.to_numeric(working["topic_id"], errors="coerce").astype("Int64")
    working["doc_count"] = pd.to_numeric(working["doc_count"], errors="coerce").fillna(0).astype(int)
    working["region_full"] = working["ip_normalized"].map(normalize_region_name)

    mapped = working[working["region_full"].notna()].copy()
    if mapped.empty:
        return empty_payload

    scopes = [{"id": "overall", "label": "全部主题"}]
    map_series: dict[str, list[dict[str, object]]] = {}
    bar_series: dict[str, list[dict[str, object]]] = {}

    def build_scope(frame: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        grouped = (
            frame.groupby("region_full", as_index=False)["doc_count"]
            .sum()
            .sort_values(["doc_count", "region_full"], ascending=[False, True])
        )
        map_rows = [{"name": str(row.region_full), "value": int(row.doc_count)} for row in grouped.itertuples(index=False)]
        bar_rows = [
            {"name": short_region_label(str(row.region_full)), "value": int(row.doc_count)}
            for row in grouped.head(15).itertuples(index=False)
        ]
        return map_rows, bar_rows

    overall_map, overall_bar = build_scope(mapped)
    map_series["overall"] = overall_map
    bar_series["overall"] = overall_bar

    for row in selected_rows[: min(12, len(selected_rows))]:
        topic_id = int(row["topic_id"])
        scope_id = f"topic:{topic_id}"
        scopes.append({"id": scope_id, "label": row["topic_label"]})
        topic_frame = mapped[mapped["topic_id"] == topic_id].copy()
        scope_map, scope_bar = build_scope(topic_frame)
        map_series[scope_id] = scope_map
        bar_series[scope_id] = scope_bar

    return {
        "has_data": True,
        "scopes": scopes,
        "map_series": map_series,
        "bar_series": bar_series,
        "mapped_region_count": len({item["name"] for item in overall_map}),
        "unmapped_doc_count": int(working.loc[working["region_full"].isna(), "doc_count"].sum()),
    }


def build_display_table(
    topic_rows: list[dict[str, object]],
    *,
    filter_applied: bool,
) -> pd.DataFrame:
    rows = []
    for rank, row in enumerate(topic_rows, start=1):
        payload = {
            "topic_rank": rank,
            "topic_id": int(row["topic_id"]),
            "label": row["topic_label_full"],
            "doc_count": int(row["topic_count"]),
            "share_of_clustered_docs_pct": round(float(row["share_of_clustered_docs_pct"]), 4),
            "peak_period": row["peak_period"],
            "peak_doc_count": int(row["peak_doc_count"]),
            "peak_doc_share_pct": round(float(row["peak_doc_share_pct"]), 4),
            "top_terms": " / ".join(row["top_terms"]),
            "keyword_share": row["keyword_share_summary"],
            "dominant_keyword": row["dominant_keyword"],
            "dominant_keyword_share_pct": round(float(row["dominant_keyword_share_pct"]), 4),
            "display_filter_applied": "yes" if filter_applied else "no",
        }
        for segment in row["keyword_segments"]:
            if segment["keyword"] == "其他":
                continue
            payload[f"share_in_{segment['keyword']}_pct"] = round(float(segment["share_pct"]), 4)
        rows.append(payload)
    return pd.DataFrame(rows)


def render_metric_cards(metrics: list[dict[str, str]]) -> str:
    return "\n".join(
        f"""        <div class="metric">
          <div class="metric-label">{html.escape(item["label"])}</div>
          <div class="metric-value">{html.escape(item["value"])}</div>
          <div class="metric-detail">{html.escape(item["detail"])}</div>
        </div>"""
        for item in metrics
    )


def render_filter_pills(filters: list[str]) -> str:
    return "\n".join(f'        <span class="pill">{html.escape(value)}</span>' for value in filters if value)


def render_dashboard_html(
    *,
    title: str,
    subtitle: str,
    source_note: str,
    metrics: list[dict[str, str]],
    filters: list[str],
    payload: dict[str, object],
    script_urls: list[str],
) -> str:
    template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f3f6f8;
      color: #102033;
    }
    .page {
      max-width: 1520px;
      margin: 0 auto;
      padding: 24px 18px 36px;
    }
    .hero {
      padding: 0 0 20px;
      border-bottom: 1px solid #d7dee5;
    }
    .eyebrow {
      margin: 0 0 8px;
      color: #0f766e;
      font-size: 12px;
      line-height: 1.2;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    h1 {
      margin: 0;
      font-size: 32px;
      line-height: 1.1;
    }
    .subtitle {
      margin: 12px 0 0;
      max-width: 1040px;
      color: #526274;
      font-size: 15px;
      line-height: 1.55;
    }
    .source-note {
      margin: 10px 0 0;
      color: #667889;
      font-size: 13px;
      line-height: 1.45;
    }
    .metrics {
      margin-top: 18px;
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    .metric {
      border: 1px solid #d7dee5;
      border-radius: 6px;
      background: #ffffff;
      padding: 14px 16px;
    }
    .metric-label {
      color: #607180;
      font-size: 12px;
      line-height: 1.2;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .metric-value {
      margin-top: 8px;
      font-size: 26px;
      line-height: 1.1;
      font-weight: 700;
    }
    .metric-detail {
      margin-top: 6px;
      color: #607180;
      font-size: 13px;
      line-height: 1.45;
    }
    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 0 10px;
      border-radius: 6px;
      border: 1px solid #d7dee5;
      background: #ffffff;
      color: #425264;
      font-size: 13px;
    }
    .tab-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 18px 0 0;
    }
    .tab-button {
      border: 1px solid #cfd8e1;
      border-radius: 6px;
      background: #ffffff;
      color: #334155;
      min-height: 38px;
      padding: 0 14px;
      font-size: 14px;
      cursor: pointer;
    }
    .tab-button.is-active {
      background: #102033;
      color: #ffffff;
      border-color: #102033;
    }
    .tab-panel {
      display: none;
      padding-top: 18px;
    }
    .tab-panel.is-active {
      display: block;
    }
    .section-head {
      margin-bottom: 14px;
    }
    .section-title {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
    }
    .section-note {
      margin: 8px 0 0;
      max-width: 1040px;
      color: #607180;
      font-size: 14px;
      line-height: 1.55;
    }
    .chart-grid {
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .chart-grid.single {
      grid-template-columns: 1fr;
    }
    .panel {
      border: 1px solid #d7dee5;
      border-radius: 6px;
      background: #ffffff;
      padding: 14px;
    }
    .panel-title {
      margin: 0 0 10px;
      font-size: 17px;
      line-height: 1.35;
    }
    .panel-note {
      margin: 0 0 12px;
      color: #607180;
      font-size: 13px;
      line-height: 1.5;
    }
    .chart {
      width: 100%;
      height: 420px;
    }
    .chart.tall {
      height: 640px;
    }
    .chart.medium {
      height: 500px;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: end;
      margin-bottom: 12px;
    }
    .control {
      display: grid;
      gap: 6px;
      min-width: 200px;
    }
    .control label {
      color: #425264;
      font-size: 13px;
      font-weight: 600;
    }
    .control select,
    .control input {
      min-height: 38px;
      border: 1px solid #cfd8e1;
      border-radius: 6px;
      background: #ffffff;
      color: #102033;
      padding: 0 10px;
      font-size: 14px;
    }
    .summary-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .summary-card {
      border: 1px solid #d7dee5;
      border-radius: 6px;
      background: #ffffff;
      padding: 12px;
    }
    .summary-card h3 {
      margin: 0;
      font-size: 15px;
      line-height: 1.35;
    }
    .summary-card p {
      margin: 8px 0 0;
      color: #607180;
      font-size: 13px;
      line-height: 1.5;
    }
    .browser-meta {
      margin-bottom: 12px;
      color: #607180;
      font-size: 13px;
      line-height: 1.5;
    }
    .topic-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }
    .topic-card {
      border: 1px solid #d7dee5;
      border-radius: 6px;
      background: #ffffff;
      padding: 14px;
    }
    .topic-card-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }
    .topic-card-title {
      margin: 0;
      font-size: 16px;
      line-height: 1.35;
    }
    .topic-card-count {
      flex: 0 0 auto;
      color: #102033;
      font-size: 14px;
      font-weight: 700;
    }
    .topic-meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
      color: #607180;
      font-size: 13px;
      line-height: 1.45;
    }
    .sparkline {
      margin-top: 12px;
      height: 42px;
    }
    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 12px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 0 8px;
      border-radius: 6px;
      background: #eef2f7;
      color: #334155;
      font-size: 12px;
      line-height: 1.2;
    }
    .keyword-bar {
      display: flex;
      width: 100%;
      min-height: 10px;
      margin-top: 12px;
      overflow: hidden;
      border-radius: 6px;
      background: #e2e8f0;
    }
    .keyword-segment {
      min-height: 10px;
    }
    .keyword-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
      color: #607180;
      font-size: 12px;
      line-height: 1.35;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .legend-swatch {
      width: 10px;
      height: 10px;
      border-radius: 2px;
      flex: 0 0 auto;
    }
    details {
      margin-top: 12px;
    }
    summary {
      cursor: pointer;
      color: #334155;
      font-size: 13px;
      font-weight: 600;
    }
    .doc-list {
      margin: 10px 0 0;
      padding-left: 18px;
      color: #607180;
      font-size: 13px;
      line-height: 1.55;
    }
    .doc-list li + li {
      margin-top: 8px;
    }
    .empty-state {
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 180px;
      border: 1px dashed #cbd5df;
      border-radius: 6px;
      background: #f8fafc;
      color: #607180;
      font-size: 14px;
      line-height: 1.5;
      text-align: center;
      padding: 18px;
    }
    @media (max-width: 980px) {
      .chart-grid {
        grid-template-columns: 1fr;
      }
      .chart.tall {
        height: 520px;
      }
      .chart.medium {
        height: 420px;
      }
      .chart {
        height: 360px;
      }
      .page {
        padding: 18px 14px 28px;
      }
      h1 {
        font-size: 28px;
      }
    }
  </style>
__SCRIPT_TAGS__
</head>
<body>
  <div class="page">
    <header class="hero">
      <p class="eyebrow">BERTopic Dashboard</p>
      <h1>__TITLE__</h1>
      <p class="subtitle">__SUBTITLE__</p>
      <p class="source-note">__SOURCE_NOTE__</p>
      <div class="metrics">
__METRICS__
      </div>
      <div class="pill-row">
__FILTERS__
      </div>
    </header>

    <nav class="tab-bar" aria-label="dashboard tabs">
      <button class="tab-button is-active" type="button" data-tab-target="overview">总览</button>
      <button class="tab-button" type="button" data-tab-target="keyword">关键词画像</button>
      <button class="tab-button" type="button" data-tab-target="timeline">时间演化</button>
      <button class="tab-button" type="button" data-tab-target="geography">地域分布</button>
      <button class="tab-button" type="button" data-tab-target="browser">主题浏览</button>
    </nav>

    <section class="tab-panel is-active" data-tab-panel="overview">
      <div class="section-head">
        <h2 class="section-title">总览</h2>
        <p class="section-note">先看离群占比、头部主题和长尾结构，再看关键词体量与每期文档量。这里的展示过滤只影响看板，不会改动 08 生成的原始 CSV。</p>
      </div>
      <div class="chart-grid">
        <div class="panel">
          <h3 class="panel-title">Top 主题 + 长尾桶</h3>
          <p class="panel-note">默认突出头部主题，并把剩余主题聚成一个长尾桶，直接暴露“主题太碎”还是“头部很稳”。</p>
          <div id="overview-topic-chart" class="chart tall"></div>
        </div>
        <div class="panel">
          <h3 class="panel-title">关键词体量</h3>
          <p class="panel-note">按关键词汇总文档量，用来判断“躺平 / 摆烂 / 佛系”三组语料的规模是否失衡。</p>
          <div id="overview-keyword-chart" class="chart medium"></div>
        </div>
      </div>
      <div class="chart-grid single">
        <div class="panel">
          <h3 class="panel-title">每期文档量</h3>
          <p class="panel-note">这里按已聚类文档统计每期体量，方便解释时间窗的波峰波谷和后续热力图的底盘。</p>
          <div id="overview-period-chart" class="chart"></div>
        </div>
      </div>
    </section>

    <section class="tab-panel" data-tab-panel="keyword">
      <div class="section-head">
        <h2 class="section-title">关键词画像</h2>
        <p class="section-note">一张堆叠柱看每个关键词主要被哪些主题吸附；一张雷达看三组关键词在共享主题上的形状差异。</p>
      </div>
      <div class="chart-grid">
        <div class="panel">
          <h3 class="panel-title">主题占比堆叠</h3>
          <p class="panel-note">以关键词为横轴，按“该关键词内部的主题占比”堆叠。适合回答某个关键词是不是被少数主题垄断。</p>
          <div id="keyword-stacked-chart" class="chart medium"></div>
        </div>
        <div class="panel">
          <h3 class="panel-title">关键词雷达对照</h3>
          <p class="panel-note">挑出几条共享主题轴，直接比较三组关键词的话语轮廓是否重叠。</p>
          <div id="keyword-radar-chart" class="chart medium"></div>
        </div>
      </div>
      <div class="panel" style="margin-top: 14px;">
        <h3 class="panel-title">关键词摘要</h3>
        <p class="panel-note">看每个关键词最强的头部主题，以及前三主题合计吸纳了多少文档。</p>
        <div id="keyword-summary" class="summary-grid"></div>
      </div>
    </section>

    <section class="tab-panel" data-tab-panel="timeline">
      <div class="section-head">
        <h2 class="section-title">时间演化</h2>
        <p class="section-note">折线看重点主题的动态轨迹，热力图看主题与时段的整体耦合。两个视角一起用，既能讲故事，也能看全局。</p>
      </div>
      <div class="chart-grid">
        <div class="panel">
          <h3 class="panel-title">重点主题折线</h3>
          <p class="panel-note">默认取头部主题，纵轴是该时段内的主题占比。</p>
          <div id="timeline-line-chart" class="chart medium"></div>
        </div>
        <div class="panel">
          <h3 class="panel-title">时段 × 主题热力图</h3>
          <p class="panel-note">一眼看出哪些主题在某段时间突然抬头，哪些主题只是长期底噪。</p>
          <div id="timeline-heatmap-chart" class="chart medium"></div>
        </div>
      </div>
    </section>

    <section class="tab-panel" data-tab-panel="geography">
      <div class="section-head">
        <h2 class="section-title">地域分布</h2>
        <p class="section-note">保留省级热力图，同时支持切换到头部主题，快速看某个主题是不是被少数地区拉高。</p>
      </div>
      <div class="panel">
        <div class="toolbar">
          <div class="control">
            <label for="geo-scope-select">查看范围</label>
            <select id="geo-scope-select"></select>
          </div>
        </div>
        <div class="chart-grid">
          <div id="geo-map-chart" class="chart medium"></div>
          <div id="geo-bar-chart" class="chart medium"></div>
        </div>
      </div>
    </section>

    <section class="tab-panel" data-tab-panel="browser">
      <div class="section-head">
        <h2 class="section-title">主题浏览</h2>
        <p class="section-note">按需排序、搜索、抽样浏览主题卡片。卡片同时给出文档量、峰值时间、top terms、时间 sparkline 和关键词构成。</p>
      </div>
      <div class="panel">
        <div class="toolbar">
          <div class="control">
            <label for="browser-sort-select">排序</label>
            <select id="browser-sort-select">
              <option value="count_desc">文档量</option>
              <option value="peak_desc">峰值期文档量</option>
              <option value="keyword_desc">关键词集中度</option>
              <option value="id_asc">Topic ID</option>
            </select>
          </div>
          <div class="control">
            <label for="browser-limit-select">显示数量</label>
            <select id="browser-limit-select">
              <option value="30">30</option>
              <option value="60" selected>60</option>
              <option value="120">120</option>
              <option value="all">全部</option>
            </select>
          </div>
          <div class="control">
            <label for="browser-search-input">搜索标签或 terms</label>
            <input id="browser-search-input" type="text" placeholder="例如：佛系、第五人格、减肥">
          </div>
        </div>
        <div id="browser-meta" class="browser-meta"></div>
        <div id="topic-browser-grid" class="topic-grid"></div>
      </div>
    </section>
  </div>

  <script>
const dashboardPayload = __PAYLOAD__;

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatCount(value) {
  return Number(value || 0).toLocaleString("zh-CN");
}

function formatPct(value, digits = 1) {
  return `${Number(value || 0).toFixed(digits)}%`;
}

function setEmptyState(nodeId, message) {
  const node = document.getElementById(nodeId);
  if (!node) {
    return;
  }
  node.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
}

const chartInstances = [];
function registerChart(chart) {
  chartInstances.push(chart);
  return chart;
}

function resizeCharts() {
  chartInstances.forEach((chart) => {
    try {
      chart.resize();
    } catch (error) {
      console.warn("resize chart failed", error);
    }
  });
}

window.addEventListener("resize", resizeCharts);

const tabButtons = Array.from(document.querySelectorAll("[data-tab-target]"));
const tabPanels = Array.from(document.querySelectorAll("[data-tab-panel]"));
const initializedTabs = new Set();

function activateTab(name) {
  tabButtons.forEach((button) => {
    const active = button.dataset.tabTarget === name;
    button.classList.toggle("is-active", active);
  });
  tabPanels.forEach((panel) => {
    const active = panel.dataset.tabPanel === name;
    panel.classList.toggle("is-active", active);
  });
  if (!initializedTabs.has(name)) {
    renderers[name]();
    initializedTabs.add(name);
  }
  window.setTimeout(resizeCharts, 0);
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => activateTab(button.dataset.tabTarget));
});

function renderOverviewTab() {
  const topicBars = dashboardPayload.overview.topic_bars || [];
  if (!topicBars.length) {
    setEmptyState("overview-topic-chart", "没有可展示的主题。");
  } else {
    const chart = registerChart(echarts.init(document.getElementById("overview-topic-chart")));
    const rows = [...topicBars].reverse();
    chart.setOption({
      animationDuration: 400,
      grid: { left: 220, right: 24, top: 20, bottom: 24 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: (params) => {
          const row = params[0].data.row;
          const terms = row.top_terms && row.top_terms.length ? row.top_terms.join(" / ") : "NA";
          return `<strong>${escapeHtml(row.label_full)}</strong><br>` +
            `文档量: ${formatCount(row.topic_count)}<br>` +
            `占已聚类文档: ${formatPct(row.share_clustered_pct, 2)}<br>` +
            (row.peak_period ? `峰值时间: ${escapeHtml(row.peak_period)}<br>` : "") +
            (row.peak_doc_count ? `峰值文档量: ${formatCount(row.peak_doc_count)}<br>` : "") +
            `Top terms: ${escapeHtml(terms)}`;
        }
      },
      xAxis: {
        type: "value",
        axisLabel: { color: "#425264" },
        splitLine: { lineStyle: { color: "#dde4eb" } }
      },
      yAxis: {
        type: "category",
        data: rows.map((row) => row.label),
        axisLabel: { color: "#425264", width: 200, overflow: "truncate" },
        axisLine: { lineStyle: { color: "#c8d2dc" } }
      },
      series: [{
        type: "bar",
        barMaxWidth: 28,
        data: rows.map((row) => ({
          value: row.topic_count,
          row,
          itemStyle: { color: row.kind === "tail" ? "#d97706" : "#2563eb" }
        }))
      }]
    });
  }

  const keywordPie = dashboardPayload.overview.keyword_pie || [];
  if (!keywordPie.length) {
    setEmptyState("overview-keyword-chart", "没有关键词占比数据。");
  } else {
    const chart = registerChart(echarts.init(document.getElementById("overview-keyword-chart")));
    chart.setOption({
      tooltip: {
        trigger: "item",
        formatter: (params) => `${escapeHtml(params.data.name)}<br>文档量: ${formatCount(params.data.value)}<br>占比: ${formatPct(params.data.share_pct, 2)}`
      },
      legend: {
        bottom: 0,
        itemWidth: 12,
        itemHeight: 12,
        textStyle: { color: "#425264" }
      },
      series: [{
        type: "pie",
        radius: ["36%", "68%"],
        center: ["50%", "46%"],
        minAngle: 4,
        label: {
          formatter: (params) => `${params.data.name}\n${formatPct(params.data.share_pct, 1)}`
        },
        data: keywordPie
      }]
    });
  }

  const periodTotals = dashboardPayload.overview.period_totals || [];
  if (!periodTotals.length) {
    setEmptyState("overview-period-chart", "没有可展示的时段数据。");
  } else {
    const chart = registerChart(echarts.init(document.getElementById("overview-period-chart")));
    chart.setOption({
      tooltip: {
        trigger: "axis",
        formatter: (params) => {
          const point = params[0].data.row;
          return `${escapeHtml(point.period)}<br>文档量: ${formatCount(point.doc_count)}`;
        }
      },
      grid: { left: 56, right: 20, top: 24, bottom: 36 },
      xAxis: {
        type: "category",
        data: periodTotals.map((item) => item.period),
        axisLabel: { color: "#425264", rotate: periodTotals.length > 18 ? 35 : 0 },
        axisLine: { lineStyle: { color: "#c8d2dc" } }
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#425264" },
        splitLine: { lineStyle: { color: "#dde4eb" } }
      },
      series: [{
        type: "line",
        smooth: true,
        symbolSize: 7,
        lineStyle: { width: 3, color: "#0f766e" },
        itemStyle: { color: "#0f766e" },
        areaStyle: { color: "rgba(15, 118, 110, 0.14)" },
        data: periodTotals.map((item) => ({ value: item.doc_count, row: item }))
      }]
    });
  }
}

function renderKeywordTab() {
  const keywordPayload = dashboardPayload.keyword_profile || {};
  if (!(keywordPayload.keywords || []).length || !(keywordPayload.stacked_series || []).length) {
    setEmptyState("keyword-stacked-chart", "没有可展示的关键词画像。");
    setEmptyState("keyword-radar-chart", "没有可展示的关键词画像。");
    document.getElementById("keyword-summary").innerHTML = '<div class="empty-state">没有关键词摘要。</div>';
    return;
  }

  const stackedChart = registerChart(echarts.init(document.getElementById("keyword-stacked-chart")));
    stackedChart.setOption({
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (params) => {
        const lines = [`<strong>${escapeHtml(params[0].axisValue)}</strong>`];
        params.forEach((item) => {
          lines.push(`${escapeHtml(item.seriesName)}: ${formatPct(item.data, 2)}`);
        });
        return lines.join("<br>");
      }
    },
    legend: {
      top: 0,
      textStyle: { color: "#425264" }
    },
    grid: { left: 56, right: 24, top: 48, bottom: 30 },
    xAxis: {
      type: "category",
      data: keywordPayload.keywords,
      axisLabel: { color: "#425264" },
      axisLine: { lineStyle: { color: "#c8d2dc" } }
    },
    yAxis: {
      type: "value",
      max: 100,
      axisLabel: { color: "#425264", formatter: (value) => `${value}%` },
      splitLine: { lineStyle: { color: "#dde4eb" } }
    },
      series: keywordPayload.stacked_series.map((series, index) => ({
        name: series.name,
        type: "bar",
        stack: "keyword",
        emphasis: { focus: "series" },
        itemStyle: {
        color: series.name === "其他" ? "#94a3b8" : dashboardPayload.topic_palette[index % dashboardPayload.topic_palette.length]
        },
        data: series.data
      }))
    });

  if (!(keywordPayload.radar_indicators || []).length) {
    setEmptyState("keyword-radar-chart", "没有足够的共享主题来绘制雷达图。");
  } else {
    const radarChart = registerChart(echarts.init(document.getElementById("keyword-radar-chart")));
    radarChart.setOption({
      tooltip: {
        trigger: "item"
      },
      legend: {
        top: 0,
        textStyle: { color: "#425264" }
      },
      radar: {
        indicator: keywordPayload.radar_indicators.map((item) => ({ name: item.name, max: item.max })),
        radius: "64%",
        axisName: { color: "#334155" },
        splitLine: { lineStyle: { color: "#dbe4ed" } },
        splitArea: { areaStyle: { color: ["#ffffff", "#f8fafc"] } }
      },
      series: [{
        type: "radar",
        data: keywordPayload.radar_series.map((series, index) => ({
          name: series.name,
          value: series.value,
          areaStyle: { color: `${dashboardPayload.keyword_colors[index % dashboardPayload.keyword_colors.length]}22` },
          lineStyle: { color: dashboardPayload.keyword_colors[index % dashboardPayload.keyword_colors.length], width: 2 },
          itemStyle: { color: dashboardPayload.keyword_colors[index % dashboardPayload.keyword_colors.length] }
        }))
      }]
    });
  }

  const summaryRoot = document.getElementById("keyword-summary");
  const summaryRows = keywordPayload.summary_rows || [];
  if (!summaryRows.length) {
    summaryRoot.innerHTML = '<div class="empty-state">没有关键词摘要。</div>';
  } else {
    summaryRoot.innerHTML = summaryRows.map((row) => `
      <article class="summary-card">
        <h3>${escapeHtml(row.keyword)}</h3>
        <p>文档量：${formatCount(row.doc_count)}</p>
        <p>头部主题：${escapeHtml(row.top_topic_label)}</p>
        <p>头部主题占比：${formatPct(row.top_topic_share_pct, 2)}</p>
        <p>前三主题合计：${formatPct(row.top3_share_pct, 2)}</p>
      </article>
    `).join("");
  }
}

function renderTimelineTab() {
  const timelinePayload = dashboardPayload.timeline || {};
  if (!(timelinePayload.periods || []).length || !(timelinePayload.line_series || []).length) {
    setEmptyState("timeline-line-chart", "没有可展示的时间序列。");
    setEmptyState("timeline-heatmap-chart", "没有可展示的时间序列。");
    return;
  }

  const lineChart = registerChart(echarts.init(document.getElementById("timeline-line-chart")));
  lineChart.setOption({
    tooltip: {
      trigger: "axis",
      formatter: (params) => {
        const lines = [`<strong>${escapeHtml(params[0].axisValue)}</strong>`];
        params.forEach((item) => {
          lines.push(`${escapeHtml(item.seriesName)}: ${formatPct(item.data, 2)}`);
        });
        return lines.join("<br>");
      }
    },
    legend: {
      top: 0,
      textStyle: { color: "#425264" }
    },
    grid: { left: 56, right: 20, top: 46, bottom: 34 },
    xAxis: {
      type: "category",
      data: timelinePayload.periods,
      axisLabel: { color: "#425264", rotate: timelinePayload.periods.length > 18 ? 35 : 0 },
      axisLine: { lineStyle: { color: "#c8d2dc" } }
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#425264", formatter: (value) => `${value}%` },
      splitLine: { lineStyle: { color: "#dde4eb" } }
    },
    series: timelinePayload.line_series.map((series, index) => ({
      name: series.name,
      type: "line",
      smooth: true,
      symbolSize: 6,
      lineStyle: { width: 2.5, color: dashboardPayload.topic_palette[index % dashboardPayload.topic_palette.length] },
      itemStyle: { color: dashboardPayload.topic_palette[index % dashboardPayload.topic_palette.length] },
      data: series.data
    }))
  });

  const heatmapChart = registerChart(echarts.init(document.getElementById("timeline-heatmap-chart")));
  heatmapChart.setOption({
    tooltip: {
      position: "top",
      formatter: (params) => {
        const period = timelinePayload.periods[params.data[0]];
        const topic = timelinePayload.heatmap_topics[params.data[1]];
        const share = params.data[2];
        const docCount = params.data[3];
        return `<strong>${escapeHtml(topic)}</strong><br>${escapeHtml(period)}<br>占比: ${formatPct(share, 2)}<br>文档量: ${formatCount(docCount)}`;
      }
    },
    grid: { left: 180, right: 28, top: 20, bottom: 36 },
    xAxis: {
      type: "category",
      data: timelinePayload.periods,
      axisLabel: { color: "#425264", rotate: timelinePayload.periods.length > 18 ? 35 : 0 },
      splitArea: { show: true }
    },
    yAxis: {
      type: "category",
      data: timelinePayload.heatmap_topics,
      axisLabel: { color: "#425264", width: 160, overflow: "truncate" },
      splitArea: { show: true }
    },
    visualMap: {
      min: 0,
      max: Math.max(timelinePayload.heatmap_max || 0, 1),
      calculable: true,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      inRange: { color: ["#eff6ff", "#60a5fa", "#1d4ed8"] }
    },
    series: [{
      type: "heatmap",
      data: timelinePayload.heatmap_data,
      label: { show: false },
      emphasis: {
        itemStyle: {
          shadowBlur: 8,
          shadowColor: "rgba(15, 23, 42, 0.16)"
        }
      }
    }]
  });
}

function renderGeographyTab() {
  const geography = dashboardPayload.geography || {};
  if (!geography.has_data) {
    setEmptyState("geo-map-chart", "没有可映射的地区数据。");
    setEmptyState("geo-bar-chart", "没有可映射的地区数据。");
    document.getElementById("geo-scope-select").innerHTML = "";
    return;
  }

  const scopeSelect = document.getElementById("geo-scope-select");
  scopeSelect.innerHTML = (geography.scopes || []).map((scope) => `<option value="${escapeHtml(scope.id)}">${escapeHtml(scope.label)}</option>`).join("");

  const mapChart = registerChart(echarts.init(document.getElementById("geo-map-chart")));
  const barChart = registerChart(echarts.init(document.getElementById("geo-bar-chart")));

  function renderScope(scopeId) {
    const mapRows = geography.map_series[scopeId] || [];
    const barRows = geography.bar_series[scopeId] || [];
    if (!echarts.getMap("china")) {
      setEmptyState("geo-map-chart", "未加载中国地图脚本，无法绘制省份热力图。");
    } else {
      mapChart.clear();
      mapChart.setOption({
        tooltip: {
          trigger: "item",
          formatter: (params) => `${escapeHtml(params.name)}<br>文档量: ${formatCount(params.value || 0)}`
        },
        visualMap: {
          min: 0,
          max: Math.max(...mapRows.map((row) => Number(row.value || 0)), 1),
          left: 20,
          bottom: 10,
          text: ["高", "低"],
          calculable: true,
          inRange: { color: ["#eff6ff", "#60a5fa", "#1d4ed8"] }
        },
        series: [{
          name: "文档量",
          type: "map",
          map: "china",
          roam: false,
          label: { show: false },
          emphasis: { label: { show: true, color: "#102033" } },
          data: mapRows
        }]
      });
    }

    if (!barRows.length) {
      setEmptyState("geo-bar-chart", "这个范围下没有地区分布数据。");
      return;
    }
    barChart.clear();
    const reversed = [...barRows].reverse();
    barChart.setOption({
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: (params) => {
          const row = params[0].data.row;
          return `${escapeHtml(row.name)}<br>文档量: ${formatCount(row.value)}`;
        }
      },
      grid: { left: 90, right: 20, top: 20, bottom: 20 },
      xAxis: {
        type: "value",
        axisLabel: { color: "#425264" },
        splitLine: { lineStyle: { color: "#dde4eb" } }
      },
      yAxis: {
        type: "category",
        data: reversed.map((row) => row.name),
        axisLabel: { color: "#425264" },
        axisLine: { lineStyle: { color: "#c8d2dc" } }
      },
      series: [{
        type: "bar",
        barMaxWidth: 24,
        itemStyle: { color: "#0f766e" },
        data: reversed.map((row) => ({ value: row.value, row }))
      }]
    });
  }

  scopeSelect.addEventListener("change", () => renderScope(scopeSelect.value));
  renderScope((geography.scopes || [])[0].id);
}

function buildSparkline(values) {
  const data = Array.isArray(values) ? values.map((value) => Number(value || 0)) : [];
  const width = 180;
  const height = 36;
  if (!data.length) {
    return `<svg class="sparkline" viewBox="0 0 ${width} ${height}" aria-hidden="true"></svg>`;
  }
  const maxValue = Math.max(...data, 0.0001);
  const minValue = Math.min(...data, 0);
  const step = data.length > 1 ? width / (data.length - 1) : width;
  const scale = maxValue === minValue ? 1 : (height - 6) / (maxValue - minValue);
  const points = data.map((value, index) => {
    const x = index * step;
    const y = height - 3 - (value - minValue) * scale;
    return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");
  return `
    <svg class="sparkline" viewBox="0 0 ${width} ${height}" aria-hidden="true">
      <path d="${points}" fill="none" stroke="#2563eb" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"></path>
    </svg>
  `;
}

function topicCardHtml(topic) {
  const topTerms = (topic.top_terms_display || []).map((term) => `<span class="chip">${escapeHtml(term)}</span>`).join("");
  const keywordSegments = (topic.keyword_segments || [])
    .filter((segment) => Number(segment.share_pct || 0) > 0)
    .map((segment) => `
      <div class="keyword-segment" style="width:${Math.max(Number(segment.share_pct || 0), 0)}%;background:${segment.color};"></div>
    `).join("");
  const keywordLegend = (topic.keyword_segments || [])
    .filter((segment) => Number(segment.share_pct || 0) > 0)
    .map((segment) => `
      <span class="legend-item">
        <span class="legend-swatch" style="background:${segment.color};"></span>
        <span>${escapeHtml(segment.keyword)} ${formatPct(segment.share_pct, 1)}</span>
      </span>
    `).join("");
  const docs = (topic.representative_docs || []).map((doc) => `<li>${escapeHtml(doc)}</li>`).join("");
  return `
    <article class="topic-card">
      <div class="topic-card-header">
        <h3 class="topic-card-title">${escapeHtml(topic.topic_label_full)}</h3>
        <div class="topic-card-count">${formatCount(topic.topic_count)}</div>
      </div>
      <div class="topic-meta-row">
        <span>占已聚类文档 ${formatPct(topic.share_of_clustered_docs_pct, 2)}</span>
        <span>峰值 ${escapeHtml(topic.peak_period || "NA")}</span>
        <span>峰值期 ${formatPct(topic.peak_doc_share_pct, 2)}</span>
        <span>主导关键词 ${escapeHtml(topic.dominant_keyword)} ${formatPct(topic.dominant_keyword_share_pct, 1)}</span>
      </div>
      ${buildSparkline(topic.sparkline)}
      <div class="chip-row">${topTerms || '<span class="chip">无 top terms</span>'}</div>
      <div class="keyword-bar">${keywordSegments}</div>
      <div class="keyword-legend">${keywordLegend || '<span>无关键词构成数据</span>'}</div>
      ${docs ? `
        <details>
          <summary>Representative docs</summary>
          <ul class="doc-list">${docs}</ul>
        </details>
      ` : ""}
    </article>
  `;
}

function renderBrowserTab() {
  const sortSelect = document.getElementById("browser-sort-select");
  const limitSelect = document.getElementById("browser-limit-select");
  const searchInput = document.getElementById("browser-search-input");
  const metaNode = document.getElementById("browser-meta");
  const gridNode = document.getElementById("topic-browser-grid");
  const topics = Array.isArray(dashboardPayload.browser.topics) ? [...dashboardPayload.browser.topics] : [];

  function sortedTopics(rows) {
    const current = [...rows];
    switch (sortSelect.value) {
      case "peak_desc":
        current.sort((left, right) => Number(right.peak_doc_count || 0) - Number(left.peak_doc_count || 0) || Number(left.topic_id) - Number(right.topic_id));
        break;
      case "keyword_desc":
        current.sort((left, right) => Number(right.dominant_keyword_share_pct || 0) - Number(left.dominant_keyword_share_pct || 0) || Number(left.topic_id) - Number(right.topic_id));
        break;
      case "id_asc":
        current.sort((left, right) => Number(left.topic_id) - Number(right.topic_id));
        break;
      default:
        current.sort((left, right) => Number(right.topic_count || 0) - Number(left.topic_count || 0) || Number(left.topic_id) - Number(right.topic_id));
        break;
    }
    return current;
  }

  function renderCards() {
    const query = searchInput.value.trim().toLowerCase();
    let filtered = topics;
    if (query) {
      filtered = filtered.filter((topic) => String(topic.search_text || "").includes(query));
    }
    filtered = sortedTopics(filtered);

    const limitValue = limitSelect.value;
    const visible = limitValue === "all" ? filtered : filtered.slice(0, Number(limitValue));
    metaNode.textContent = `当前显示 ${visible.length} / ${filtered.length} 个主题${query ? "（已应用搜索）" : ""}。`;

    if (!visible.length) {
      gridNode.innerHTML = '<div class="empty-state">没有命中当前筛选条件的主题。</div>';
      return;
    }
    gridNode.innerHTML = visible.map((topic) => topicCardHtml(topic)).join("");
  }

  sortSelect.addEventListener("change", renderCards);
  limitSelect.addEventListener("change", renderCards);
  searchInput.addEventListener("input", renderCards);
  renderCards();
}

const renderers = {
  overview: renderOverviewTab,
  keyword: renderKeywordTab,
  timeline: renderTimelineTab,
  geography: renderGeographyTab,
  browser: renderBrowserTab
};

activateTab("overview");
  </script>
</body>
</html>
"""

    script_tags = "\n".join(f'  <script src="{html.escape(url)}"></script>' for url in script_urls)
    return (
        template.replace("__TITLE__", html.escape(title))
        .replace("__SUBTITLE__", html.escape(subtitle))
        .replace("__SOURCE_NOTE__", html.escape(source_note))
        .replace("__METRICS__", render_metric_cards(metrics))
        .replace("__FILTERS__", render_filter_pills(filters))
        .replace("__SCRIPT_TAGS__", script_tags)
        .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    )


def main() -> None:
    args = parse_args()
    emit = resolve_emit("topic-dashboard", None)
    total_start = time.perf_counter()

    config = resolve_input_config(args)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    emit("Loading topic model outputs")
    topic_info = pd.read_csv(config["topic_info_path"])
    topic_terms = pd.read_csv(config["topic_terms_path"])
    share_by_period = pd.read_csv(config["topic_share_by_period_path"])
    share_by_period_keyword = pd.read_csv(config["topic_share_by_period_keyword_path"])
    share_by_ip = pd.read_csv(config["topic_share_by_ip_path"]) if Path(config["topic_share_by_ip_path"]).exists() else pd.DataFrame()

    topic_term_lookup = build_topic_term_lookup(topic_terms)
    keywords, keyword_lookup, keyword_totals = build_keyword_count_lookup(share_by_period_keyword)
    keyword_colors = {
        keyword: KEYWORD_PALETTE[index % len(KEYWORD_PALETTE)]
        for index, keyword in enumerate(keywords)
    }
    peak_lookup = build_peak_lookup(share_by_period)
    noise_patterns = compile_noise_patterns(args.noise_pattern)

    all_topic_rows, clustered_total, outlier_count = build_topic_rows(
        topic_info,
        topic_term_lookup=topic_term_lookup,
        peak_lookup=peak_lookup,
        keyword_lookup=keyword_lookup,
        keywords=keywords,
        keyword_colors=keyword_colors,
        min_topic_size=args.min_topic_size,
        noise_patterns=noise_patterns,
        top_n_terms=args.top_n_terms,
    )
    if not all_topic_rows:
        raise ValueError("No non-outlier topics found in topic_info.csv.")

    all_topic_ids = [int(row["topic_id"]) for row in all_topic_rows]
    periods, period_totals, share_lookup, count_lookup = build_period_lookup(
        share_by_period,
        topic_ids=all_topic_ids,
        min_period_docs=args.min_period_docs,
    )
    attach_period_series(
        all_topic_rows,
        periods=periods,
        share_lookup=share_lookup,
        count_lookup=count_lookup,
    )

    display_rows = [row for row in all_topic_rows if not row["exclusion_reason"]]
    if not display_rows:
        raise ValueError("Display filters removed every topic. Lower --min_topic_size or relax --noise_pattern.")

    selected_rows = display_rows[: args.top_n_topics]
    total_docs = clustered_total + outlier_count
    display_doc_count = sum(int(row["topic_count"]) for row in display_rows)
    selected_doc_count = sum(int(row["topic_count"]) for row in selected_rows)
    excluded_topic_count = len(all_topic_rows) - len(display_rows)
    excluded_doc_count = clustered_total - display_doc_count

    emit(
        "Prepared dashboard payload "
        f"(display_topics={len(display_rows)}, highlighted={len(selected_rows)}, periods={len(periods)})"
    )

    metrics = [
        {
            "label": "总文档量",
            "value": f"{total_docs:,}",
            "detail": "topic_info.csv 里的全部文档，包含 Topic -1。",
        },
        {
            "label": "已聚类文档",
            "value": f"{clustered_total:,}",
            "detail": f"{(100.0 * clustered_total / total_docs):.2f}% 进入具体主题。" if total_docs else "暂无数据。",
        },
        {
            "label": "Topic -1",
            "value": f"{(100.0 * outlier_count / total_docs):.2f}%",
            "detail": f"{outlier_count:,} 篇文档落到离群主题。" if total_docs else "暂无数据。",
        },
        {
            "label": "展示主题",
            "value": f"{len(display_rows)} / {len(all_topic_rows)}",
            "detail": f"展示过滤覆盖 {display_doc_count:,} 篇文档。",
        },
        {
            "label": f"Top {len(selected_rows)} 覆盖率",
            "value": f"{(100.0 * selected_doc_count / clustered_total):.2f}%",
            "detail": "头部主题在已聚类文档中的覆盖率。",
        },
    ]

    filter_pills = [
        f"from_summary={Path(config['summary_path']).name}" if config["summary_path"] else "from_summary=off",
        f"min_topic_size>={args.min_topic_size}" if args.min_topic_size > 0 else "min_topic_size=off",
        f"noise_pattern x{len(args.noise_pattern)}" if args.noise_pattern else "noise_pattern=off",
        f"min_period_docs>={args.min_period_docs}",
    ]

    dashboard_payload = {
        "keyword_colors": [keyword_colors[keyword] for keyword in keywords] or KEYWORD_PALETTE[:3],
        "topic_palette": ["#2563eb", "#0f766e", "#dc2626", "#d97706", "#7c3aed", "#0891b2", "#ea580c", "#65a30d"],
        "overview": {
            "topic_bars": build_overview_topic_bars(display_rows, top_n_topics=args.top_n_topics),
            "keyword_pie": build_keyword_pie(keyword_totals=keyword_totals, clustered_total=clustered_total),
            "period_totals": [{"period": period, "doc_count": int(period_totals[period])} for period in periods],
        },
        "keyword_profile": build_keyword_profile_payload(
            share_by_period_keyword,
            selected_rows=selected_rows,
        ),
        "timeline": build_evolution_payload(periods=periods, selected_rows=selected_rows),
        "geography": build_geography_payload(share_by_ip, selected_rows=selected_rows),
        "browser": {
            "topics": display_rows,
        },
    }

    display_table = build_display_table(
        display_rows,
        filter_applied=bool(args.min_topic_size > 0 or args.noise_pattern),
    )

    summary_path: Path | None = config["summary_path"]
    source_note = (
        f"来源 summary: {summary_path}"
        if summary_path
        else f"来源目录: {config['topic_model_dir']}"
    )
    subtitle = (
        "单页看板直接对齐 08 的 BERTopic 产物：总览、关键词画像、时间演化、地域分布、主题浏览。"
        " 展示过滤只影响 11 的输出，不会改动 08 的底表。"
    )

    dashboard_html = render_dashboard_html(
        title="BERTopic 主题看板",
        subtitle=subtitle,
        source_note=source_note,
        metrics=metrics,
        filters=filter_pills,
        payload=dashboard_payload,
        script_urls=[args.echarts_js_url, args.echarts_china_map_js_url],
    )

    dashboard_path = output_dir / "topic_dashboard.html"
    display_table_path = output_dir / "topic_display_table.csv"
    summary_output_path = output_dir / "topic_visualization_summary.json"

    emit("Saving topic dashboard outputs")
    dashboard_path.write_text(dashboard_html, encoding="utf-8")
    display_table.to_csv(display_table_path, index=False, encoding="utf-8-sig")
    save_json(
        summary_output_path,
        {
            "from_summary_path": str(summary_path) if summary_path else "",
            "topic_info_path": str(Path(config["topic_info_path"]).resolve()),
            "topic_terms_path": str(Path(config["topic_terms_path"]).resolve()),
            "topic_share_by_period_path": str(Path(config["topic_share_by_period_path"]).resolve()),
            "topic_share_by_period_and_keyword_path": str(Path(config["topic_share_by_period_keyword_path"]).resolve()),
            "topic_share_by_ip_path": str(Path(config["topic_share_by_ip_path"]).resolve()) if Path(config["topic_share_by_ip_path"]).exists() else "",
            "output_dir": str(output_dir.resolve()),
            "dashboard_path": str(dashboard_path.resolve()),
            "display_table_path": str(display_table_path.resolve()),
            "total_docs": int(total_docs),
            "clustered_doc_count": int(clustered_total),
            "outlier_count": int(outlier_count),
            "outlier_share": float(outlier_count / total_docs) if total_docs else 0.0,
            "display_topic_count": int(len(display_rows)),
            "display_doc_count": int(display_doc_count),
            "display_doc_share": float(display_doc_count / clustered_total) if clustered_total else 0.0,
            "excluded_topic_count": int(excluded_topic_count),
            "excluded_doc_count": int(excluded_doc_count),
            "top_n_topics": int(args.top_n_topics),
            "top_n_terms": int(args.top_n_terms),
            "min_period_docs": int(args.min_period_docs),
            "min_topic_size": int(args.min_topic_size),
            "noise_patterns": args.noise_pattern,
            "retained_periods": periods,
            "keywords": keywords,
            "echarts_js_url": args.echarts_js_url,
            "echarts_china_map_js_url": args.echarts_china_map_js_url,
        },
    )
    emit(f"Saved topic dashboard to {dashboard_path}")
    emit(f"Total runtime: {format_elapsed(total_start)}")


if __name__ == "__main__":
    main()
