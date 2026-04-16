#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path

import pandas as pd

from lib.analysis_utils import resolve_emit, save_dataframe, sort_period_labels
from lib.io_utils import save_json

ECHARTS_JS_URL = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"
CHINA_MAP_JS_URL = "https://cdn.jsdelivr.net/npm/echarts@5/map/js/china.js"
DEFAULT_TOPIC_MODEL_DIR = Path("bert/artifacts/broad_analysis/topic_model")
CHINA_PROVINCE_NAMES = [
    "北京",
    "天津",
    "上海",
    "重庆",
    "河北",
    "山西",
    "内蒙古",
    "辽宁",
    "吉林",
    "黑龙江",
    "江苏",
    "浙江",
    "安徽",
    "福建",
    "江西",
    "山东",
    "河南",
    "湖北",
    "湖南",
    "广东",
    "广西",
    "海南",
    "四川",
    "贵州",
    "云南",
    "西藏",
    "陕西",
    "甘肃",
    "青海",
    "宁夏",
    "新疆",
    "香港",
    "澳门",
    "台湾",
]
PROVINCE_NAME_ALIASES = {
    "北京市": "北京",
    "天津市": "天津",
    "上海市": "上海",
    "重庆市": "重庆",
    "内蒙古自治区": "内蒙古",
    "广西壮族自治区": "广西",
    "西藏自治区": "西藏",
    "宁夏回族自治区": "宁夏",
    "新疆维吾尔自治区": "新疆",
    "香港特别行政区": "香港",
    "澳门特别行政区": "澳门",
    "中国香港": "香港",
    "中国澳门": "澳门",
    "中国台湾": "台湾",
}


def format_elapsed(start_time: float) -> str:
    return f"{time.perf_counter() - start_time:.2f}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HTML visualizations from BERTopic topic-share outputs."
    )
    parser.add_argument(
        "--topic_info_path",
        default=str(DEFAULT_TOPIC_MODEL_DIR / "topic_info.csv"),
        help="Path to topic_info.csv.",
    )
    parser.add_argument(
        "--topic_terms_path",
        default=str(DEFAULT_TOPIC_MODEL_DIR / "topic_terms.csv"),
        help="Path to topic_terms.csv.",
    )
    parser.add_argument(
        "--topic_share_by_period_path",
        default=str(DEFAULT_TOPIC_MODEL_DIR / "topic_share_by_period.csv"),
        help="Path to topic_share_by_period.csv.",
    )
    parser.add_argument(
        "--topic_share_by_period_keyword_path",
        default=str(DEFAULT_TOPIC_MODEL_DIR / "topic_share_by_period_and_keyword.csv"),
        help="Path to topic_share_by_period_and_keyword.csv.",
    )
    parser.add_argument(
        "--topic_share_by_ip_path",
        default=str(DEFAULT_TOPIC_MODEL_DIR / "topic_share_by_ip.csv"),
        help="Path to topic_share_by_ip.csv.",
    )
    parser.add_argument(
        "--topic_share_by_period_ip_path",
        default=str(DEFAULT_TOPIC_MODEL_DIR / "topic_share_by_period_and_ip.csv"),
        help="Path to topic_share_by_period_and_ip.csv.",
    )
    parser.add_argument(
        "--output_dir",
        default="bert/artifacts/broad_analysis/topic_visuals",
        help="Directory for generated HTML and helper CSV outputs.",
    )
    parser.add_argument(
        "--top_n_topics",
        type=int,
        default=8,
        help="Number of high-volume topics to keep in the profile and line charts.",
    )
    parser.add_argument(
        "--map_top_n_topics",
        type=int,
        default=8,
        help="Number of high-volume topics to expose in the province heatmap.",
    )
    parser.add_argument(
        "--map_min_total_docs",
        type=int,
        default=30,
        help="Minimum documents required within a province-period cell before the map applies color.",
    )
    parser.add_argument(
        "--echarts_js_url",
        default=ECHARTS_JS_URL,
        help="URL for echarts.min.js.",
    )
    parser.add_argument(
        "--china_map_js_url",
        default=CHINA_MAP_JS_URL,
        help="URL for the ECharts China map bundle.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def shorten_label(text: str, *, max_length: int = 28) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def infer_period_granularity(labels: list[str]) -> str:
    usable = [label for label in labels if label and label not in {"全部时间", "NA"}]
    if not usable:
        return "month"
    if any("Q" in label for label in usable):
        return "quarter"
    if all(re.fullmatch(r"\d{4}", label) for label in usable):
        return "year"
    return "month"


def try_parse_list(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    if not (text.startswith("[") and text.endswith("]")):
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    terms = []
    for item in parsed:
        candidate = clean_text(item)
        if candidate:
            terms.append(candidate)
    return terms


def build_topic_label_map(topic_info: pd.DataFrame, topic_terms: pd.DataFrame) -> dict[int, str]:
    terms_by_topic: dict[int, list[str]] = {}
    if not topic_terms.empty and {"topic_id", "term_rank", "term"}.issubset(topic_terms.columns):
        working = topic_terms.copy()
        working["topic_id"] = pd.to_numeric(working["topic_id"], errors="coerce").astype("Int64")
        working["term_rank"] = pd.to_numeric(working["term_rank"], errors="coerce")
        working = working.dropna(subset=["topic_id", "term_rank", "term"])
        working = working.sort_values(["topic_id", "term_rank"])
        for topic_id, frame in working.groupby("topic_id"):
            top_terms = [clean_text(term) for term in frame["term"].tolist() if clean_text(term)]
            terms_by_topic[int(topic_id)] = top_terms[:3]

    label_map: dict[int, str] = {}
    for _, row in topic_info.iterrows():
        topic_id = pd.to_numeric(row.get("Topic"), errors="coerce")
        if pd.isna(topic_id):
            continue
        topic_id_int = int(topic_id)
        if topic_id_int < 0:
            continue

        label_zh = clean_text(row.get("topic_label_zh"))
        label_machine = clean_text(row.get("topic_label_machine"))
        if not label_machine:
            label_machine = clean_text(row.get("Name"))

        if label_zh:
            base_label = label_zh
        elif topic_id_int in terms_by_topic and terms_by_topic[topic_id_int]:
            base_label = " / ".join(terms_by_topic[topic_id_int])
        else:
            parsed_terms = try_parse_list(row.get("Representation"))
            if parsed_terms:
                base_label = " / ".join(parsed_terms[:3])
            else:
                base_label = label_machine or f"Topic {topic_id_int}"

        label_map[topic_id_int] = f"T{topic_id_int} {shorten_label(base_label)}"
    return label_map


def choose_top_topics(df: pd.DataFrame, *, top_n: int) -> list[int]:
    working = df.copy()
    working["topic_id"] = pd.to_numeric(working["topic_id"], errors="coerce")
    working["doc_count"] = pd.to_numeric(working["doc_count"], errors="coerce")
    working = working.dropna(subset=["topic_id", "doc_count"])
    working = working[working["topic_id"] >= 0]
    totals = (
        working.groupby("topic_id", as_index=False)["doc_count"]
        .sum()
        .sort_values(["doc_count", "topic_id"], ascending=[False, True])
    )
    return totals["topic_id"].astype(int).head(top_n).tolist()


def build_profile_period_frame(df: pd.DataFrame, *, period_label: str | None) -> pd.DataFrame:
    working = df.copy()
    if period_label is not None:
        working = working[working["period_label"].astype(str) == period_label].copy()

    grouped = (
        working.groupby(["keyword_normalized", "topic_id"], as_index=False)["doc_count"]
        .sum()
        .sort_values(["keyword_normalized", "doc_count", "topic_id"], ascending=[True, False, True])
    )
    if grouped.empty:
        return grouped.assign(doc_share=pd.Series(dtype=float))

    grouped["doc_share"] = grouped["doc_count"] / grouped.groupby("keyword_normalized")["doc_count"].transform("sum")
    return grouped


def build_profile_payload(
    df: pd.DataFrame,
    *,
    topic_label_map: dict[int, str],
    top_topics: list[int],
) -> tuple[dict[str, dict[str, object]], pd.DataFrame, pd.DataFrame]:
    keywords = sorted(df["keyword_normalized"].dropna().astype(str).unique().tolist())
    periods = df["period_label"].dropna().astype(str).tolist()
    period_granularity = infer_period_granularity(periods)
    periods = sort_period_labels(periods, period_granularity)
    payload: dict[str, dict[str, object]] = {}

    def _build_payload_for_frame(frame: pd.DataFrame) -> dict[str, object]:
        shares_by_keyword_topic: dict[str, dict[int, float]] = {keyword: {} for keyword in keywords}
        for row in frame.itertuples(index=False):
            keyword = str(row.keyword_normalized)
            topic_id = int(row.topic_id)
            shares_by_keyword_topic.setdefault(keyword, {})[topic_id] = float(row.doc_share)

        stacked_series = []
        for topic_id in top_topics:
            stacked_series.append(
                {
                    "name": topic_label_map.get(topic_id, f"T{topic_id}"),
                    "type": "bar",
                    "stack": "topic_share",
                    "emphasis": {"focus": "series"},
                    "data": [round(100.0 * shares_by_keyword_topic.get(keyword, {}).get(topic_id, 0.0), 4) for keyword in keywords],
                }
            )

        other_values = []
        for keyword in keywords:
            top_share = sum(shares_by_keyword_topic.get(keyword, {}).get(topic_id, 0.0) for topic_id in top_topics)
            other_values.append(round(100.0 * max(0.0, 1.0 - top_share), 4))
        stacked_series.append(
            {
                "name": "其他主题",
                "type": "bar",
                "stack": "topic_share",
                "emphasis": {"focus": "series"},
                "data": other_values,
            }
        )

        radar_indicators = [{"name": topic_label_map.get(topic_id, f"T{topic_id}"), "max": 100} for topic_id in top_topics]
        radar_series = []
        for keyword in keywords:
            radar_series.append(
                {
                    "name": keyword,
                    "value": [round(100.0 * shares_by_keyword_topic.get(keyword, {}).get(topic_id, 0.0), 4) for topic_id in top_topics],
                }
            )

        matrix_rows = []
        for keyword in keywords:
            row: dict[str, object] = {"keyword": keyword}
            for topic_id in top_topics:
                row[topic_label_map.get(topic_id, f"T{topic_id}")] = round(
                    100.0 * shares_by_keyword_topic.get(keyword, {}).get(topic_id, 0.0),
                    4,
                )
            row["其他主题"] = round(
                100.0 * max(
                    0.0,
                    1.0 - sum(shares_by_keyword_topic.get(keyword, {}).get(topic_id, 0.0) for topic_id in top_topics),
                ),
                4,
            )
            matrix_rows.append(row)

        return {
            "keywords": keywords,
            "stacked_series": stacked_series,
            "radar_indicators": radar_indicators,
            "radar_series": radar_series,
            "matrix_rows": matrix_rows,
        }

    overall_frame = build_profile_period_frame(df, period_label=None)
    payload["全部时间"] = _build_payload_for_frame(overall_frame)
    overall_matrix = pd.DataFrame(payload["全部时间"]["matrix_rows"])

    latest_period = periods[-1] if periods else None
    latest_matrix = overall_matrix.copy()
    for period_label in periods:
        frame = build_profile_period_frame(df, period_label=period_label)
        payload[period_label] = _build_payload_for_frame(frame)
        if latest_period == period_label:
            latest_matrix = pd.DataFrame(payload[period_label]["matrix_rows"])

    return payload, overall_matrix, latest_matrix


def build_time_payload(
    df: pd.DataFrame,
    *,
    topic_label_map: dict[int, str],
    top_topics: list[int],
) -> dict[str, object]:
    working = df.copy()
    working["period_label"] = working["period_label"].astype(str)
    periods = sort_period_labels(
        working["period_label"].dropna().astype(str).tolist(),
        infer_period_granularity(working["period_label"].dropna().astype(str).tolist()),
    )
    series = []
    for topic_id in top_topics:
        topic_frame = working[working["topic_id"] == topic_id].copy()
        share_by_period = {
            str(row.period_label): float(row.doc_share) * 100.0 for row in topic_frame.itertuples(index=False)
        }
        series.append(
            {
                "name": topic_label_map.get(topic_id, f"T{topic_id}"),
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "data": [round(share_by_period.get(period_label, 0.0), 4) for period_label in periods],
            }
        )
    return {"periods": periods, "series": series}


def normalize_province_name(value: object) -> str | None:
    text = clean_text(value)
    if not text or text == "UNKNOWN_IP":
        return None
    text = PROVINCE_NAME_ALIASES.get(text, text)
    if text in CHINA_PROVINCE_NAMES:
        return text
    return None


def build_map_payload(
    share_by_ip: pd.DataFrame,
    share_by_period_ip: pd.DataFrame,
    *,
    topic_label_map: dict[int, str],
    top_topics: list[int],
) -> tuple[pd.DataFrame, dict[str, dict[str, list[dict[str, object]]]], list[str], list[int], list[str]]:
    def _normalize_rows(frame: pd.DataFrame, *, period_label_col: str | None, default_period_label: str | None) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(
                columns=["period_label", "topic_id", "province", "doc_count", "period_total", "doc_share_pct"]
            )

        working = frame.copy()
        working["province"] = working["ip_normalized"].map(normalize_province_name)
        working = working[working["province"].notna()].copy()
        if working.empty:
            return pd.DataFrame(
                columns=["period_label", "topic_id", "province", "doc_count", "period_total", "doc_share_pct"]
            )

        working["topic_id"] = pd.to_numeric(working["topic_id"], errors="coerce").astype("Int64")
        working["doc_count"] = pd.to_numeric(working["doc_count"], errors="coerce")
        working["doc_share"] = pd.to_numeric(working["doc_share"], errors="coerce")
        working = working.dropna(subset=["topic_id", "doc_count", "doc_share"]).copy()
        working = working[working["topic_id"].isin(top_topics)].copy()
        if working.empty:
            return pd.DataFrame(
                columns=["period_label", "topic_id", "province", "doc_count", "period_total", "doc_share_pct"]
            )

        if period_label_col is not None:
            working["period_label"] = working[period_label_col].astype(str)
        else:
            working["period_label"] = default_period_label

        totals = (
            working.groupby(["period_label", "province"], as_index=False)["doc_count"]
            .sum()
            .rename(columns={"doc_count": "period_total"})
        )
        working = working.merge(totals, on=["period_label", "province"], how="left")
        working["topic_id"] = working["topic_id"].astype(int)
        working["period_total"] = pd.to_numeric(working["period_total"], errors="coerce").fillna(0).astype(int)
        working["doc_count"] = pd.to_numeric(working["doc_count"], errors="coerce").fillna(0).astype(int)
        working["doc_share_pct"] = pd.to_numeric(working["doc_share"], errors="coerce").fillna(0.0) * 100.0
        return working[["period_label", "topic_id", "province", "doc_count", "period_total", "doc_share_pct"]]

    overall_df = _normalize_rows(share_by_ip, period_label_col=None, default_period_label="全部时间")
    period_df = _normalize_rows(share_by_period_ip, period_label_col="period_label", default_period_label=None)
    region_df = pd.concat([overall_df, period_df], ignore_index=True)
    periods = ["全部时间"]
    if not share_by_period_ip.empty:
        raw_periods = share_by_period_ip["period_label"].dropna().astype(str).tolist()
        periods.extend(sort_period_labels(raw_periods, infer_period_granularity(raw_periods)))

    payload: dict[str, dict[str, list[dict[str, object]]]] = {}
    included_regions = sorted(region_df["province"].drop_duplicates().tolist()) if not region_df.empty else []

    for period_label in periods:
        period_payload: dict[str, list[dict[str, object]]] = {}
        current = region_df[region_df["period_label"] == period_label].copy() if not region_df.empty else pd.DataFrame()
        for topic_id in top_topics:
            values: dict[str, dict[str, object]] = {
                province: {
                    "name": province,
                    "value": None,
                    "share_pct": 0.0,
                    "doc_count": 0,
                    "period_total": 0,
                }
                for province in CHINA_PROVINCE_NAMES
            }
            if not current.empty:
                topic_frame = current[current["topic_id"] == topic_id]
                for topic_row in topic_frame.itertuples(index=False):
                    values[str(topic_row.province)] = {
                        "name": str(topic_row.province),
                        "value": round(float(topic_row.doc_share_pct), 4),
                        "share_pct": round(float(topic_row.doc_share_pct), 4),
                        "doc_count": int(topic_row.doc_count),
                        "period_total": int(topic_row.period_total),
                    }
            period_payload[str(topic_id)] = [values[province] for province in CHINA_PROVINCE_NAMES]
        payload[period_label] = period_payload

    return region_df, payload, periods, top_topics, included_regions


def render_summary_html(items: list[dict[str, str]]) -> str:
    blocks = []
    for item in items:
        blocks.append(
            f"""      <div class="metric">
        <div class="metric-label">{item["label"]}</div>
        <div class="metric-value">{item["value"]}</div>
        <div class="metric-detail">{item["detail"]}</div>
      </div>"""
        )
    return "\n".join(blocks)


def render_resource_links(links: list[dict[str, str]]) -> str:
    if not links:
        return ""
    tags = "\n".join(
        f'      <a class="quick-link" href="{link["href"]}" target="_blank" rel="noreferrer">{link["label"]}</a>'
        for link in links
    )
    return f"""    <div class="quick-links">
{tags}
    </div>"""


def html_shell(
    *,
    title: str,
    subtitle: str,
    summary_html: str,
    controls_heading: str,
    controls_note: str,
    controls_html: str,
    panels_html: str,
    script_urls: list[str],
    inline_script: str,
    resource_links_html: str = "",
) -> str:
    script_tags = "\n".join(f'  <script src="{url}"></script>' for url in script_urls)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #eef3ee;
      color: #17212b;
    }}
    .page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 28px 40px;
    }}
    h1 {{
      margin: 0;
      font-size: 34px;
      line-height: 1.1;
    }}
    h2 {{
      margin: 0;
      font-size: 21px;
      line-height: 1.2;
    }}
    .hero {{
      display: grid;
      gap: 20px;
      padding: 8px 0 24px;
      border-bottom: 1px solid #d7e1d4;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #0f766e;
    }}
    .lead {{
      margin: 12px 0 0;
      max-width: 760px;
      color: #425466;
      font-size: 16px;
      line-height: 1.55;
    }}
    .summary-strip {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .metric {{
      min-height: 78px;
      background: #fbfcf8;
      border: 1px solid #d7e1d4;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric-label {{
      color: #57706b;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .metric-value {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
      color: #14213d;
    }}
    .metric-detail {{
      margin-top: 6px;
      font-size: 13px;
      line-height: 1.45;
      color: #556474;
    }}
    .toolbar-band {{
      display: grid;
      gap: 16px;
      padding: 18px 0 22px;
      border-bottom: 1px solid #d7e1d4;
    }}
    .toolbar-meta {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      align-items: end;
    }}
    .toolbar-heading {{
      display: grid;
      gap: 4px;
    }}
    .toolbar-kicker {{
      margin: 0;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #b45309;
    }}
    .toolbar-note {{
      margin: 0;
      color: #556474;
      font-size: 14px;
      line-height: 1.45;
      max-width: 760px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: end;
    }}
    .control-group {{
      display: grid;
      gap: 6px;
      min-width: 220px;
    }}
    label {{
      font-size: 13px;
      font-weight: 600;
      color: #405261;
    }}
    select {{
      min-width: 180px;
      height: 42px;
      border: 1px solid #bfd0c4;
      border-radius: 8px;
      background: #fff;
      padding: 0 12px;
      font-size: 14px;
      color: #17212b;
      box-shadow: 0 1px 0 rgba(16, 24, 40, 0.03);
    }}
    .quick-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .quick-link {{
      display: inline-flex;
      align-items: center;
      min-height: 36px;
      padding: 0 12px;
      border-radius: 999px;
      border: 1px solid #cfd9ef;
      background: #ffffff;
      color: #1d4ed8;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
    }}
    .grid {{
      margin-top: 24px;
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    .panel {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: #fbfcf8;
      border: 1px solid #d7e1d4;
      border-radius: 8px;
      padding: 18px 18px 12px;
      box-shadow: 0 10px 24px rgba(17, 24, 39, 0.04);
    }}
    .panel-head {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px 18px;
      align-items: end;
    }}
    .panel-eyebrow {{
      margin: 0 0 6px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #0f766e;
    }}
    .panel-note {{
      margin: 0;
      max-width: 540px;
      color: #556474;
      font-size: 14px;
      line-height: 1.45;
    }}
    .chart {{
      width: 100%;
      height: 520px;
    }}
    .chart.tall {{
      height: 640px;
    }}
    @media (max-width: 900px) {{
      .page {{
        padding: 18px 16px 28px;
      }}
      h1 {{
        font-size: 30px;
      }}
      .chart {{
        height: 420px;
      }}
      .chart.tall {{
        height: 520px;
      }}
      .metric-value {{
        font-size: 24px;
      }}
      .control-group {{
        min-width: min(100%, 260px);
      }}
    }}
  </style>
{script_tags}
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <p class="eyebrow">BERTopic Output Review</p>
        <h1>{title}</h1>
        <p class="lead">{subtitle}</p>
      </div>
      <div class="summary-strip">
{summary_html}
      </div>
    </section>
    <section class="toolbar-band">
      <div class="toolbar-meta">
        <div class="toolbar-heading">
          <p class="toolbar-kicker">{controls_heading}</p>
          <p class="toolbar-note">{controls_note}</p>
        </div>
{resource_links_html}
      </div>
      <div class="toolbar">
{controls_html}
      </div>
    </section>
    <div class="grid">
{panels_html}
    </div>
  </div>
  <script>
{inline_script}
  </script>
</body>
</html>
"""


def render_profile_html(profile_payload: dict[str, dict[str, object]], *, echarts_js_url: str) -> str:
    profile_json = json.dumps(profile_payload, ensure_ascii=False)
    periods = list(profile_payload.keys())
    analysis_periods = [period for period in periods if period != "全部时间"]
    overall_payload = profile_payload.get("全部时间", {})
    keywords = overall_payload.get("keywords", [])
    top_topic_count = max(0, len(overall_payload.get("radar_indicators", [])))
    options_html = f"""        <div class="control-group">
          <label for="profile-period">比较时间段</label>
          <select id="profile-period">
{chr(10).join(f'            <option value="{period}">{period}</option>' for period in periods)}
          </select>
        </div>"""
    summary_html = render_summary_html(
        [
            {
                "label": "主题窗口",
                "value": f"Top {top_topic_count}",
                "detail": "默认聚焦高频主题，剩余主题合并为“其他主题”。",
            },
            {
                "label": "时间切面",
                "value": f"{len(analysis_periods)} 期",
                "detail": "包含一个“全部时间”总览，方便对比阶段变化。",
            },
            {
                "label": "关键词数",
                "value": str(len(keywords)),
                "detail": "当前画像面板会同时对比这些关键词的主题组成。",
            },
        ]
    )
    resource_links_html = render_resource_links(
        [
            {"href": "topic_keyword_profile_overall.csv", "label": "下载总体画像 CSV"},
            {"href": "topic_keyword_profile_latest.csv", "label": "下载最新画像 CSV"},
        ]
    )
    inline_script = f"""
const profileData = {profile_json};
const periodSelect = document.getElementById("profile-period");
const stackedChart = echarts.init(document.getElementById("stacked-chart"));
const radarChart = echarts.init(document.getElementById("radar-chart"));
const profilePalette = ["#0f766e", "#d97706", "#2563eb", "#dc2626", "#65a30d", "#0891b2", "#b45309", "#be123c", "#94a3b8"];

function renderProfile() {{
  const current = profileData[periodSelect.value];
  stackedChart.setOption({{
    color: profilePalette,
    animationDuration: 400,
    tooltip: {{ trigger: "axis", axisPointer: {{ type: "shadow" }}, valueFormatter: (value) => value.toFixed(2) + "%" }},
    legend: {{ top: 0, type: "scroll" }},
    grid: {{ top: 84, left: 62, right: 30, bottom: 76 }},
    xAxis: {{
      type: "category",
      data: current.keywords,
      axisLabel: {{ interval: 0, color: "#415365" }},
      axisLine: {{ lineStyle: {{ color: "#9fb0aa" }} }}
    }},
    yAxis: {{
      type: "value",
      name: "占比 (%)",
      max: 100,
      nameTextStyle: {{ color: "#415365" }},
      axisLabel: {{ color: "#415365" }},
      splitLine: {{ lineStyle: {{ color: "#d7e1d4" }} }}
    }},
    series: current.stacked_series.map((series) => ({{
      ...series,
      barMaxWidth: 44,
      itemStyle: {{ borderRadius: [4, 4, 0, 0] }}
    }}))
  }});

  radarChart.setOption({{
    color: profilePalette,
    animationDuration: 400,
    tooltip: {{ trigger: "item" }},
    legend: {{ top: 0, type: "scroll" }},
    radar: {{
      radius: "68%",
      indicator: current.radar_indicators,
      splitArea: {{ areaStyle: {{ color: ["rgba(15,118,110,0.03)", "rgba(217,119,6,0.02)"] }} }},
      splitLine: {{ lineStyle: {{ color: "#c8d7cf" }} }},
      axisLine: {{ lineStyle: {{ color: "#b3c3bc" }} }},
      name: {{ color: "#415365" }}
    }},
    series: [{{
      type: "radar",
      data: current.radar_series.map((item) => ({{
        ...item,
        areaStyle: {{ opacity: 0.12 }}
      }}))
    }}]
  }});
}}

periodSelect.addEventListener("change", renderProfile);
window.addEventListener("resize", () => {{
  stackedChart.resize();
  radarChart.resize();
}});
renderProfile();
	"""
    controls_html = options_html
    panels_html = """      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="panel-eyebrow">关键词切面</p>
            <h2>主题占比堆叠</h2>
          </div>
          <p class="panel-note">横向比较同一时间切片下，不同关键词分别被哪些高频主题占据。</p>
        </div>
        <div id="stacked-chart" class="chart"></div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="panel-eyebrow">关键词轮廓</p>
            <h2>主题雷达对照</h2>
          </div>
          <p class="panel-note">把同一批高频主题映射到统一坐标上，快速看出各关键词偏向哪些主题。</p>
        </div>
        <div id="radar-chart" class="chart"></div>
      </section>"""
    return html_shell(
        title="主题 x 关键词画像",
        subtitle="把 `08` 产出的 topic share 表转成适合汇报的关键词画像视图，方便横向对照“佛系 / 摆烂 / 躺平”在不同时间片里的主题组成。",
        summary_html=summary_html,
        controls_heading="筛选视角",
        controls_note="先挑时间切片，再看三类关键词在高频主题上的分布差异。自动补出的 topic 标签会优先使用人工中文标签，没有的话再退回到 topic term。",
        controls_html=controls_html,
        panels_html=panels_html,
        script_urls=[echarts_js_url],
        inline_script=inline_script,
        resource_links_html=resource_links_html,
    )


def render_time_html(time_payload: dict[str, object], *, echarts_js_url: str) -> str:
    time_json = json.dumps(time_payload, ensure_ascii=False)
    periods = [str(period) for period in time_payload.get("periods", [])]
    series = time_payload.get("series", [])
    summary_html = render_summary_html(
        [
            {
                "label": "时间跨度",
                "value": f"{len(periods)} 期",
                "detail": f"{periods[0]} 至 {periods[-1]}" if periods else "暂无可用时间切片。",
            },
            {
                "label": "趋势线数",
                "value": str(len(series)),
                "detail": "默认展示高频主题，方便先看大盘，再回头补细分主题。",
            },
            {
                "label": "阅读方式",
                "value": "拖拽 / 缩放",
                "detail": "底部滑块可以压缩时间窗，适合长时间序列浏览。",
            },
        ]
    )
    inline_script = f"""
const timeData = {time_json};
const chart = echarts.init(document.getElementById("time-chart"));
const timePalette = ["#0f766e", "#d97706", "#2563eb", "#dc2626", "#65a30d", "#0891b2", "#b45309", "#be123c"];
chart.setOption({{
  color: timePalette,
  animationDuration: 500,
  tooltip: {{ trigger: "axis", valueFormatter: (value) => value.toFixed(2) + "%" }},
  legend: {{ top: 0, type: "scroll" }},
  grid: {{ top: 86, left: 62, right: 30, bottom: 118 }},
  xAxis: {{
    type: "category",
    data: timeData.periods,
    axisLabel: {{ interval: 0, rotate: 35, color: "#415365" }},
    axisLine: {{ lineStyle: {{ color: "#9fb0aa" }} }}
  }},
  yAxis: {{
    type: "value",
    name: "占比 (%)",
    nameTextStyle: {{ color: "#415365" }},
    axisLabel: {{ color: "#415365" }},
    splitLine: {{ lineStyle: {{ color: "#d7e1d4" }} }}
  }},
  dataZoom: [
    {{ type: "inside", start: 0, end: 100 }},
    {{ type: "slider", bottom: 32, height: 26, borderColor: "#d7e1d4", brushSelect: false }}
  ],
  series: timeData.series.map((item) => ({{
    ...item,
    lineStyle: {{ width: 3 }},
    emphasis: {{ focus: "series" }}
  }}))
}});
window.addEventListener("resize", () => chart.resize());
"""
    panels_html = """      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="panel-eyebrow">时间走势</p>
            <h2>主题占比演化</h2>
          </div>
          <p class="panel-note">把高频主题拉到同一条时间轴上，方便快速看出扩张、回落和阶段性交替。</p>
        </div>
        <div id="time-chart" class="chart tall"></div>
      </section>"""
    return html_shell(
        title="主题时间演化",
        subtitle="按时间顺序追踪高频主题的份额变化，适合抓住主题起落、阶段替换和短期冲高的窗口。",
        summary_html=summary_html,
        controls_heading="阅读提示",
        controls_note="这页默认不做额外筛选，重点是把高频主题放在同一张时间轴上。时间跨度较长时，可以直接拖动底部缩放条。",
        controls_html="",
        panels_html=panels_html,
        script_urls=[echarts_js_url],
        inline_script=inline_script,
        resource_links_html="",
    )


def render_map_html(
    map_payload: dict[str, dict[str, list[dict[str, object]]]],
    *,
    periods: list[str],
    top_topics: list[int],
    included_regions: list[str],
    map_min_total_docs: int,
    topic_label_map: dict[int, str],
    echarts_js_url: str,
    china_map_js_url: str,
) -> str:
    map_json = json.dumps(map_payload, ensure_ascii=False)
    topic_options_html = "\n".join(
        f'      <option value="{topic_id}">{topic_label_map.get(topic_id, f"T{topic_id}")}</option>'
        for topic_id in top_topics
    )
    period_options_html = "\n".join(f'      <option value="{period}">{period}</option>' for period in periods)
    summary_html = render_summary_html(
        [
            {
                "label": "地图主题",
                "value": f"Top {len(top_topics)}",
                "detail": "只把高频主题放进省份图层，避免地图切换时噪声太多。",
            },
            {
                "label": "时间切片",
                "value": f"{len(periods)} 期",
                "detail": "既能看全部时间，也能切到单月或单季度。",
            },
            {
                "label": "覆盖地区",
                "value": str(len(included_regions)),
                "detail": f"省份样本少于 {map_min_total_docs} 条时不着色，避免把偶发点误读成热点。",
            },
        ]
    )
    inline_script = f"""
const mapData = {map_json};
const periodSelect = document.getElementById("map-period");
const topicSelect = document.getElementById("map-topic");
const chart = echarts.init(document.getElementById("map-chart"));

function renderMap() {{
  const period = periodSelect.value;
  const topicId = topicSelect.value;
  const current = (mapData[period] && mapData[period][topicId]) ? mapData[period][topicId] : [];
  const threshold = {map_min_total_docs};
  const rendered = current.map((item) => {{
    if (Number(item.period_total || 0) < threshold) {{
      return {{
        ...item,
        value: null,
        insufficient: true
      }};
    }}
    return {{
      ...item,
      insufficient: false
    }};
  }});
  const values = rendered
    .map((item) => Number(item.value))
    .filter((value) => Number.isFinite(value));
  const maxValue = Math.max(5, ...values);
  chart.setOption({{
    animationDuration: 400,
    backgroundColor: "transparent",
    tooltip: {{
      trigger: "item",
      formatter: (params) => {{
        const item = params.data || {{}};
        const total = Number(item.period_total || 0);
        const docCount = Number(item.doc_count || 0);
        const share = Number(item.share_pct || 0);
        if (item.insufficient) {{
          return `${{params.name}}<br/>样本不足：${{total}} 条<br/>主题帖数：${{docCount}}`;
        }}
        return `${{params.name}}<br/>主题占比：${{share.toFixed(2)}}%<br/>主题帖数：${{docCount}}<br/>该省该期总帖数：${{total}}`;
      }}
    }},
    visualMap: {{
      min: 0,
      max: maxValue,
      left: "left",
      bottom: 20,
      text: ["高", "低"],
      calculable: true,
      inRange: {{
        color: ["#eef7f1", "#b7e4c7", "#52b788", "#1b4332"]
      }}
    }},
    series: [{{
      name: "主题占比",
      type: "map",
      map: "china",
      roam: true,
      data: rendered,
      itemStyle: {{
        borderColor: "#d7e1d4",
        areaColor: "#f7faf7"
      }},
      emphasis: {{
        label: {{ show: true, color: "#17212b" }},
        itemStyle: {{
          areaColor: "#ffd166"
        }}
      }}
    }}]
  }});
}}

periodSelect.addEventListener("change", renderMap);
topicSelect.addEventListener("change", renderMap);
window.addEventListener("resize", () => chart.resize());
renderMap();
"""
    controls_html = f"""        <div class="control-group">
          <label for="map-period">时间段</label>
          <select id="map-period">
{period_options_html}
          </select>
        </div>
        <div class="control-group">
          <label for="map-topic">主题</label>
          <select id="map-topic">
{topic_options_html}
          </select>
        </div>"""
    panels_html = """      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="panel-eyebrow">空间分布</p>
            <h2>省份主题热力图</h2>
          </div>
          <p class="panel-note">切换时间和主题后，地图显示的是各省份内部该主题的相对占比；样本少于阈值的省份会留白，避免把 1/1 这种点误判为热点。</p>
        </div>
        <div id="map-chart" class="chart tall"></div>
      </section>"""
    return html_shell(
        title="省份主题热力图",
        subtitle="把 topic share 的 IP 维度投到中国地图上，适合快速看主题地域差异、阶段扩散和省份间冷热对照。",
        summary_html=summary_html,
        controls_heading="筛选视角",
        controls_note=f"先定时间，再切主题。地图显示的是各省份内部该主题的相对占比，因此更适合比较空间结构，而不是单纯比较帖子总量；样本少于 {map_min_total_docs} 条的省份会自动留白。",
        controls_html=controls_html,
        panels_html=panels_html,
        script_urls=[echarts_js_url, china_map_js_url],
        inline_script=inline_script,
        resource_links_html="",
    )


def main() -> None:
    args = parse_args()
    emit = resolve_emit("topic-viz", None)
    total_start = time.perf_counter()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emit("Loading topic outputs")
    topic_info = pd.read_csv(args.topic_info_path)
    topic_terms = pd.read_csv(args.topic_terms_path)
    share_by_period = pd.read_csv(args.topic_share_by_period_path)
    share_by_period_keyword = pd.read_csv(args.topic_share_by_period_keyword_path)
    share_by_ip = pd.read_csv(args.topic_share_by_ip_path)
    share_by_period_ip = pd.read_csv(args.topic_share_by_period_ip_path)
    emit(
        "Loaded tables "
        f"(topic_info={len(topic_info)}, topic_terms={len(topic_terms)}, "
        f"share_by_period={len(share_by_period)}, share_by_period_keyword={len(share_by_period_keyword)}, "
        f"share_by_ip={len(share_by_ip)}, share_by_period_ip={len(share_by_period_ip)})"
    )

    topic_label_map = build_topic_label_map(topic_info, topic_terms)
    profile_top_topics = choose_top_topics(share_by_period.rename(columns={"Topic": "topic_id"}), top_n=args.top_n_topics)
    map_top_topics = choose_top_topics(share_by_period.rename(columns={"Topic": "topic_id"}), top_n=args.map_top_n_topics)

    emit(
        "Selected top topics "
        f"for profile/time={profile_top_topics}, map={map_top_topics}"
    )

    profile_payload, overall_matrix_df, latest_matrix_df = build_profile_payload(
        share_by_period_keyword,
        topic_label_map=topic_label_map,
        top_topics=profile_top_topics,
    )
    time_payload = build_time_payload(
        share_by_period,
        topic_label_map=topic_label_map,
        top_topics=profile_top_topics,
    )
    map_region_df, map_payload, map_periods, map_topic_ids, map_regions = build_map_payload(
        share_by_ip,
        share_by_period_ip,
        topic_label_map=topic_label_map,
        top_topics=map_top_topics,
    )

    emit("Rendering HTML outputs")
    profile_html = render_profile_html(profile_payload, echarts_js_url=args.echarts_js_url)
    time_html = render_time_html(time_payload, echarts_js_url=args.echarts_js_url)
    map_html = render_map_html(
        map_payload,
        periods=map_periods,
        top_topics=map_topic_ids,
        included_regions=map_regions,
        map_min_total_docs=args.map_min_total_docs,
        topic_label_map=topic_label_map,
        echarts_js_url=args.echarts_js_url,
        china_map_js_url=args.china_map_js_url,
    )

    profile_html_path = output_dir / "topic_keyword_profiles.html"
    time_html_path = output_dir / "topic_time_evolution.html"
    map_html_path = output_dir / "topic_ip_heatmap.html"
    overall_matrix_path = output_dir / "topic_keyword_profile_overall.csv"
    latest_matrix_path = output_dir / "topic_keyword_profile_latest.csv"
    summary_path = output_dir / "topic_visualization_summary.json"

    profile_html_path.write_text(profile_html, encoding="utf-8")
    time_html_path.write_text(time_html, encoding="utf-8")
    map_html_path.write_text(map_html, encoding="utf-8")
    save_dataframe(overall_matrix_df, overall_matrix_path)
    save_dataframe(latest_matrix_df, latest_matrix_path)

    save_json(
        summary_path,
        {
            "output_dir": str(output_dir.resolve()),
            "profile_html_path": str(profile_html_path.resolve()),
            "time_html_path": str(time_html_path.resolve()),
            "map_html_path": str(map_html_path.resolve()),
            "overall_matrix_path": str(overall_matrix_path.resolve()),
            "latest_matrix_path": str(latest_matrix_path.resolve()),
            "profile_top_topics": profile_top_topics,
            "map_top_topics": map_top_topics,
            "map_periods": map_periods,
            "china_regions_included": map_regions,
            "map_min_total_docs": int(args.map_min_total_docs),
            "map_region_rows": int(len(map_region_df)),
            "echarts_js_url": args.echarts_js_url,
            "china_map_js_url": args.china_map_js_url,
        },
    )

    emit(f"Saved visualization bundle to {output_dir}")
    emit(f"Total runtime: {format_elapsed(total_start)}")


if __name__ == "__main__":
    main()
