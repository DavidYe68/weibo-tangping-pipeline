#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path

import pandas as pd

from lib.analysis_utils import resolve_emit, sort_period_labels
from lib.io_utils import save_json

ECHARTS_JS_URL = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"
ECHARTS_WORDCLOUD_JS_URL = "https://cdn.jsdelivr.net/npm/echarts-wordcloud@2/dist/echarts-wordcloud.min.js"
DEFAULT_TOPIC_MODEL_DIR = Path("bert/artifacts/broad_analysis/topic_model_BAAI")
DEFAULT_TOPIC_VISUALIZATION_DIR = Path("bert/artifacts/broad_analysis/topic_visualization")
LEADING_TOPIC_ID_RE = re.compile(r"^-?\d+_")


def format_elapsed(start_time: float) -> str:
    return f"{time.perf_counter() - start_time:.2f}s"


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def shorten_label(text: str, *, max_length: int = 42) -> str:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate topic-centric interpretability visualizations from BERTopic outputs."
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
        "--output_dir",
        default=str(DEFAULT_TOPIC_VISUALIZATION_DIR),
        help="Directory for generated topic-visualization outputs.",
    )
    parser.add_argument(
        "--top_n_topics",
        type=int,
        default=20,
        help="Number of head topics to keep in overview and evolution plots.",
    )
    parser.add_argument(
        "--wordcloud_top_n_topics",
        type=int,
        default=12,
        help="Number of head topics to render in the word-cloud gallery.",
    )
    parser.add_argument(
        "--top_n_terms",
        type=int,
        default=15,
        help="Top terms per topic to display.",
    )
    parser.add_argument(
        "--min_period_docs",
        type=int,
        default=1000,
        help="Drop very sparse tail periods from evolution plots.",
    )
    parser.add_argument(
        "--coding_template_top_n",
        type=int,
        default=80,
        help="Number of head topics to include in the manual coding template.",
    )
    parser.add_argument(
        "--echarts_js_url",
        default=ECHARTS_JS_URL,
        help="URL for echarts.min.js.",
    )
    parser.add_argument(
        "--echarts_wordcloud_js_url",
        default=ECHARTS_WORDCLOUD_JS_URL,
        help="URL for the echarts-wordcloud plugin.",
    )
    return parser.parse_args()


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


def build_topic_rows(
    topic_info: pd.DataFrame,
    topic_terms_lookup: dict[int, list[dict[str, object]]],
    share_by_period: pd.DataFrame,
    *,
    top_n_terms: int,
) -> tuple[list[dict[str, object]], int, int]:
    info = topic_info.copy()
    info["Topic"] = pd.to_numeric(info["Topic"], errors="coerce").astype("Int64")
    info["Count"] = pd.to_numeric(info["Count"], errors="coerce").fillna(0).astype(int)
    outlier_count = int(info.loc[info["Topic"] == -1, "Count"].sum())
    non_outlier_total = int(info.loc[info["Topic"] >= 0, "Count"].sum())
    total_docs = outlier_count + non_outlier_total

    share = share_by_period.copy()
    share["topic_id"] = pd.to_numeric(share["topic_id"], errors="coerce").astype("Int64")
    share["doc_count"] = pd.to_numeric(share["doc_count"], errors="coerce").fillna(0).astype(int)
    share["doc_share"] = pd.to_numeric(share["doc_share"], errors="coerce").fillna(0.0)

    rows: list[dict[str, object]] = []
    for row in info.itertuples(index=False):
        if pd.isna(row.Topic):
            continue
        topic_id = int(row.Topic)
        if topic_id < 0:
            continue

        terms = topic_terms_lookup.get(topic_id, [])[:top_n_terms]
        term_labels = [item["term"] for item in terms]
        machine_label = clean_text(getattr(row, "topic_label_machine", "")) or clean_text(getattr(row, "Name", ""))
        zh_label = clean_text(getattr(row, "topic_label_zh", ""))
        normalized_machine_label = normalize_topic_name(machine_label)

        if zh_label:
            base_label = zh_label
        elif term_labels:
            base_label = " / ".join(term_labels[:4])
        else:
            base_label = normalized_machine_label or f"Topic {topic_id}"

        share_frame = share[share["topic_id"] == topic_id].copy()
        peak_period = ""
        peak_doc_count = 0
        peak_doc_share_pct = 0.0
        if not share_frame.empty:
            peak_row = share_frame.sort_values(["doc_count", "doc_share"], ascending=[False, False]).iloc[0]
            peak_period = clean_text(peak_row["period_label"])
            peak_doc_count = int(peak_row["doc_count"])
            peak_doc_share_pct = float(peak_row["doc_share"]) * 100.0

        representative_docs = [
            shorten_label(doc, max_length=160) for doc in parse_representative_docs(getattr(row, "Representative_Docs", ""))[:3]
        ]
        count = int(row.Count)
        rows.append(
            {
                "topic_id": topic_id,
                "topic_count": count,
                "share_of_all_docs_pct": (100.0 * count / total_docs) if total_docs else 0.0,
                "share_of_clustered_docs_pct": (100.0 * count / non_outlier_total) if non_outlier_total else 0.0,
                "topic_label": f"T{topic_id} {shorten_label(base_label)}",
                "topic_label_full": f"T{topic_id} {base_label}",
                "topic_label_zh": zh_label,
                "topic_label_machine": normalized_machine_label,
                "top_terms": term_labels,
                "term_payload": terms,
                "peak_period": peak_period,
                "peak_doc_count": peak_doc_count,
                "peak_doc_share_pct": peak_doc_share_pct,
                "representative_docs": representative_docs,
            }
        )
    rows.sort(key=lambda item: (-int(item["topic_count"]), int(item["topic_id"])))
    return rows, total_docs, outlier_count


def build_period_payload(
    share_by_period: pd.DataFrame,
    *,
    top_topic_ids: list[int],
    min_period_docs: int,
) -> tuple[list[str], dict[str, int], dict[int, dict[str, float]]]:
    working = share_by_period.copy()
    working["topic_id"] = pd.to_numeric(working["topic_id"], errors="coerce").astype("Int64")
    working["doc_count"] = pd.to_numeric(working["doc_count"], errors="coerce").fillna(0).astype(int)
    working["doc_share"] = pd.to_numeric(working["doc_share"], errors="coerce").fillna(0.0)
    working["period_label"] = working["period_label"].astype(str)

    all_periods = working["period_label"].dropna().astype(str).tolist()
    period_totals_series = working.groupby("period_label", as_index=True)["doc_count"].sum()
    period_totals = {str(period): int(total) for period, total in period_totals_series.items()}
    ordered_periods = sort_period_labels(list(period_totals.keys()), infer_period_granularity(all_periods))
    filtered_periods = [period for period in ordered_periods if period_totals.get(period, 0) >= min_period_docs]

    share_lookup: dict[int, dict[str, float]] = {topic_id: {} for topic_id in top_topic_ids}
    subset = working[working["topic_id"].isin(top_topic_ids) & working["period_label"].isin(filtered_periods)].copy()
    for row in subset.itertuples(index=False):
        share_lookup[int(row.topic_id)][str(row.period_label)] = float(row.doc_share) * 100.0
    return filtered_periods, {period: period_totals.get(period, 0) for period in filtered_periods}, share_lookup


def build_keyword_alignment_payload(
    share_by_period_keyword: pd.DataFrame,
    *,
    topic_rows: list[dict[str, object]],
    selected_topic_ids: list[int],
) -> dict[str, object]:
    selected_set = {int(topic_id) for topic_id in selected_topic_ids}
    topic_label_lookup = {int(row["topic_id"]): str(row["topic_label"]) for row in topic_rows}
    topic_label_full_lookup = {int(row["topic_id"]): str(row["topic_label_full"]) for row in topic_rows}

    empty_payload = {
        "keywords": [],
        "selected_topic_labels": [],
        "heatmap_data": [],
        "heatmap_max": 0.0,
        "keyword_summary": [],
        "share_lookup_all": {},
        "table_rows": [],
    }
    if share_by_period_keyword.empty:
        return empty_payload

    working = share_by_period_keyword.copy()
    if "keyword_normalized" not in working.columns:
        return empty_payload

    working["topic_id"] = pd.to_numeric(working["topic_id"], errors="coerce").astype("Int64")
    working["doc_count"] = pd.to_numeric(working["doc_count"], errors="coerce").fillna(0).astype(int)
    working = working.dropna(subset=["topic_id"]).copy()
    if working.empty:
        return empty_payload

    grouped = (
        working.groupby(["keyword_normalized", "topic_id"], as_index=False)["doc_count"]
        .sum()
        .sort_values(["keyword_normalized", "doc_count", "topic_id"], ascending=[True, False, True])
    )
    grouped["keyword_total"] = grouped.groupby("keyword_normalized")["doc_count"].transform("sum")
    grouped["share_pct"] = grouped["doc_count"] / grouped["keyword_total"] * 100.0

    keywords = sorted(grouped["keyword_normalized"].dropna().astype(str).unique().tolist())
    share_lookup_all: dict[int, dict[str, float]] = {}
    count_lookup_all: dict[int, dict[str, int]] = {}
    for row in grouped.itertuples(index=False):
        topic_id = int(row.topic_id)
        keyword = str(row.keyword_normalized)
        share_lookup_all.setdefault(topic_id, {})[keyword] = float(row.share_pct)
        count_lookup_all.setdefault(topic_id, {})[keyword] = int(row.doc_count)

    heatmap_data: list[list[object]] = []
    table_rows: list[dict[str, object]] = []
    heatmap_max = 0.0
    selected_topic_labels: list[str] = []
    for y_index, topic_id in enumerate(reversed(selected_topic_ids)):
        selected_topic_labels.append(topic_label_lookup.get(topic_id, f"T{topic_id}"))
        row_payload: dict[str, object] = {
            "topic_id": topic_id,
            "topic_label": topic_label_lookup.get(topic_id, f"T{topic_id}"),
            "topic_label_full": topic_label_full_lookup.get(topic_id, f"T{topic_id}"),
        }
        for x_index, keyword in enumerate(keywords):
            share_pct = round(float(share_lookup_all.get(topic_id, {}).get(keyword, 0.0)), 4)
            doc_count = int(count_lookup_all.get(topic_id, {}).get(keyword, 0))
            heatmap_data.append([x_index, y_index, share_pct, doc_count])
            row_payload[f"share_within_{keyword}_pct"] = share_pct
            row_payload[f"doc_count_in_{keyword}"] = doc_count
            heatmap_max = max(heatmap_max, share_pct)
        table_rows.append(row_payload)

    keyword_summary: list[dict[str, object]] = []
    for keyword in keywords:
        keyword_frame = grouped[grouped["keyword_normalized"].astype(str) == keyword].copy()
        if keyword_frame.empty:
            continue
        top_row = keyword_frame.iloc[0]
        top3_share = float(keyword_frame["share_pct"].head(3).sum())
        keyword_summary.append(
            {
                "keyword": keyword,
                "top_topic_id": int(top_row["topic_id"]),
                "top_topic_label": topic_label_lookup.get(int(top_row["topic_id"]), f"T{int(top_row['topic_id'])}"),
                "top_topic_share_pct": float(top_row["share_pct"]),
                "top3_share_pct": top3_share,
            }
        )

    return {
        "keywords": keywords,
        "selected_topic_labels": selected_topic_labels,
        "heatmap_data": heatmap_data,
        "heatmap_max": round(heatmap_max, 4),
        "keyword_summary": keyword_summary,
        "share_lookup_all": share_lookup_all,
        "table_rows": table_rows,
    }


def build_topic_tables(
    topic_rows: list[dict[str, object]],
    *,
    keyword_alignment_payload: dict[str, object],
    coding_template_top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keywords = [str(keyword) for keyword in keyword_alignment_payload.get("keywords", [])]
    share_lookup_all = {
        int(topic_id): {str(keyword): float(value) for keyword, value in shares.items()}
        for topic_id, shares in keyword_alignment_payload.get("share_lookup_all", {}).items()
    }

    overview_rows: list[dict[str, object]] = []
    for rank, row in enumerate(topic_rows, start=1):
        topic_id = int(row["topic_id"])
        base_row = {
            "topic_rank": rank,
            "topic_id": topic_id,
            "topic_label": row["topic_label_full"],
            "topic_count": int(row["topic_count"]),
            "share_of_all_docs_pct": round(float(row["share_of_all_docs_pct"]), 4),
            "share_of_clustered_docs_pct": round(float(row["share_of_clustered_docs_pct"]), 4),
            "peak_period": row["peak_period"],
            "peak_doc_count": int(row["peak_doc_count"]),
            "peak_doc_share_pct": round(float(row["peak_doc_share_pct"]), 4),
            "top_terms": " / ".join(row["top_terms"]),
            "representative_doc_1": row["representative_docs"][0] if len(row["representative_docs"]) > 0 else "",
            "representative_doc_2": row["representative_docs"][1] if len(row["representative_docs"]) > 1 else "",
            "representative_doc_3": row["representative_docs"][2] if len(row["representative_docs"]) > 2 else "",
        }
        for keyword in keywords:
            base_row[f"share_within_{keyword}_pct"] = round(float(share_lookup_all.get(topic_id, {}).get(keyword, 0.0)), 4)
        overview_rows.append(base_row)

    overview_df = pd.DataFrame(overview_rows)
    coding_df = overview_df.head(coding_template_top_n).copy()
    coding_df["manual_theme_bucket"] = ""
    coding_df["manual_research_relevance"] = ""
    coding_df["manual_keep_for_midterm"] = ""
    coding_df["manual_noise_flag"] = ""
    coding_df["manual_note"] = ""
    return overview_df, coding_df


def render_summary_metrics(metrics: list[dict[str, str]]) -> str:
    return "\n".join(
        f"""      <div class="metric">
        <div class="metric-label">{item["label"]}</div>
        <div class="metric-value">{item["value"]}</div>
        <div class="metric-detail">{item["detail"]}</div>
      </div>"""
        for item in metrics
    )


def html_page(
    *,
    title: str,
    subtitle: str,
    metrics_html: str,
    body_html: str,
    inline_script: str,
    script_urls: list[str],
) -> str:
    script_tags = "\n".join(f'  <script src="{url}"></script>' for url in script_urls)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7f9;
      color: #16202a;
    }}
    .page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px 20px 40px;
    }}
    .hero {{
      padding: 4px 0 20px;
      border-bottom: 1px solid #d7dee5;
    }}
    .kicker {{
      margin: 0 0 8px;
      font-size: 12px;
      line-height: 1.2;
      color: #0f766e;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: 32px;
      line-height: 1.1;
    }}
    .subtitle {{
      margin: 12px 0 0;
      max-width: 900px;
      font-size: 15px;
      line-height: 1.55;
      color: #556372;
    }}
    .metrics {{
      margin-top: 18px;
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .metric {{
      padding: 14px 16px;
      border: 1px solid #d7dee5;
      border-radius: 6px;
      background: #ffffff;
    }}
    .metric-label {{
      font-size: 12px;
      line-height: 1.2;
      color: #607180;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .metric-value {{
      margin-top: 8px;
      font-size: 26px;
      line-height: 1.1;
      font-weight: 700;
      color: #132238;
    }}
    .metric-detail {{
      margin-top: 6px;
      font-size: 13px;
      line-height: 1.45;
      color: #607180;
    }}
    .section {{
      padding: 22px 0 0;
    }}
    .section-title {{
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.2;
    }}
    .section-note {{
      margin: 0 0 14px;
      font-size: 14px;
      line-height: 1.5;
      color: #607180;
      max-width: 960px;
    }}
    .panel {{
      border: 1px solid #d7dee5;
      border-radius: 6px;
      background: #ffffff;
      padding: 14px;
    }}
    .chart {{
      width: 100%;
      height: 680px;
    }}
    .chart.medium {{
      height: 560px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
      margin-bottom: 12px;
    }}
    .control-group {{
      display: grid;
      gap: 6px;
      min-width: 280px;
    }}
    label {{
      font-size: 13px;
      font-weight: 600;
      color: #425264;
    }}
    select {{
      height: 40px;
      border: 1px solid #c8d2dc;
      border-radius: 6px;
      background: #ffffff;
      padding: 0 10px;
      font-size: 14px;
      color: #16202a;
    }}
    .topic-meta {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      margin-bottom: 12px;
    }}
    .topic-meta-item {{
      padding: 12px;
      border: 1px solid #d7dee5;
      border-radius: 6px;
      background: #f8fafb;
    }}
    .topic-meta-label {{
      font-size: 12px;
      color: #607180;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .topic-meta-value {{
      margin-top: 6px;
      font-size: 18px;
      line-height: 1.25;
      font-weight: 700;
      color: #132238;
    }}
    .topic-meta-detail {{
      margin-top: 4px;
      font-size: 13px;
      line-height: 1.45;
      color: #607180;
    }}
    .doc-list {{
      margin: 14px 0 0;
      padding-left: 18px;
      color: #425264;
      font-size: 14px;
      line-height: 1.55;
    }}
    .doc-list li + li {{
      margin-top: 8px;
    }}
    .cloud-grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    .cloud-card {{
      border: 1px solid #d7dee5;
      border-radius: 6px;
      background: #ffffff;
      padding: 14px;
    }}
    .cloud-card h3 {{
      margin: 0;
      font-size: 17px;
      line-height: 1.3;
    }}
    .cloud-card p {{
      margin: 8px 0 0;
      font-size: 13px;
      line-height: 1.5;
      color: #607180;
    }}
    .wordcloud {{
      width: 100%;
      height: 300px;
      margin-top: 12px;
    }}
    @media (max-width: 900px) {{
      .page {{
        padding: 18px 14px 28px;
      }}
      h1 {{
        font-size: 28px;
      }}
      .chart {{
        height: 520px;
      }}
      .chart.medium {{
        height: 460px;
      }}
      .wordcloud {{
        height: 260px;
      }}
      .control-group {{
        min-width: min(100%, 320px);
      }}
    }}
  </style>
{script_tags}
</head>
<body>
  <div class="page">
    <section class="hero">
      <p class="kicker">Topic Model Interpretability</p>
      <h1>{title}</h1>
      <p class="subtitle">{subtitle}</p>
      <div class="metrics">
{metrics_html}
      </div>
    </section>
{body_html}
  </div>
  <script>
{inline_script}
  </script>
</body>
</html>
"""


def render_prevalence_html(
    topic_rows: list[dict[str, object]],
    *,
    total_docs: int,
    outlier_count: int,
    top_n_topics: int,
    echarts_js_url: str,
) -> str:
    selected = topic_rows[:top_n_topics]
    payload = [
        {
            "topic_id": row["topic_id"],
            "label": row["topic_label"],
            "topic_count": row["topic_count"],
            "share_all": round(float(row["share_of_all_docs_pct"]), 4),
            "share_clustered": round(float(row["share_of_clustered_docs_pct"]), 4),
            "peak_period": row["peak_period"],
            "peak_doc_count": row["peak_doc_count"],
            "peak_doc_share_pct": round(float(row["peak_doc_share_pct"]), 4),
            "top_terms": row["top_terms"][:6],
        }
        for row in reversed(selected)
    ]
    metrics_html = render_summary_metrics(
        [
            {
                "label": "总文档量",
                "value": f"{total_docs:,}",
                "detail": "包含 outlier 与已聚类文档。",
            },
            {
                "label": "已聚类文档",
                "value": f"{(total_docs - outlier_count):,}",
                "detail": f"{(100.0 * (total_docs - outlier_count) / total_docs):.2f}% 的文档进入具体主题。"
                if total_docs
                else "暂无数据。",
            },
            {
                "label": "Outlier",
                "value": f"{outlier_count:,}",
                "detail": f"{(100.0 * outlier_count / total_docs):.2f}% 被分到 -1。"
                if total_docs
                else "暂无数据。",
            },
            {
                "label": "头部主题",
                "value": f"Top {top_n_topics}",
                "detail": "按 Count 排序，适合先看语义最稳定的一层。",
            },
        ]
    )
    body_html = """    <section class="section">
      <h2 class="section-title">主题规模分布</h2>
      <p class="section-note">先看哪些主题真正“站住了”。这一步比看关键词切面更接近 BERTopic 的核心结果，因为它直接回答哪些语义簇在语料中最稳定、最可解释。</p>
      <div class="panel">
        <div id="prevalence-chart" class="chart"></div>
      </div>
    </section>"""
    inline_script = f"""
const prevalenceData = {json.dumps(payload, ensure_ascii=False)};
const prevalenceChart = echarts.init(document.getElementById("prevalence-chart"));
prevalenceChart.setOption({{
  animationDuration: 500,
  grid: {{ left: 210, right: 34, top: 30, bottom: 28 }},
  tooltip: {{
    trigger: "axis",
    axisPointer: {{ type: "shadow" }},
    formatter: (params) => {{
      const row = params[0].data.row;
      const terms = row.top_terms.join(" / ");
      return `<strong>${{row.label}}</strong><br>` +
        `文档数: ${{row.topic_count.toLocaleString()}}<br>` +
        `占全部文档: ${{row.share_all.toFixed(2)}}%<br>` +
        `占已聚类文档: ${{row.share_clustered.toFixed(2)}}%<br>` +
        `峰值时间: ${{row.peak_period || "NA"}}<br>` +
        `峰值文档数: ${{row.peak_doc_count.toLocaleString()}}<br>` +
        `峰值期占比: ${{row.peak_doc_share_pct.toFixed(2)}}%<br>` +
        `Top terms: ${{terms}}`;
    }}
  }},
  xAxis: {{
    type: "value",
    axisLabel: {{ color: "#425264" }},
    splitLine: {{ lineStyle: {{ color: "#dde4eb" }} }}
  }},
  yAxis: {{
    type: "category",
    axisLabel: {{ color: "#425264", width: 180, overflow: "truncate" }},
    data: prevalenceData.map((item) => item.label)
  }},
  series: [{{
    type: "bar",
    data: prevalenceData.map((item) => ({{
      value: item.topic_count,
      row: item,
      itemStyle: {{
        color: item.share_clustered >= 2.0 ? "#2563eb" : item.share_clustered >= 1.0 ? "#0891b2" : "#0f766e"
      }}
    }})),
    label: {{
      show: true,
      position: "right",
      color: "#425264",
      formatter: (params) => params.data.row.share_clustered.toFixed(2) + "%"
    }},
    barMaxWidth: 28
  }}]
}});
window.addEventListener("resize", () => prevalenceChart.resize());
"""
    return html_page(
        title="Topic Prevalence",
        subtitle="头部主题的规模分布。这里优先看 Count、占已聚类文档比例，以及各主题的峰值时间，而不是先看外部维度切片。",
        metrics_html=metrics_html,
        body_html=body_html,
        inline_script=inline_script,
        script_urls=[echarts_js_url],
    )


def render_term_detail_html(
    topic_rows: list[dict[str, object]],
    *,
    echarts_js_url: str,
) -> str:
    payload = [
        {
            "topic_id": row["topic_id"],
            "label": row["topic_label"],
            "label_full": row["topic_label_full"],
            "topic_count": row["topic_count"],
            "share_all": round(float(row["share_of_all_docs_pct"]), 4),
            "share_clustered": round(float(row["share_of_clustered_docs_pct"]), 4),
            "peak_period": row["peak_period"],
            "peak_doc_count": row["peak_doc_count"],
            "peak_doc_share_pct": round(float(row["peak_doc_share_pct"]), 4),
            "terms": [
                {
                    "term": item["term"],
                    "weight": round(float(item["term_weight"]), 8),
                }
                for item in row["term_payload"]
            ],
            "docs": row["representative_docs"],
        }
        for row in topic_rows
    ]
    options_html = "\n".join(
        f'            <option value="{item["topic_id"]}">{item["label"]}</option>' for item in payload
    )
    metrics_html = render_summary_metrics(
        [
            {
                "label": "阅读目的",
                "value": "解释主题",
                "detail": "比词云更精确地看每个 topic 的高权重词。",
            },
            {
                "label": "代表文本",
                "value": "Top 3",
                "detail": "用代表文档辅助判断主题语义是否跑偏。",
            },
            {
                "label": "排序方式",
                "value": "c-TF-IDF",
                "detail": "条形图按 term weight 从高到低排列。",
            },
        ]
    )
    body_html = f"""    <section class="section">
      <h2 class="section-title">主题词与代表文本</h2>
      <p class="section-note">这一步才是 topic interpretability 的核心。词云给直观感受，条形图给精确权重，代表文本负责校验这些词是不是在讲同一个语义对象。</p>
      <div class="panel">
        <div class="toolbar">
          <div class="control-group">
            <label for="topic-select">选择主题</label>
            <select id="topic-select">
{options_html}
            </select>
          </div>
        </div>
        <div id="topic-meta" class="topic-meta"></div>
        <div id="term-chart" class="chart medium"></div>
        <ul id="doc-list" class="doc-list"></ul>
      </div>
    </section>"""
    inline_script = f"""
const topicDetailData = {json.dumps(payload, ensure_ascii=False)};
const topicSelect = document.getElementById("topic-select");
const topicMeta = document.getElementById("topic-meta");
const docList = document.getElementById("doc-list");
const termChart = echarts.init(document.getElementById("term-chart"));

function renderTopicDetail() {{
  const current = topicDetailData.find((item) => String(item.topic_id) === topicSelect.value) || topicDetailData[0];
  topicMeta.innerHTML = `
    <div class="topic-meta-item">
      <div class="topic-meta-label">主题标签</div>
      <div class="topic-meta-value">${{current.label_full}}</div>
      <div class="topic-meta-detail">按 Count 排序的稳定主题。</div>
    </div>
    <div class="topic-meta-item">
      <div class="topic-meta-label">文档数</div>
      <div class="topic-meta-value">${{current.topic_count.toLocaleString()}}</div>
      <div class="topic-meta-detail">占全部文档 ${{current.share_all.toFixed(2)}}%，占已聚类文档 ${{current.share_clustered.toFixed(2)}}%。</div>
    </div>
    <div class="topic-meta-item">
      <div class="topic-meta-label">峰值时间</div>
      <div class="topic-meta-value">${{current.peak_period || "NA"}}</div>
      <div class="topic-meta-detail">峰值文档数 ${{current.peak_doc_count.toLocaleString()}}，该期占比 ${{current.peak_doc_share_pct.toFixed(2)}}%。</div>
    </div>
  `;

  termChart.setOption({{
    animationDuration: 400,
    grid: {{ left: 86, right: 24, top: 24, bottom: 70 }},
    tooltip: {{
      trigger: "axis",
      axisPointer: {{ type: "shadow" }},
      formatter: (params) => {{
        const row = params[0].data.row;
        return `<strong>${{row.term}}</strong><br>term weight: ${{row.weight.toFixed(6)}}`;
      }}
    }},
    xAxis: {{
      type: "category",
      data: current.terms.map((item) => item.term),
      axisLabel: {{ interval: 0, rotate: 35, color: "#425264" }},
      axisLine: {{ lineStyle: {{ color: "#c8d2dc" }} }}
    }},
    yAxis: {{
      type: "value",
      axisLabel: {{ color: "#425264" }},
      splitLine: {{ lineStyle: {{ color: "#dde4eb" }} }}
    }},
    series: [{{
      type: "bar",
      data: current.terms.map((item) => ({{
        value: item.weight,
        row: item,
        itemStyle: {{ color: "#2563eb" }}
      }})),
      barMaxWidth: 30
    }}]
  }});

  const docs = current.docs.length ? current.docs : ["没有可用的 representative docs。"];
  docList.innerHTML = docs.map((doc) => `<li>${{doc}}</li>`).join("");
}}

topicSelect.addEventListener("change", renderTopicDetail);
window.addEventListener("resize", () => termChart.resize());
renderTopicDetail();
"""
    return html_page(
        title="Topic Terms And Exemplars",
        subtitle="单主题解释页。用高权重词和代表文本一起校验主题语义，避免只看词云时把“常见词”误认为“主题中心”。",
        metrics_html=metrics_html,
        body_html=body_html,
        inline_script=inline_script,
        script_urls=[echarts_js_url],
    )


def render_keyword_alignment_html(
    keyword_alignment_payload: dict[str, object],
    *,
    echarts_js_url: str,
) -> str:
    metrics_html = render_summary_metrics(
        [
            {
                "label": "研究对应",
                "value": "关键词 x Topic",
                "detail": "把 BERTopic 结果重新接回“躺平 / 摆烂 / 佛系”三个研究入口。",
            },
            {
                "label": "读法",
                "value": "颜色越深越集中",
                "detail": "单元格数值表示“某关键词内部，有多少比例落入这个 topic”。",
            },
            {
                "label": "展示对象",
                "value": "头部主题",
                "detail": "优先看稳定 topic，先解释大盘，再补尾部细分。",
            },
        ]
    )
    body_html = """    <section class="section">
      <h2 class="section-title">关键词与主题对应关系</h2>
      <p class="section-note">这张图最适合中期汇报。它不再把重点放在省份或时间切面，而是直接回答：三个关键词分别被哪些 topic 主导，以及哪些 topic 同时跨越多个关键词。</p>
      <div class="panel">
        <div id="keyword-alignment-chart" class="chart medium"></div>
      </div>
    </section>"""
    inline_script = f"""
const keywordAlignmentData = {json.dumps(keyword_alignment_payload, ensure_ascii=False)};
const keywordAlignmentChart = echarts.init(document.getElementById("keyword-alignment-chart"));
keywordAlignmentChart.setOption({{
  animationDuration: 400,
  tooltip: {{
    position: "top",
    formatter: (params) => {{
      const keyword = keywordAlignmentData.keywords[params.data[0]];
      const label = keywordAlignmentData.selected_topic_labels[params.data[1]];
      const share = Number(params.data[2] || 0);
      const docCount = Number(params.data[3] || 0);
      return `<strong>${{label}}</strong><br>` +
        `关键词: ${{keyword}}<br>` +
        `关键词内部占比: ${{share.toFixed(2)}}%<br>` +
        `文档数: ${{docCount.toLocaleString()}}`;
    }}
  }},
  grid: {{ left: 210, right: 28, top: 26, bottom: 56 }},
  xAxis: {{
    type: "category",
    data: keywordAlignmentData.keywords,
    axisLabel: {{ color: "#425264" }},
    axisLine: {{ lineStyle: {{ color: "#c8d2dc" }} }}
  }},
  yAxis: {{
    type: "category",
    data: keywordAlignmentData.selected_topic_labels,
    axisLabel: {{ color: "#425264", width: 180, overflow: "truncate" }},
    axisLine: {{ lineStyle: {{ color: "#c8d2dc" }} }}
  }},
  visualMap: {{
    min: 0,
    max: Math.max(Number(keywordAlignmentData.heatmap_max || 0), 1),
    calculable: true,
    orient: "horizontal",
    left: "center",
    bottom: 14,
    inRange: {{
      color: ["#eef5fb", "#bfdbfe", "#60a5fa", "#2563eb", "#1d4ed8"]
    }}
  }},
  series: [{{
    type: "heatmap",
    data: keywordAlignmentData.heatmap_data,
    label: {{
      show: true,
      color: "#102a43",
      formatter: (params) => Number(params.data[2] || 0).toFixed(1) + "%"
    }},
    emphasis: {{
      itemStyle: {{
        shadowBlur: 10,
        shadowColor: "rgba(0, 0, 0, 0.18)"
      }}
    }}
  }}]
}});
window.addEventListener("resize", () => keywordAlignmentChart.resize());
"""
    return html_page(
        title="Keyword Topic Alignment",
        subtitle="把 BERTopic 结果接回研究问题：看“躺平 / 摆烂 / 佛系”分别被哪些主题吸附，以及哪些主题跨关键词共享。",
        metrics_html=metrics_html,
        body_html=body_html,
        inline_script=inline_script,
        script_urls=[echarts_js_url],
    )


def render_wordcloud_html(
    topic_rows: list[dict[str, object]],
    *,
    echarts_js_url: str,
    echarts_wordcloud_js_url: str,
) -> str:
    payload = []
    for row in topic_rows:
        terms = row["term_payload"]
        if not terms:
            continue
        max_weight = max(float(item["term_weight"]) for item in terms) or 1.0
        payload.append(
            {
                "topic_id": row["topic_id"],
                "label": row["topic_label"],
                "topic_count": row["topic_count"],
                "share_clustered": round(float(row["share_of_clustered_docs_pct"]), 4),
                "peak_period": row["peak_period"],
                "words": [
                    {
                        "name": item["term"],
                        "value": max(1, round(float(item["term_weight"]) / max_weight * 100)),
                        "raw_weight": round(float(item["term_weight"]), 8),
                    }
                    for item in terms
                ],
            }
        )
    metrics_html = render_summary_metrics(
        [
            {
                "label": "阅读方式",
                "value": "词云速览",
                "detail": "适合先抓每个 topic 的语义外形。",
            },
            {
                "label": "主题数量",
                "value": str(len(payload)),
                "detail": "默认展示头部主题，避免一次放太多导致阅读失焦。",
            },
            {
                "label": "注意事项",
                "value": "先看大词",
                "detail": "词云只负责直觉，不负责精确比较权重差。",
            },
        ]
    )
    cards_html = "\n".join(
        f"""      <div class="cloud-card">
        <h3>{item["label"]}</h3>
        <p>文档数 {item["topic_count"]:,}，占已聚类文档 {item["share_clustered"]:.2f}%，峰值时间 {item["peak_period"] or "NA"}。</p>
        <div id="wordcloud-{item["topic_id"]}" class="wordcloud"></div>
      </div>"""
        for item in payload
    )
    body_html = f"""    <section class="section">
      <h2 class="section-title">主题词云</h2>
      <p class="section-note">词云不是论文里的唯一图，但它很适合第一眼呈现 topic 的语义轮廓。这里每个词云只保留该 topic 的高权重词，不再混入 keyword、IP 之类的外部切面。</p>
      <div class="cloud-grid">
{cards_html}
      </div>
    </section>"""
    inline_script_lines = [
        f"const wordcloudPayload = {json.dumps(payload, ensure_ascii=False)};",
        """const wordcloudPalette = ["#2563eb", "#0891b2", "#0f766e", "#4f46e5", "#d97706", "#dc2626", "#65a30d"];
wordcloudPayload.forEach((item) => {
  const chart = echarts.init(document.getElementById(`wordcloud-${item.topic_id}`));
  chart.setOption({
    tooltip: {
      formatter: (params) => `<strong>${params.data.name}</strong><br>scaled weight: ${params.data.value}<br>term weight: ${params.data.raw_weight.toFixed(6)}`
    },
    series: [{
      type: "wordCloud",
      shape: "circle",
      gridSize: 6,
      sizeRange: [14, 54],
      rotationRange: [-45, 45],
      textStyle: {
        color: () => wordcloudPalette[Math.floor(Math.random() * wordcloudPalette.length)]
      },
      data: item.words
    }]
  });
  window.addEventListener("resize", () => chart.resize());
});""",
    ]
    return html_page(
        title="Topic Word Clouds",
        subtitle="头部主题的词云速览。它不替代 term-weight 条形图，但很适合快速确认主题有没有明显跑偏。",
        metrics_html=metrics_html,
        body_html=body_html,
        inline_script="\n".join(inline_script_lines),
        script_urls=[echarts_js_url, echarts_wordcloud_js_url],
    )


def render_evolution_html(
    topic_rows: list[dict[str, object]],
    periods: list[str],
    period_totals: dict[str, int],
    share_lookup: dict[int, dict[str, float]],
    *,
    echarts_js_url: str,
) -> str:
    y_topics = list(reversed(topic_rows))
    heatmap_data = []
    for y_index, row in enumerate(y_topics):
        for x_index, period in enumerate(periods):
            heatmap_data.append(
                [
                    x_index,
                    y_index,
                    round(float(share_lookup.get(int(row["topic_id"]), {}).get(period, 0.0)), 4),
                ]
            )
    payload = {
        "periods": periods,
        "period_totals": [period_totals.get(period, 0) for period in periods],
        "topic_labels": [row["topic_label"] for row in y_topics],
        "heatmap_data": heatmap_data,
    }
    metrics_html = render_summary_metrics(
        [
            {
                "label": "时间窗",
                "value": f"{len(periods)} 期",
                "detail": f"{periods[0]} 至 {periods[-1]}" if periods else "暂无可用时间窗。",
            },
            {
                "label": "过滤规则",
                "value": "稀疏期已剔除",
                "detail": "默认去掉文档数过低的尾部时间点，避免把偶然噪声误判成 topic 波动。",
            },
            {
                "label": "着色单位",
                "value": "期内占比",
                "detail": "颜色越深，说明该主题在该时间期里占比越高。",
            },
        ]
    )
    body_html = """    <section class="section">
      <h2 class="section-title">主题时间演化热力图</h2>
      <p class="section-note">把 topic 当作行、时间当作列，看哪些主题有持续存在性，哪些主题更像阶段性事件峰值。这比只看单条折线更适合做整体解释。</p>
      <div class="panel">
        <div id="evolution-chart" class="chart"></div>
      </div>
    </section>"""
    inline_script = f"""
const evolutionData = {json.dumps(payload, ensure_ascii=False)};
const evolutionChart = echarts.init(document.getElementById("evolution-chart"));
evolutionChart.setOption({{
  animationDuration: 400,
  tooltip: {{
    position: "top",
    formatter: (params) => {{
      const period = evolutionData.periods[params.data[0]];
      const label = evolutionData.topic_labels[params.data[1]];
      const totalDocs = evolutionData.period_totals[params.data[0]];
      return `<strong>${{label}}</strong><br>` +
        `时间: ${{period}}<br>` +
        `topic 占比: ${{params.data[2].toFixed(2)}}%<br>` +
        `该期已聚类文档: ${{Number(totalDocs).toLocaleString()}}`;
    }}
  }},
  grid: {{ left: 210, right: 26, top: 26, bottom: 90 }},
  xAxis: {{
    type: "category",
    data: evolutionData.periods,
    axisLabel: {{ interval: 0, rotate: 35, color: "#425264" }},
    axisLine: {{ lineStyle: {{ color: "#c8d2dc" }} }}
  }},
  yAxis: {{
    type: "category",
    data: evolutionData.topic_labels,
    axisLabel: {{ color: "#425264", width: 180, overflow: "truncate" }},
    axisLine: {{ lineStyle: {{ color: "#c8d2dc" }} }}
  }},
  visualMap: {{
    min: 0,
    max: 12,
    calculable: true,
    orient: "horizontal",
    left: "center",
    bottom: 20,
    inRange: {{
      color: ["#eef5fb", "#bfdbfe", "#60a5fa", "#2563eb", "#1d4ed8"]
    }}
  }},
  series: [{{
    type: "heatmap",
    data: evolutionData.heatmap_data,
    label: {{
      show: false
    }},
    emphasis: {{
      itemStyle: {{
        shadowBlur: 10,
        shadowColor: "rgba(0, 0, 0, 0.18)"
      }}
    }}
  }}]
}});
window.addEventListener("resize", () => evolutionChart.resize());
"""
    return html_page(
        title="Topic Evolution Heatmap",
        subtitle="头部主题在各时间期中的占比热力图。它更适合看 topic 的持续性、波峰和事件性，而不是把所有变化都压成几条线。",
        metrics_html=metrics_html,
        body_html=body_html,
        inline_script=inline_script,
        script_urls=[echarts_js_url],
    )


def render_midterm_dashboard_html(
    topic_rows: list[dict[str, object]],
    periods: list[str],
    period_totals: dict[str, int],
    share_lookup: dict[int, dict[str, float]],
    keyword_alignment_payload: dict[str, object],
    *,
    total_docs: int,
    outlier_count: int,
    top_n_topics: int,
    echarts_js_url: str,
    appendix_links: list[dict[str, str]],
) -> str:
    overview_rows = topic_rows[:top_n_topics]
    prevalence_payload = [
        {
            "topic_id": row["topic_id"],
            "label": row["topic_label"],
            "label_full": row["topic_label_full"],
            "topic_count": row["topic_count"],
            "share_all": round(float(row["share_of_all_docs_pct"]), 4),
            "share_clustered": round(float(row["share_of_clustered_docs_pct"]), 4),
            "peak_period": row["peak_period"],
            "peak_doc_count": row["peak_doc_count"],
            "peak_doc_share_pct": round(float(row["peak_doc_share_pct"]), 4),
            "top_terms": row["top_terms"][:6],
        }
        for row in reversed(overview_rows)
    ]
    detail_payload = [
        {
            "topic_id": row["topic_id"],
            "label": row["topic_label"],
            "label_full": row["topic_label_full"],
            "topic_count": row["topic_count"],
            "share_all": round(float(row["share_of_all_docs_pct"]), 4),
            "share_clustered": round(float(row["share_of_clustered_docs_pct"]), 4),
            "peak_period": row["peak_period"],
            "peak_doc_count": row["peak_doc_count"],
            "peak_doc_share_pct": round(float(row["peak_doc_share_pct"]), 4),
            "terms": [
                {"term": item["term"], "weight": round(float(item["term_weight"]), 8)}
                for item in row["term_payload"]
            ],
            "docs": row["representative_docs"],
        }
        for row in topic_rows
    ]
    evolution_payload = {
        "periods": periods,
        "period_totals": [period_totals.get(period, 0) for period in periods],
        "topic_labels": [row["topic_label"] for row in reversed(overview_rows)],
        "heatmap_data": [
            [
                x_index,
                y_index,
                round(float(share_lookup.get(int(row["topic_id"]), {}).get(period, 0.0)), 4),
            ]
            for y_index, row in enumerate(reversed(overview_rows))
            for x_index, period in enumerate(periods)
        ],
    }
    keyword_summary_html = "\n".join(
        f"""      <div class="keyword-card">
        <div class="keyword-name">{item["keyword"]}</div>
        <div class="keyword-top">{item["top_topic_label"]}</div>
        <div class="keyword-detail">Top1 占比 {float(item["top_topic_share_pct"]):.2f}% · Top3 合计 {float(item["top3_share_pct"]):.2f}%</div>
      </div>"""
        for item in keyword_alignment_payload.get("keyword_summary", [])
    )
    quick_links_html = "\n".join(
        f'      <a class="quick-link" href="{item["href"]}" target="_blank" rel="noreferrer">{item["label"]}</a>'
        for item in appendix_links
    )
    topic_options_html = "\n".join(
        f'              <option value="{item["topic_id"]}">{item["label"]}</option>' for item in detail_payload
    )

    script_tags = f'  <script src="{echarts_js_url}"></script>'
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BERTopic Midterm Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f7f8;
      color: #16202a;
    }}
    .page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px 20px 40px;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      padding-bottom: 22px;
      border-bottom: 1px solid #d6dde3;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      font-size: 12px;
      color: #0f766e;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: 34px;
      line-height: 1.1;
    }}
    .lead {{
      margin: 12px 0 0;
      max-width: 960px;
      font-size: 15px;
      line-height: 1.6;
      color: #556372;
    }}
    .metrics {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .metric {{
      padding: 14px 16px;
      border: 1px solid #d6dde3;
      border-radius: 6px;
      background: #ffffff;
    }}
    .metric-label {{
      font-size: 12px;
      color: #607180;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .metric-value {{
      margin-top: 8px;
      font-size: 28px;
      line-height: 1.1;
      font-weight: 700;
      color: #102a43;
    }}
    .metric-detail {{
      margin-top: 6px;
      font-size: 13px;
      line-height: 1.45;
      color: #607180;
    }}
    .keyword-strip {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .keyword-card {{
      padding: 14px 16px;
      border: 1px solid #d6dde3;
      border-radius: 6px;
      background: #ffffff;
    }}
    .keyword-name {{
      font-size: 12px;
      color: #0f766e;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .keyword-top {{
      margin-top: 8px;
      font-size: 16px;
      line-height: 1.4;
      font-weight: 700;
      color: #102a43;
    }}
    .keyword-detail {{
      margin-top: 6px;
      font-size: 13px;
      line-height: 1.45;
      color: #607180;
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
      border: 1px solid #cfd9ef;
      border-radius: 6px;
      background: #ffffff;
      color: #1d4ed8;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
    }}
    .section {{
      padding-top: 22px;
    }}
    .section-title {{
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.2;
    }}
    .section-note {{
      margin: 0 0 14px;
      font-size: 14px;
      line-height: 1.55;
      color: #607180;
      max-width: 960px;
    }}
    .grid-two {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .panel {{
      border: 1px solid #d6dde3;
      border-radius: 6px;
      background: #ffffff;
      padding: 14px;
    }}
    .chart {{
      width: 100%;
      height: 560px;
    }}
    .chart.medium {{
      height: 500px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
      margin-bottom: 12px;
    }}
    .control-group {{
      display: grid;
      gap: 6px;
      min-width: 320px;
    }}
    label {{
      font-size: 13px;
      font-weight: 600;
      color: #425264;
    }}
    select {{
      height: 40px;
      border: 1px solid #c8d2dc;
      border-radius: 6px;
      background: #ffffff;
      padding: 0 10px;
      font-size: 14px;
      color: #16202a;
    }}
    .topic-meta {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      margin-bottom: 12px;
    }}
    .topic-meta-item {{
      padding: 12px;
      border: 1px solid #d6dde3;
      border-radius: 6px;
      background: #f8fafb;
    }}
    .topic-meta-label {{
      font-size: 12px;
      color: #607180;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .topic-meta-value {{
      margin-top: 6px;
      font-size: 18px;
      line-height: 1.25;
      font-weight: 700;
      color: #102a43;
    }}
    .topic-meta-detail {{
      margin-top: 4px;
      font-size: 13px;
      line-height: 1.45;
      color: #607180;
    }}
    .doc-list {{
      margin: 14px 0 0;
      padding-left: 18px;
      color: #425264;
      font-size: 14px;
      line-height: 1.55;
    }}
    .doc-list li + li {{
      margin-top: 8px;
    }}
    @media (max-width: 1080px) {{
      .grid-two {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 900px) {{
      .page {{
        padding: 18px 14px 28px;
      }}
      h1 {{
        font-size: 28px;
      }}
      .chart, .chart.medium {{
        height: 440px;
      }}
      .control-group {{
        min-width: min(100%, 320px);
      }}
    }}
  </style>
{script_tags}
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <p class="eyebrow">BERTopic Midterm Pack</p>
        <h1>躺平 / 摆烂 / 佛系主题结构总览</h1>
        <p class="lead">先回答 BERTopic 自己给出了什么结果，再回到研究问题。中期展示建议优先讲四件事：头部主题有哪些、这些主题在讲什么、三个关键词如何被不同主题吸附、主题在时间上何时变强。</p>
      </div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">总文档量</div>
          <div class="metric-value">{total_docs:,}</div>
          <div class="metric-detail">包含 outlier 与已聚类文档。</div>
        </div>
        <div class="metric">
          <div class="metric-label">已聚类文档</div>
          <div class="metric-value">{(total_docs - outlier_count):,}</div>
          <div class="metric-detail">{(100.0 * (total_docs - outlier_count) / total_docs):.2f}% 进入具体主题。</div>
        </div>
        <div class="metric">
          <div class="metric-label">Outlier 占比</div>
          <div class="metric-value">{(100.0 * outlier_count / total_docs):.2f}%</div>
          <div class="metric-detail">适合在汇报里当作“语料异质性”诊断指标。</div>
        </div>
        <div class="metric">
          <div class="metric-label">头部主题</div>
          <div class="metric-value">Top {top_n_topics}</div>
          <div class="metric-detail">默认先展示规模最大、语义最稳的一层。</div>
        </div>
      </div>
      <div class="keyword-strip">
{keyword_summary_html}
      </div>
      <div class="quick-links">
{quick_links_html}
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">主题规模分布</h2>
      <p class="section-note">这是最适合开场的一张图。它直接回答：哪些 topic 真正构成了 broad 语料的主体，哪些只是零散尾部。</p>
      <div class="panel">
        <div id="prevalence-chart" class="chart medium"></div>
      </div>
    </section>

    <section class="section">
      <div class="grid-two">
        <div>
          <h2 class="section-title">关键词与主题对应关系</h2>
          <p class="section-note">看三个关键词分别被哪些 topic 主导，也能一眼看出“躺平”里存在明显领域漂移，而“佛系”相对更集中。</p>
          <div class="panel">
            <div id="keyword-alignment-chart" class="chart medium"></div>
          </div>
        </div>
        <div>
          <h2 class="section-title">主题时间演化</h2>
          <p class="section-note">看头部主题在各时间期的强弱变化，更适合讲“持续性主题”和“阶段性事件主题”的区别。</p>
          <div class="panel">
            <div id="evolution-chart" class="chart medium"></div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2 class="section-title">主题解释卡片</h2>
      <p class="section-note">中期展示里不要只放词云。这里把高权重词、规模、峰值时间和代表文本放在一起，更容易判断某个 topic 是否真的在讲同一个语义对象。</p>
      <div class="panel">
        <div class="toolbar">
          <div class="control-group">
            <label for="topic-select">选择主题</label>
            <select id="topic-select">
{topic_options_html}
            </select>
          </div>
        </div>
        <div id="topic-meta" class="topic-meta"></div>
        <div id="term-chart" class="chart medium"></div>
        <ul id="doc-list" class="doc-list"></ul>
      </div>
    </section>
  </div>
  <script>
const prevalenceData = {json.dumps(prevalence_payload, ensure_ascii=False)};
const keywordAlignmentData = {json.dumps(keyword_alignment_payload, ensure_ascii=False)};
const evolutionData = {json.dumps(evolution_payload, ensure_ascii=False)};
const topicDetailData = {json.dumps(detail_payload, ensure_ascii=False)};

const prevalenceChart = echarts.init(document.getElementById("prevalence-chart"));
prevalenceChart.setOption({{
  animationDuration: 500,
  grid: {{ left: 210, right: 34, top: 26, bottom: 28 }},
  tooltip: {{
    trigger: "axis",
    axisPointer: {{ type: "shadow" }},
    formatter: (params) => {{
      const row = params[0].data.row;
      return `<strong>${{row.label_full}}</strong><br>` +
        `文档数: ${{row.topic_count.toLocaleString()}}<br>` +
        `占全部文档: ${{row.share_all.toFixed(2)}}%<br>` +
        `占已聚类文档: ${{row.share_clustered.toFixed(2)}}%<br>` +
        `峰值时间: ${{row.peak_period || "NA"}}<br>` +
        `峰值期文档数: ${{row.peak_doc_count.toLocaleString()}}<br>` +
        `峰值期占比: ${{row.peak_doc_share_pct.toFixed(2)}}%<br>` +
        `Top terms: ${{row.top_terms.join(" / ")}}`;
    }}
  }},
  xAxis: {{
    type: "value",
    axisLabel: {{ color: "#425264" }},
    splitLine: {{ lineStyle: {{ color: "#dde4eb" }} }}
  }},
  yAxis: {{
    type: "category",
    axisLabel: {{ color: "#425264", width: 180, overflow: "truncate" }},
    data: prevalenceData.map((item) => item.label)
  }},
  series: [{{
    type: "bar",
    data: prevalenceData.map((item) => ({{
      value: item.topic_count,
      row: item,
      itemStyle: {{
        color: item.share_clustered >= 2.0 ? "#2563eb" : item.share_clustered >= 1.0 ? "#0891b2" : "#0f766e"
      }}
    }})),
    label: {{
      show: true,
      position: "right",
      color: "#425264",
      formatter: (params) => params.data.row.share_clustered.toFixed(2) + "%"
    }},
    barMaxWidth: 28
  }}]
}});

const keywordAlignmentChart = echarts.init(document.getElementById("keyword-alignment-chart"));
keywordAlignmentChart.setOption({{
  animationDuration: 400,
  tooltip: {{
    position: "top",
    formatter: (params) => {{
      const keyword = keywordAlignmentData.keywords[params.data[0]];
      const label = keywordAlignmentData.selected_topic_labels[params.data[1]];
      const share = Number(params.data[2] || 0);
      const docCount = Number(params.data[3] || 0);
      return `<strong>${{label}}</strong><br>` +
        `关键词: ${{keyword}}<br>` +
        `关键词内部占比: ${{share.toFixed(2)}}%<br>` +
        `文档数: ${{docCount.toLocaleString()}}`;
    }}
  }},
  grid: {{ left: 210, right: 24, top: 26, bottom: 64 }},
  xAxis: {{
    type: "category",
    data: keywordAlignmentData.keywords,
    axisLabel: {{ color: "#425264" }},
    axisLine: {{ lineStyle: {{ color: "#c8d2dc" }} }}
  }},
  yAxis: {{
    type: "category",
    data: keywordAlignmentData.selected_topic_labels,
    axisLabel: {{ color: "#425264", width: 180, overflow: "truncate" }},
    axisLine: {{ lineStyle: {{ color: "#c8d2dc" }} }}
  }},
  visualMap: {{
    min: 0,
    max: Math.max(Number(keywordAlignmentData.heatmap_max || 0), 1),
    calculable: true,
    orient: "horizontal",
    left: "center",
    bottom: 14,
    inRange: {{
      color: ["#eef5fb", "#bfdbfe", "#60a5fa", "#2563eb", "#1d4ed8"]
    }}
  }},
  series: [{{
    type: "heatmap",
    data: keywordAlignmentData.heatmap_data,
    label: {{
      show: true,
      color: "#102a43",
      formatter: (params) => Number(params.data[2] || 0).toFixed(1) + "%"
    }}
  }}]
}});

const evolutionChart = echarts.init(document.getElementById("evolution-chart"));
evolutionChart.setOption({{
  animationDuration: 400,
  tooltip: {{
    position: "top",
    formatter: (params) => {{
      const period = evolutionData.periods[params.data[0]];
      const label = evolutionData.topic_labels[params.data[1]];
      const totalDocs = evolutionData.period_totals[params.data[0]];
      return `<strong>${{label}}</strong><br>` +
        `时间: ${{period}}<br>` +
        `topic 占比: ${{Number(params.data[2] || 0).toFixed(2)}}%<br>` +
        `该期已聚类文档: ${{Number(totalDocs).toLocaleString()}}`;
    }}
  }},
  grid: {{ left: 210, right: 24, top: 26, bottom: 88 }},
  xAxis: {{
    type: "category",
    data: evolutionData.periods,
    axisLabel: {{ interval: 0, rotate: 35, color: "#425264" }},
    axisLine: {{ lineStyle: {{ color: "#c8d2dc" }} }}
  }},
  yAxis: {{
    type: "category",
    data: evolutionData.topic_labels,
    axisLabel: {{ color: "#425264", width: 180, overflow: "truncate" }},
    axisLine: {{ lineStyle: {{ color: "#c8d2dc" }} }}
  }},
  visualMap: {{
    min: 0,
    max: Math.max(...evolutionData.heatmap_data.map((item) => Number(item[2] || 0)), 1),
    calculable: true,
    orient: "horizontal",
    left: "center",
    bottom: 20,
    inRange: {{
      color: ["#eef5fb", "#bfdbfe", "#60a5fa", "#2563eb", "#1d4ed8"]
    }}
  }},
  series: [{{
    type: "heatmap",
    data: evolutionData.heatmap_data,
    emphasis: {{
      itemStyle: {{
        shadowBlur: 10,
        shadowColor: "rgba(0, 0, 0, 0.18)"
      }}
    }}
  }}]
}});

const topicSelect = document.getElementById("topic-select");
const topicMeta = document.getElementById("topic-meta");
const docList = document.getElementById("doc-list");
const termChart = echarts.init(document.getElementById("term-chart"));

function renderTopicDetail() {{
  const current = topicDetailData.find((item) => String(item.topic_id) === topicSelect.value) || topicDetailData[0];
  topicMeta.innerHTML = `
    <div class="topic-meta-item">
      <div class="topic-meta-label">主题标签</div>
      <div class="topic-meta-value">${{current.label_full}}</div>
      <div class="topic-meta-detail">按 Count 排序的稳定主题。</div>
    </div>
    <div class="topic-meta-item">
      <div class="topic-meta-label">文档数</div>
      <div class="topic-meta-value">${{Number(current.topic_count).toLocaleString()}}</div>
      <div class="topic-meta-detail">占全部文档 ${{Number(current.share_all).toFixed(2)}}%，占已聚类文档 ${{Number(current.share_clustered).toFixed(2)}}%。</div>
    </div>
    <div class="topic-meta-item">
      <div class="topic-meta-label">峰值时间</div>
      <div class="topic-meta-value">${{current.peak_period || "NA"}}</div>
      <div class="topic-meta-detail">峰值文档数 ${{Number(current.peak_doc_count).toLocaleString()}}, 该期占比 ${{Number(current.peak_doc_share_pct).toFixed(2)}}%。</div>
    </div>
  `;

  termChart.setOption({{
    animationDuration: 400,
    grid: {{ left: 72, right: 24, top: 24, bottom: 78 }},
    tooltip: {{
      trigger: "axis",
      axisPointer: {{ type: "shadow" }},
      formatter: (params) => {{
        const row = params[0].data.row;
        return `<strong>${{row.term}}</strong><br>term weight: ${{Number(row.weight).toFixed(6)}}`;
      }}
    }},
    xAxis: {{
      type: "category",
      data: current.terms.map((item) => item.term),
      axisLabel: {{ interval: 0, rotate: 35, color: "#425264" }},
      axisLine: {{ lineStyle: {{ color: "#c8d2dc" }} }}
    }},
    yAxis: {{
      type: "value",
      axisLabel: {{ color: "#425264" }},
      splitLine: {{ lineStyle: {{ color: "#dde4eb" }} }}
    }},
    series: [{{
      type: "bar",
      data: current.terms.map((item) => ({{
        value: item.weight,
        row: item,
        itemStyle: {{ color: "#2563eb" }}
      }})),
      barMaxWidth: 30
    }}]
  }});

  const docs = current.docs.length ? current.docs : ["没有可用的 representative docs。"];
  docList.innerHTML = docs.map((doc) => `<li>${{doc}}</li>`).join("");
}}

topicSelect.addEventListener("change", renderTopicDetail);
window.addEventListener("resize", () => {{
  prevalenceChart.resize();
  keywordAlignmentChart.resize();
  evolutionChart.resize();
  termChart.resize();
}});
renderTopicDetail();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    emit = resolve_emit("topic-interpret", None)
    total_start = time.perf_counter()

    output_dir = Path(args.output_dir)
    html_dir = output_dir / "html"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    emit("Loading topic model outputs")
    topic_info = pd.read_csv(args.topic_info_path)
    topic_terms = pd.read_csv(args.topic_terms_path)
    share_by_period = pd.read_csv(args.topic_share_by_period_path)
    share_by_period_keyword = pd.read_csv(args.topic_share_by_period_keyword_path)

    topic_terms_lookup = build_topic_term_lookup(topic_terms)
    topic_rows, total_docs, outlier_count = build_topic_rows(
        topic_info,
        topic_terms_lookup,
        share_by_period,
        top_n_terms=args.top_n_terms,
    )
    if not topic_rows:
        raise ValueError("No non-outlier topics found in topic_info.csv.")

    overview_rows = topic_rows[: args.top_n_topics]
    wordcloud_rows = topic_rows[: args.wordcloud_top_n_topics]
    top_topic_ids = [int(row["topic_id"]) for row in overview_rows]
    periods, period_totals, share_lookup = build_period_payload(
        share_by_period,
        top_topic_ids=top_topic_ids,
        min_period_docs=args.min_period_docs,
    )
    keyword_alignment_payload = build_keyword_alignment_payload(
        share_by_period_keyword,
        topic_rows=topic_rows,
        selected_topic_ids=top_topic_ids,
    )
    overview_table, coding_template = build_topic_tables(
        topic_rows,
        keyword_alignment_payload=keyword_alignment_payload,
        coding_template_top_n=args.coding_template_top_n,
    )

    emit(
        "Prepared interpretability payload "
        f"(topics={len(topic_rows)}, overview={len(overview_rows)}, "
        f"wordcloud={len(wordcloud_rows)}, periods={len(periods)})"
    )

    appendix_links = [
        {"label": "主题规模附图", "href": "html/topic_prevalence.html"},
        {"label": "关键词对应附图", "href": "html/topic_keyword_alignment.html"},
        {"label": "时间演化附图", "href": "html/topic_evolution_heatmap.html"},
        {"label": "主题解释附图", "href": "html/topic_term_detail.html"},
        {"label": "主题词云附录", "href": "html/topic_wordclouds_appendix.html"},
        {"label": "主题总表", "href": "tables/topic_overview_table.csv"},
        {"label": "人工编码模板", "href": "tables/topic_coding_template.csv"},
    ]
    dashboard_html = render_midterm_dashboard_html(
        topic_rows,
        periods,
        period_totals,
        share_lookup,
        keyword_alignment_payload,
        total_docs=total_docs,
        outlier_count=outlier_count,
        top_n_topics=args.top_n_topics,
        echarts_js_url=args.echarts_js_url,
        appendix_links=appendix_links,
    )
    prevalence_html = render_prevalence_html(
        topic_rows,
        total_docs=total_docs,
        outlier_count=outlier_count,
        top_n_topics=args.top_n_topics,
        echarts_js_url=args.echarts_js_url,
    )
    keyword_alignment_html = render_keyword_alignment_html(
        keyword_alignment_payload,
        echarts_js_url=args.echarts_js_url,
    )
    detail_html = render_term_detail_html(topic_rows, echarts_js_url=args.echarts_js_url)
    wordcloud_html = render_wordcloud_html(
        wordcloud_rows,
        echarts_js_url=args.echarts_js_url,
        echarts_wordcloud_js_url=args.echarts_wordcloud_js_url,
    )
    evolution_html = render_evolution_html(
        overview_rows,
        periods,
        period_totals,
        share_lookup,
        echarts_js_url=args.echarts_js_url,
    )

    dashboard_path = output_dir / "topic_midterm_dashboard.html"
    prevalence_path = html_dir / "topic_prevalence.html"
    keyword_alignment_path = html_dir / "topic_keyword_alignment.html"
    detail_path = html_dir / "topic_term_detail.html"
    wordcloud_path = html_dir / "topic_wordclouds_appendix.html"
    evolution_path = html_dir / "topic_evolution_heatmap.html"
    overview_table_path = tables_dir / "topic_overview_table.csv"
    keyword_matrix_path = tables_dir / "keyword_topic_matrix.csv"
    coding_template_path = tables_dir / "topic_coding_template.csv"
    summary_path = output_dir / "topic_visualization_summary.json"

    emit("Saving topic-visualization outputs")
    dashboard_path.write_text(dashboard_html, encoding="utf-8")
    prevalence_path.write_text(prevalence_html, encoding="utf-8")
    keyword_alignment_path.write_text(keyword_alignment_html, encoding="utf-8")
    detail_path.write_text(detail_html, encoding="utf-8")
    wordcloud_path.write_text(wordcloud_html, encoding="utf-8")
    evolution_path.write_text(evolution_html, encoding="utf-8")

    overview_table.to_csv(overview_table_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(keyword_alignment_payload.get("table_rows", [])).to_csv(
        keyword_matrix_path,
        index=False,
        encoding="utf-8-sig",
    )
    coding_template.to_csv(coding_template_path, index=False, encoding="utf-8-sig")

    save_json(
        summary_path,
        {
            "topic_info_path": str(Path(args.topic_info_path).resolve()),
            "topic_terms_path": str(Path(args.topic_terms_path).resolve()),
            "topic_share_by_period_path": str(Path(args.topic_share_by_period_path).resolve()),
            "topic_share_by_period_and_keyword_path": str(Path(args.topic_share_by_period_keyword_path).resolve()),
            "output_dir": str(output_dir.resolve()),
            "total_docs": int(total_docs),
            "outlier_count": int(outlier_count),
            "outlier_share": float(outlier_count / total_docs) if total_docs else 0.0,
            "clustered_doc_count": int(total_docs - outlier_count),
            "top_n_topics": int(args.top_n_topics),
            "wordcloud_top_n_topics": int(args.wordcloud_top_n_topics),
            "top_n_terms": int(args.top_n_terms),
            "min_period_docs": int(args.min_period_docs),
            "coding_template_top_n": int(args.coding_template_top_n),
            "retained_periods": periods,
            "dashboard_path": str(dashboard_path.resolve()),
            "prevalence_path": str(prevalence_path.resolve()),
            "keyword_alignment_path": str(keyword_alignment_path.resolve()),
            "detail_path": str(detail_path.resolve()),
            "wordcloud_path": str(wordcloud_path.resolve()),
            "evolution_path": str(evolution_path.resolve()),
            "overview_table_path": str(overview_table_path.resolve()),
            "keyword_matrix_path": str(keyword_matrix_path.resolve()),
            "coding_template_path": str(coding_template_path.resolve()),
            "echarts_js_url": args.echarts_js_url,
            "echarts_wordcloud_js_url": args.echarts_wordcloud_js_url,
        },
    )
    emit(f"Saved topic-visualization bundle to {output_dir}")
    emit(f"Total runtime: {format_elapsed(total_start)}")


if __name__ == "__main__":
    main()
