#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from lib.io_utils import save_json

DEFAULT_SEMANTIC_DIR = "bert/artifacts/broad_analysis/semantic_analysis"
DEFAULT_NOISE_TERMS_PATH = "bert/config/semantic_midterm_noise_terms.txt"
PURE_ASCII_TERM_RE = re.compile(r"^[A-Za-z0-9._-]+$")
HAS_ASCII_RE = re.compile(r"[A-Za-z]")
HAS_DIGIT_RE = re.compile(r"\d")
NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
NOISE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("platform_recruitment", re.compile(r"(进团|代招|招代招|回官博|官博|单秒|男店)")),
    ("keyword_self_variant", re.compile(r"(佛系|摆烂|躺平)")),
)
THEME_BUCKET_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("金融/市场", ("利息", "证券", "保监", "董事", "etf", "基金", "股", "央行", "减持", "质押")),
    ("工作/学习", ("工作", "上班", "换岗", "岗位", "学校", "考试", "作业", "老师", "同学")),
    ("身体/健康", ("体脂", "感冒", "腰肌", "锻炼", "免疫力", "身体", "休养", "运动")),
    ("家庭/关系", ("爸妈", "家人", "朋友", "同学", "家庭", "老人", "葬礼", "过世")),
    ("平台/社群招募", ("进团", "代招", "官博", "男店", "单秒", "特收", "帮团")),
    ("自我态度/情绪", ("焦虑", "幸福", "快乐", "满足", "摆烂", "佛系", "躺平")),
)
CONTEXT_BUCKET_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("粉圈/交易/超话", ("超话", "周边", "应援", "拼车", "卡", "爱豆", "追星", "中转站", "收一张", "佛系收", "佛系出")),
    ("生活态度/心态", ("看淡", "独处", "心态", "人生", "幸福", "快乐", "满足", "看破", "焦虑", "自爱", "放下")),
    ("工作/职场/学业", ("工作", "上班", "领导", "岗位", "学校", "考试", "作业", "事业", "换岗", "同事")),
    ("财经/投资", ("基金", "证券", "财务自由", "利息", "降准", "降息", "加仓", "大盘", "etf", "股")),
    ("社群招募/平台劳动", ("招新", "代招", "陪玩", "团", "审核", "流水", "应聘", "日入", "跳单", "俱乐部")),
    ("维权/社会事件", ("维权", "监管部门", "住建局", "保险", "撞伤", "全责", "商铺", "交房", "部门", "投诉")),
    ("家庭/健康/关系", ("爸妈", "家人", "家庭", "老人", "体脂", "锻炼", "身体", "休养", "葬礼", "过世")),
    ("游戏/二次元", ("第五人格", "原神", "光遇", "玩家", "战队", "地窖", "谷", "吧唧", "抽卡", "游戏")),
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
        help="Output directory for the midterm bundle. Defaults to <semantic_dir>/midterm_bundle.",
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
        "--top_n_all",
        type=int,
        default=15,
        help="Top cleaned ALL-period terms to keep per keyword.",
    )
    parser.add_argument(
        "--top_n_period",
        type=int,
        default=5,
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


def resolve_tokenized_analysis_base_path(summary: dict, semantic_dir: Path) -> Path:
    candidates: list[Path] = []
    raw_path = summary.get("tokenized_analysis_base_path")
    if raw_path:
        candidate = Path(str(raw_path))
        if not candidate.is_absolute():
            candidate = (semantic_dir / candidate).resolve()
        candidates.append(candidate)
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


def infer_theme_bucket(term: str, reasons: list[str]) -> str:
    if "keyword_self_variant" in reasons:
        return "关键词自变体"
    if any(reason in {"ascii_or_id_term", "alpha_numeric_noise", "custom_noise_term"} for reason in reasons):
        return "账号/代码噪声"
    lowered = term.lower()
    for bucket, markers in THEME_BUCKET_RULES:
        if any(marker.lower() in lowered for marker in markers):
            return bucket
    return "待人工判断"


def classify_term(term: str, keyword: str, exact_noise_terms: set[str]) -> tuple[list[str], str]:
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
    theme_bucket = infer_theme_bucket(term, reasons)
    return sorted(set(reasons)), theme_bucket


def compute_midterm_score(row: pd.Series) -> float:
    similarity = float(row["embedding_similarity"]) if pd.notna(row["embedding_similarity"]) else 0.0
    return math.log1p(float(row["term_doc_freq"])) * float(row["lift"]) * (1.0 + max(similarity, 0.0))


def prepare_candidate_frame(
    cooccurrence_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    selected_keywords: list[str],
    exact_noise_terms: set[str],
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
        lambda row: classify_term(str(row["term"]), str(row["keyword"]), exact_noise_terms),
        axis=1,
    )
    merged["auto_drop_reasons"] = reason_payload.map(lambda item: " | ".join(item[0]))
    merged["auto_theme_bucket"] = reason_payload.map(lambda item: item[1])
    merged["auto_noise_flag"] = merged["auto_drop_reasons"].ne("")
    merged["semantic_supported"] = merged["embedding_similarity"].notna()
    merged["midterm_score"] = merged.apply(compute_midterm_score, axis=1)
    merged["auto_keep_for_midterm"] = ~merged["auto_noise_flag"]
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
                "lead_theme_buckets": " / ".join(ordered["auto_theme_bucket"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["keyword", "period_label"]).reset_index(drop=True)


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
    for row in tokenized_base.itertuples(index=False, name=None):
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


def infer_context_bucket(text: object) -> str:
    lowered = str(text or "").lower()
    for bucket, markers in CONTEXT_BUCKET_RULES:
        if any(marker.lower() in lowered for marker in markers):
            return bucket
    return "待人工判断"


def attach_context_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    result = df.copy()
    result["auto_context_bucket"] = result["example_1_text"].map(infer_context_bucket)
    return result


def build_context_bucket_summary(overall_shortlist: pd.DataFrame) -> pd.DataFrame:
    if overall_shortlist.empty:
        return pd.DataFrame(columns=["keyword", "auto_context_bucket", "term_count", "example_terms"])
    grouped = (
        overall_shortlist.groupby(["keyword", "auto_context_bucket"], dropna=False)["term"]
        .agg(["count", lambda values: " / ".join(list(values)[:5])])
        .reset_index()
        .rename(columns={"count": "term_count", "<lambda_0>": "example_terms"})
        .sort_values(["keyword", "term_count", "auto_context_bucket"], ascending=[True, False, True])
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
                "manual_theme_bucket",
                "manual_keep_for_midterm",
                "manual_quote_candidate",
                "manual_note",
            ]
        )
    template = template.drop_duplicates(subset=["keyword", "period_label", "term"]).copy()
    template["manual_theme_bucket"] = ""
    template["manual_keep_for_midterm"] = ""
    template["manual_quote_candidate"] = ""
    template["manual_note"] = ""
    return template.reset_index(drop=True)


def render_markdown_summary(
    *,
    candidates: pd.DataFrame,
    overall_shortlist: pd.DataFrame,
    period_overview: pd.DataFrame,
    noise_diagnostics: pd.DataFrame,
    context_bucket_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = ["# 09 Semantic Midterm Notes", ""]
    lines.append(
        f"- 原始候选词行数：{len(candidates)}；自动保留用于中期整理的候选词行数：{int(candidates['auto_keep_for_midterm'].sum())}。"
    )
    lines.append("- 这一版建议把 `keyword_cooccurrence.csv` 视为候选词池，把本目录下的清洗 shortlist 视为汇报入口。")
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
            for bucket, count in zip(bucket_subset["auto_context_bucket"], bucket_subset["term_count"])
        )
        lead_terms = " / ".join(subset["term"].head(5).tolist())
        lines.append(f"- `{keyword}`: 主要上下文桶 = {bucket_text}；代表词 = {lead_terms}")
    lines.append("")

    lines.append("## 怎么讲 09")
    lines.append("- 先讲 `semantic_keyword_overview.csv`：每个关键词总体上最值得进入中期报告的词和自动主题桶。")
    lines.append("- 再讲 `semantic_period_overview.csv`：每个月该关键词的 lead terms。")
    lines.append("- 最后在 `semantic_midterm_coding_template.csv` 里人工勾选保留项，并用 example_text 回原文核对。")
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    semantic_dir = Path(args.semantic_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = semantic_dir / "midterm_bundle"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = OperationLogger(output_dir / "semantic_midterm_operation_log.md")
    logger.log(f"Start 09 midterm preparation from {semantic_dir}")

    cooccurrence_path = semantic_dir / "keyword_cooccurrence.csv"
    neighbors_path = semantic_dir / "keyword_semantic_neighbors.csv"
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
    candidates = prepare_candidate_frame(cooccurrence_df, neighbor_df, selected_keywords, exact_noise_terms)
    save_dataframe(candidates, output_dir / "semantic_midterm_candidates.csv")

    logger.log("Select report-ready shortlists for overall keywords and per-period reading")
    overall_shortlist, period_shortlist = rank_shortlists(
        candidates,
        top_n_all=args.top_n_all,
        top_n_period=args.top_n_period,
        min_doc_freq_all=args.min_doc_freq_all,
        min_doc_freq_period=args.min_doc_freq_period,
    )
    period_overview = build_period_overview(period_shortlist)
    noise_diagnostics = build_noise_diagnostics(candidates)
    overall_candidates = candidates[
        (candidates["period_label"] == "ALL")
        & candidates["auto_keep_for_midterm"]
        & (candidates["term_doc_freq"] >= args.min_doc_freq_all)
    ].copy()

    tokenized_path = resolve_tokenized_analysis_base_path(summary, semantic_dir)
    logger.log(f"Load tokenized analysis base from {tokenized_path}")
    tokenized_base = pd.read_parquet(
        tokenized_path,
        columns=[
            "keyword_normalized",
            "period_label",
            "analysis_text",
            "tokens",
            "点赞数",
            "评论数",
            "转发数",
        ],
    )

    requested_rows = pd.concat([overall_candidates, period_shortlist], ignore_index=True)
    example_lookup, stats_lookup = build_example_lookup(
        tokenized_base,
        requested_rows,
        example_count=args.example_count,
        logger=logger,
    )
    overall_candidates = attach_match_stats(overall_candidates, stats_lookup)
    overall_shortlist = rerank_overall_with_diversity(overall_candidates, top_n_all=args.top_n_all)
    period_shortlist = attach_match_stats(period_shortlist, stats_lookup)
    overall_shortlist = attach_examples(overall_shortlist, example_lookup, args.example_count)
    period_shortlist = attach_examples(period_shortlist, example_lookup, args.example_count)
    coding_template = build_coding_template(overall_shortlist, period_shortlist)

    overall_shortlist = attach_context_buckets(overall_shortlist)
    period_shortlist = attach_context_buckets(period_shortlist)
    context_bucket_summary = build_context_bucket_summary(overall_shortlist)
    coding_template = attach_context_buckets(coding_template)

    logger.log("Write report-facing CSV/Markdown/JSON bundle for 09")
    save_dataframe(overall_shortlist, output_dir / "semantic_keyword_overview.csv")
    save_dataframe(period_shortlist, output_dir / "semantic_period_shortlist.csv")
    save_dataframe(period_overview, output_dir / "semantic_period_overview.csv")
    save_dataframe(noise_diagnostics, output_dir / "semantic_noise_diagnostics.csv")
    save_dataframe(coding_template, output_dir / "semantic_midterm_coding_template.csv")

    render_markdown_summary(
        candidates=candidates,
        overall_shortlist=overall_shortlist,
        period_overview=period_overview,
        noise_diagnostics=noise_diagnostics,
        context_bucket_summary=context_bucket_summary,
        output_path=output_dir / "semantic_midterm_notes.md",
    )

    summary_payload = {
        "semantic_dir": str(semantic_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selected_keywords": selected_keywords,
        "candidate_row_count": int(len(candidates)),
        "auto_keep_row_count": int(candidates["auto_keep_for_midterm"].sum()),
        "overall_shortlist_row_count": int(len(overall_shortlist)),
        "period_shortlist_row_count": int(len(period_shortlist)),
        "noise_diagnostic_row_count": int(len(noise_diagnostics)),
        "context_bucket_summary_row_count": int(len(context_bucket_summary)),
        "coding_template_row_count": int(len(coding_template)),
        "exact_noise_term_count": int(len(exact_noise_terms)),
    }
    save_json(output_dir / "semantic_midterm_summary.json", summary_payload)
    save_dataframe(context_bucket_summary, output_dir / "semantic_context_bucket_summary.csv")
    logger.log(
        "Finished 09 midterm preparation "
        f"(overall_shortlist={len(overall_shortlist)}, period_shortlist={len(period_shortlist)})"
    )
    logger.save()


if __name__ == "__main__":
    main()
