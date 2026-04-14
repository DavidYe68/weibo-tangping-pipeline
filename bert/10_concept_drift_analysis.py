#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lib.analysis_utils import (
    DEFAULT_ANALYSIS_KEYWORDS,
    js_divergence,
    normalize_cli_keywords,
    save_dataframe,
    sort_period_labels,
)
from lib.io_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure collocation drift, semantic-neighbor drift, and topic drift."
    )
    parser.add_argument(
        "--cooccurrence_path",
        default="bert/artifacts/broad_analysis/semantic_analysis/keyword_cooccurrence.csv",
        help="Path to keyword co-occurrence output.",
    )
    parser.add_argument(
        "--neighbor_path",
        default="bert/artifacts/broad_analysis/semantic_analysis/keyword_semantic_neighbors.csv",
        help="Path to keyword semantic-neighbor output.",
    )
    parser.add_argument(
        "--topic_share_path",
        default="bert/artifacts/broad_analysis/topic_model/topic_share_by_period_and_keyword.csv",
        help="Path to keyword-by-period topic share output.",
    )
    parser.add_argument(
        "--overall_topic_share_path",
        default="bert/artifacts/broad_analysis/topic_model/topic_share_by_period.csv",
        help="Path to overall topic-share output.",
    )
    parser.add_argument(
        "--output_dir",
        default="bert/artifacts/broad_analysis/drift_analysis",
        help="Directory for drift-analysis outputs.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=list(DEFAULT_ANALYSIS_KEYWORDS),
        help="Canonical keywords to keep. Defaults to 躺平 摆烂 佛系.",
    )
    parser.add_argument(
        "--time_granularity",
        choices=["month", "quarter", "year"],
        default="month",
        help="Granularity used when sorting period labels.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="How many top terms/neighbors to compare per period.",
    )
    return parser.parse_args()


def load_csv_if_exists(path: str) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved)


def adjacent_pairs(labels: list[str]) -> list[tuple[str, str]]:
    return [(labels[index], labels[index + 1]) for index in range(len(labels) - 1)]


def compare_ranked_terms(
    df: pd.DataFrame,
    *,
    keyword_col: str,
    period_col: str,
    term_col: str,
    score_col: str,
    selected_keywords: list[str],
    top_n: int,
    time_granularity: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if df.empty:
        return pd.DataFrame(rows)

    filtered = df[df[keyword_col].isin(selected_keywords)].copy()
    filtered = filtered[filtered[period_col] != "ALL"].copy()
    for keyword in selected_keywords:
        keyword_df = filtered[filtered[keyword_col] == keyword].copy()
        if keyword_df.empty:
            continue

        ordered_periods = sort_period_labels(keyword_df[period_col].astype(str).unique().tolist(), time_granularity)
        for previous_period, current_period in adjacent_pairs(ordered_periods):
            previous = keyword_df[keyword_df[period_col] == previous_period].nlargest(top_n, score_col)
            current = keyword_df[keyword_df[period_col] == current_period].nlargest(top_n, score_col)
            previous_terms = previous[term_col].astype(str).tolist()
            current_terms = current[term_col].astype(str).tolist()
            if not previous_terms and not current_terms:
                continue

            union_terms = sorted(set(previous_terms) | set(current_terms))
            previous_scores = []
            current_scores = []
            previous_map = previous.set_index(term_col)[score_col].to_dict()
            current_map = current.set_index(term_col)[score_col].to_dict()
            for term in union_terms:
                previous_scores.append(float(previous_map.get(term, 0.0)))
                current_scores.append(float(current_map.get(term, 0.0)))

            overlap = sorted(set(previous_terms) & set(current_terms))
            added = sorted(set(current_terms) - set(previous_terms))
            removed = sorted(set(previous_terms) - set(current_terms))
            denominator = len(set(previous_terms) | set(current_terms))
            jaccard = len(overlap) / denominator if denominator else float("nan")

            rows.append(
                {
                    "keyword": keyword,
                    "previous_period": previous_period,
                    "current_period": current_period,
                    "overlap_count": len(overlap),
                    "jaccard_top_terms": float(jaccard),
                    "js_divergence": float(js_divergence(previous_scores, current_scores)),
                    "overlap_terms": " | ".join(overlap),
                    "added_terms": " | ".join(added),
                    "removed_terms": " | ".join(removed),
                }
            )

    return pd.DataFrame(rows)


def compare_topic_shares(
    df: pd.DataFrame,
    *,
    keyword_col: str | None,
    period_col: str,
    topic_col: str,
    share_col: str,
    selected_keywords: list[str],
    time_granularity: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drift_rows: list[dict[str, object]] = []
    change_rows: list[dict[str, object]] = []
    if df.empty:
        return pd.DataFrame(drift_rows), pd.DataFrame(change_rows)

    if keyword_col:
        grouped_items = [(keyword, df[df[keyword_col] == keyword].copy()) for keyword in selected_keywords]
    else:
        grouped_items = [("ALL", df.copy())]

    for group_key, group_df in grouped_items:
        if group_df.empty:
            continue
        ordered_periods = sort_period_labels(group_df[period_col].astype(str).unique().tolist(), time_granularity)
        for previous_period, current_period in adjacent_pairs(ordered_periods):
            previous = group_df[group_df[period_col] == previous_period].copy()
            current = group_df[group_df[period_col] == current_period].copy()
            if previous.empty and current.empty:
                continue

            topics = sorted(set(previous[topic_col].tolist()) | set(current[topic_col].tolist()))
            previous_map = previous.set_index(topic_col)[share_col].to_dict()
            current_map = current.set_index(topic_col)[share_col].to_dict()
            previous_vector = [float(previous_map.get(topic, 0.0)) for topic in topics]
            current_vector = [float(current_map.get(topic, 0.0)) for topic in topics]

            drift_rows.append(
                {
                    "keyword": group_key,
                    "previous_period": previous_period,
                    "current_period": current_period,
                    "topic_js_divergence": float(js_divergence(previous_vector, current_vector)),
                    "topic_count": len(topics),
                }
            )

            for topic in topics:
                change_rows.append(
                    {
                        "keyword": group_key,
                        "previous_period": previous_period,
                        "current_period": current_period,
                        "topic_id": topic,
                        "previous_share": float(previous_map.get(topic, 0.0)),
                        "current_share": float(current_map.get(topic, 0.0)),
                        "share_delta": float(current_map.get(topic, 0.0) - previous_map.get(topic, 0.0)),
                    }
                )

    change_df = pd.DataFrame(change_rows)
    if not change_df.empty:
        change_df = change_df.sort_values(
            ["keyword", "previous_period", "share_delta"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
    return pd.DataFrame(drift_rows), change_df


def main() -> None:
    args = parse_args()
    selected_keywords = normalize_cli_keywords(args.keywords)

    cooccurrence_df = load_csv_if_exists(args.cooccurrence_path)
    neighbor_df = load_csv_if_exists(args.neighbor_path)
    topic_share_df = load_csv_if_exists(args.topic_share_path)
    overall_topic_share_df = load_csv_if_exists(args.overall_topic_share_path)

    collocation_drift_df = compare_ranked_terms(
        cooccurrence_df,
        keyword_col="keyword",
        period_col="period_label",
        term_col="term",
        score_col="pmi",
        selected_keywords=selected_keywords,
        top_n=args.top_n,
        time_granularity=args.time_granularity,
    )
    neighbor_drift_df = compare_ranked_terms(
        neighbor_df,
        keyword_col="keyword",
        period_col="period_label",
        term_col="neighbor_term",
        score_col="embedding_similarity",
        selected_keywords=selected_keywords,
        top_n=args.top_n,
        time_granularity=args.time_granularity,
    )

    topic_drift_df, topic_change_df = compare_topic_shares(
        topic_share_df,
        keyword_col="keyword_normalized" if "keyword_normalized" in topic_share_df.columns else "keyword",
        period_col="period_label",
        topic_col="topic_id",
        share_col="doc_share",
        selected_keywords=selected_keywords,
        time_granularity=args.time_granularity,
    )
    overall_topic_drift_df, overall_topic_change_df = compare_topic_shares(
        overall_topic_share_df,
        keyword_col=None,
        period_col="period_label",
        topic_col="topic_id",
        share_col="doc_share",
        selected_keywords=selected_keywords,
        time_granularity=args.time_granularity,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collocation_path = output_dir / "keyword_collocation_drift.csv"
    neighbor_path = output_dir / "keyword_neighbor_drift.csv"
    topic_path = output_dir / "topic_drift_by_keyword.csv"
    topic_change_path = output_dir / "topic_share_change_by_keyword.csv"
    overall_topic_path = output_dir / "topic_drift_overall.csv"
    overall_topic_change_path = output_dir / "topic_share_change_overall.csv"
    summary_path = output_dir / "drift_analysis_summary.json"

    save_dataframe(collocation_drift_df, collocation_path)
    save_dataframe(neighbor_drift_df, neighbor_path)
    save_dataframe(topic_drift_df, topic_path)
    save_dataframe(topic_change_df, topic_change_path)
    save_dataframe(overall_topic_drift_df, overall_topic_path)
    save_dataframe(overall_topic_change_df, overall_topic_change_path)

    summary = {
        "selected_keywords": selected_keywords,
        "time_granularity": args.time_granularity,
        "collocation_drift_path": str(collocation_path.resolve()),
        "neighbor_drift_path": str(neighbor_path.resolve()),
        "topic_drift_path": str(topic_path.resolve()),
        "topic_share_change_path": str(topic_change_path.resolve()),
        "overall_topic_drift_path": str(overall_topic_path.resolve()),
        "overall_topic_change_path": str(overall_topic_change_path.resolve()),
    }
    save_json(summary_path, summary)


if __name__ == "__main__":
    main()
