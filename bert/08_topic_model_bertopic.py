#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lib.analysis_utils import (
    DEFAULT_ANALYSIS_KEYWORDS,
    flatten_topic_terms,
    load_tabular_files,
    normalize_cli_keywords,
    period_column_name,
    resolve_emit,
    save_dataframe,
    sort_period_labels,
)
from lib.io_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BERTopic on the normalized broad-analysis base table."
    )
    parser.add_argument(
        "--input_path",
        default="bert/artifacts/broad_analysis/analysis_base.parquet",
        help="Path or glob pattern for the analysis base table.",
    )
    parser.add_argument(
        "--output_dir",
        default="bert/artifacts/broad_analysis/topic_model",
        help="Directory for BERTopic outputs.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=list(DEFAULT_ANALYSIS_KEYWORDS),
        help="Canonical keywords to keep. Defaults to 躺平 摆烂 佛系.",
    )
    parser.add_argument("--text_col", default="analysis_text", help="Text column name.")
    parser.add_argument("--keyword_col", default="keyword_normalized", help="Keyword column name.")
    parser.add_argument("--time_col", default="publish_time", help="Timestamp column name.")
    parser.add_argument(
        "--time_granularity",
        choices=["month", "quarter", "year"],
        default="month",
        help="Time granularity for topic-share outputs.",
    )
    parser.add_argument("--min_topic_size", type=int, default=30, help="BERTopic min_topic_size.")
    parser.add_argument("--top_n_words", type=int, default=10, help="Top words per topic.")
    parser.add_argument("--nr_topics", default=None, help="Optional BERTopic nr_topics value.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load embedding artifacts from local files.",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Persist the BERTopic model under output_dir/model.",
    )
    return parser.parse_args()


def load_bertopic_model(args: argparse.Namespace):
    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise ImportError(
            "BERTopic is not installed. Please install requirements and retry."
        ) from exc

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. Please install requirements and retry."
        ) from exc

    embedding_model = SentenceTransformer(
        args.embedding_model,
        local_files_only=bool(args.local_files_only),
    )
    return BERTopic(
        embedding_model=embedding_model,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        nr_topics=args.nr_topics,
        calculate_probabilities=True,
        verbose=False,
    )


def main() -> None:
    args = parse_args()
    emit = resolve_emit("bertopic", None)

    df, _ = load_tabular_files(args.input_path, emit=emit)
    selected_keywords = normalize_cli_keywords(args.keywords)
    if args.keyword_col not in df.columns:
        raise ValueError(f"Keyword column '{args.keyword_col}' not found in analysis base.")
    if args.text_col not in df.columns:
        raise ValueError(f"Text column '{args.text_col}' not found in analysis base.")
    if args.time_col not in df.columns:
        raise ValueError(f"Time column '{args.time_col}' not found in analysis base.")

    filtered = df[df[args.keyword_col].isin(selected_keywords)].copy()
    filtered[args.text_col] = filtered[args.text_col].fillna("").astype("string").str.strip()
    filtered = filtered[filtered[args.text_col].ne("")].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No rows left after keyword filtering.")

    filtered["period_label"] = pd.to_datetime(filtered[args.time_col], errors="coerce")
    filtered["period_label"] = filtered["period_label"].dt.to_period(
        {"month": "M", "quarter": "Q", "year": "Y"}[args.time_granularity]
    ).astype("string")

    texts = filtered[args.text_col].tolist()
    model = load_bertopic_model(args)
    emit(f"Fitting BERTopic on {len(texts)} documents")
    topics, probabilities = model.fit_transform(texts)

    doc_topics = filtered.copy()
    doc_topics["topic_id"] = topics
    if probabilities is not None:
        topic_probabilities = []
        for row_index, topic_id in enumerate(topics):
            if topic_id < 0:
                topic_probabilities.append(float("nan"))
                continue
            topic_probabilities.append(float(probabilities[row_index][topic_id]))
        doc_topics["topic_probability"] = topic_probabilities
    else:
        doc_topics["topic_probability"] = float("nan")

    topic_info = model.get_topic_info()
    topic_labels = {}
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        label = row.get("Name")
        if pd.isna(label):
            label = f"Topic {topic_id}"
        topic_labels[topic_id] = str(label)

    doc_topics["topic_label"] = doc_topics["topic_id"].map(topic_labels).fillna("Outlier")

    topic_terms = {}
    for topic_id in topic_info["Topic"].tolist():
        topic_id_int = int(topic_id)
        if topic_id_int < 0:
            continue
        topic_terms[topic_id_int] = model.get_topic(topic_id_int) or []

    topic_terms_df = flatten_topic_terms(topic_terms)

    period_col = "period_label"
    share_by_period = (
        doc_topics[doc_topics["topic_id"] >= 0]
        .groupby([period_col, "topic_id", "topic_label"], dropna=False)
        .size()
        .reset_index(name="doc_count")
    )
    if not share_by_period.empty:
        period_totals = share_by_period.groupby(period_col)["doc_count"].transform("sum")
        share_by_period["doc_share"] = share_by_period["doc_count"] / period_totals
        ordered_periods = sort_period_labels(share_by_period[period_col].astype(str).tolist(), args.time_granularity)
        share_by_period[period_col] = pd.Categorical(
            share_by_period[period_col],
            categories=ordered_periods,
            ordered=True,
        )
        share_by_period = share_by_period.sort_values([period_col, "doc_share", "topic_id"], ascending=[True, False, True])

    share_by_period_and_keyword = (
        doc_topics[doc_topics["topic_id"] >= 0]
        .groupby([args.keyword_col, period_col, "topic_id", "topic_label"], dropna=False)
        .size()
        .reset_index(name="doc_count")
    )
    if not share_by_period_and_keyword.empty:
        grouped_totals = share_by_period_and_keyword.groupby([args.keyword_col, period_col])["doc_count"].transform("sum")
        share_by_period_and_keyword["doc_share"] = share_by_period_and_keyword["doc_count"] / grouped_totals
        ordered_periods = sort_period_labels(
            share_by_period_and_keyword[period_col].astype(str).tolist(),
            args.time_granularity,
        )
        share_by_period_and_keyword[period_col] = pd.Categorical(
            share_by_period_and_keyword[period_col],
            categories=ordered_periods,
            ordered=True,
        )
        share_by_period_and_keyword = share_by_period_and_keyword.sort_values(
            [args.keyword_col, period_col, "doc_share", "topic_id"],
            ascending=[True, True, False, True],
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    documents_path = output_dir / "document_topics.parquet"
    topic_info_path = output_dir / "topic_info.csv"
    topic_terms_path = output_dir / "topic_terms.csv"
    share_by_period_path = output_dir / "topic_share_by_period.csv"
    share_by_period_keyword_path = output_dir / "topic_share_by_period_and_keyword.csv"
    summary_path = output_dir / "topic_model_summary.json"

    save_dataframe(doc_topics, documents_path)
    save_dataframe(topic_info, topic_info_path)
    save_dataframe(topic_terms_df, topic_terms_path)
    save_dataframe(share_by_period, share_by_period_path)
    save_dataframe(share_by_period_and_keyword, share_by_period_keyword_path)

    if args.save_model:
        emit("Saving BERTopic model")
        model.save(output_dir / "model")

    summary = {
        "input_path": args.input_path,
        "output_dir": str(output_dir.resolve()),
        "selected_keywords": selected_keywords,
        "document_count": int(len(doc_topics)),
        "topic_count_excluding_outliers": int((topic_info["Topic"] >= 0).sum()),
        "time_granularity": args.time_granularity,
        "period_column": period_column_name(args.time_granularity),
        "documents_path": str(documents_path.resolve()),
        "topic_info_path": str(topic_info_path.resolve()),
        "topic_terms_path": str(topic_terms_path.resolve()),
        "topic_share_by_period_path": str(share_by_period_path.resolve()),
        "topic_share_by_period_and_keyword_path": str(share_by_period_keyword_path.resolve()),
    }
    save_json(summary_path, summary)
    emit(f"Saved BERTopic outputs to {output_dir}")


if __name__ == "__main__":
    main()
