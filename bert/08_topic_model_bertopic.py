#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib.analysis_utils import (
    DEFAULT_ANALYSIS_KEYWORDS,
    MISSING_IP_LABEL,
    attach_ip_columns,
    detect_ip_column,
    flatten_topic_terms,
    load_tabular_files,
    normalize_cli_keywords,
    period_column_name,
    resolve_emit,
    save_dataframe,
    sort_period_labels,
)
from lib.io_utils import save_json


def parse_nr_topics(value: str) -> int | str:
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("--nr_topics cannot be empty.")
    if normalized.lower() == "auto":
        return "auto"
    try:
        return int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--nr_topics must be an integer or 'auto'.") from exc


def format_elapsed(start_time: float) -> str:
    return f"{time.perf_counter() - start_time:.2f}s"


def resolve_embedding_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def missing_dependency_message(package_name: str) -> str:
    return (
        f"{package_name} is not installed in the current environment. "
        "Install the project dependencies first, for example with "
        "`.venv/bin/pip install -r requirements.txt` on macOS/Linux or "
        "`.\\.venv\\Scripts\\pip.exe install -r requirements.txt` on Windows."
    )


def sort_ip_labels(labels: list[str]) -> list[str]:
    unique = sorted({str(label) for label in labels if label is not None and str(label) != ""})
    if MISSING_IP_LABEL in unique:
        unique = [label for label in unique if label != MISSING_IP_LABEL] + [MISSING_IP_LABEL]
    return unique


def compute_document_fingerprint(
    df: pd.DataFrame,
    *,
    text_col: str,
    keyword_col: str,
    time_col: str,
) -> str:
    columns = [text_col, keyword_col, time_col]
    if "ip_normalized" in df.columns:
        columns.append("ip_normalized")
    frame = df[columns].copy()
    frame = frame.fillna("").astype("string")
    row_hashes = pd.util.hash_pandas_object(frame, index=False)
    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()


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
    parser.add_argument("--ip_col", default=None, help="Optional IP column name. Defaults to auto-detect.")
    parser.add_argument(
        "--time_granularity",
        choices=["month", "quarter", "year"],
        default="month",
        help="Time granularity for topic-share outputs.",
    )
    parser.add_argument("--min_topic_size", type=int, default=30, help="BERTopic min_topic_size.")
    parser.add_argument("--top_n_words", type=int, default=10, help="Top words per topic.")
    parser.add_argument(
        "--nr_topics",
        type=parse_nr_topics,
        default=None,
        help="Optional BERTopic nr_topics value. Accepts an integer or 'auto'.",
    )
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
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for the sentence-transformers embedding model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpointed embeddings if available.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Optional checkpoint directory. Defaults to <output_dir>/checkpoints.",
    )
    return parser.parse_args()


def load_bertopic_class():
    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise ImportError(missing_dependency_message("BERTopic")) from exc
    return BERTopic


def load_sentence_encoder(args: argparse.Namespace, *, emit):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(missing_dependency_message("sentence-transformers")) from exc

    resolved_device = resolve_embedding_device(args.device)
    emit(
        "Loading sentence-transformers model "
        f"{args.embedding_model} on device={resolved_device}"
    )
    encoder = SentenceTransformer(
        args.embedding_model,
        device=resolved_device,
        local_files_only=bool(args.local_files_only),
    )
    actual_device = str(getattr(encoder, "device", resolved_device))
    emit(f"Embedding model reports device={actual_device}")
    return encoder, actual_device


def build_bertopic_model(args: argparse.Namespace, *, embedding_model, emit):
    BERTopic = load_bertopic_class()
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        nr_topics=args.nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )
    emit(
        "BERTopic model initialized "
        f"(min_topic_size={args.min_topic_size}, top_n_words={args.top_n_words}, nr_topics={args.nr_topics})"
    )
    return topic_model


def main() -> None:
    args = parse_args()
    emit = resolve_emit("bertopic", None)
    total_start = time.perf_counter()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filtered_checkpoint_path = checkpoint_dir / "filtered_documents.parquet"
    embeddings_checkpoint_path = checkpoint_dir / "document_embeddings.npy"
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"

    emit(f"Loading analysis base from {args.input_path}")
    load_start = time.perf_counter()
    df, _ = load_tabular_files(args.input_path, emit=emit)
    emit(f"Loaded {len(df)} rows in {format_elapsed(load_start)}")
    selected_keywords = normalize_cli_keywords(args.keywords)
    if args.keyword_col not in df.columns:
        raise ValueError(f"Keyword column '{args.keyword_col}' not found in analysis base.")
    if args.text_col not in df.columns:
        raise ValueError(f"Text column '{args.text_col}' not found in analysis base.")
    if args.time_col not in df.columns:
        raise ValueError(f"Time column '{args.time_col}' not found in analysis base.")
    resolved_ip_col = detect_ip_column(df, args.ip_col, required=False)
    df = attach_ip_columns(df, ip_col=resolved_ip_col)
    emit(
        "Using IP column "
        f"{resolved_ip_col if resolved_ip_col else 'not found; assigning all rows to ' + MISSING_IP_LABEL}"
    )

    filtered = df[df[args.keyword_col].isin(selected_keywords)].copy()
    filtered[args.text_col] = filtered[args.text_col].fillna("").astype("string").str.strip()
    filtered = filtered[filtered[args.text_col].ne("")].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No rows left after keyword filtering.")

    filtered["period_label"] = pd.to_datetime(filtered[args.time_col], errors="coerce")
    filtered["period_label"] = filtered["period_label"].dt.to_period(
        {"month": "M", "quarter": "Q", "year": "Y"}[args.time_granularity]
    ).astype("string")
    emit(
        "Prepared filtered dataset "
        f"with {len(filtered)} documents, {filtered['period_label'].nunique(dropna=False)} periods, "
        f"keywords={selected_keywords}, unique_ip={filtered.loc[~filtered['ip_missing'], 'ip_normalized'].nunique()}, "
        f"missing_ip_rows={int(filtered['ip_missing'].sum())}"
    )
    current_fingerprint = compute_document_fingerprint(
        filtered,
        text_col=args.text_col,
        keyword_col=args.keyword_col,
        time_col=args.time_col,
    )

    texts = filtered[args.text_col].tolist()
    encoder = None
    embedding_device = "checkpoint"
    embeddings: np.ndarray

    if args.resume and embeddings_checkpoint_path.exists() and checkpoint_manifest_path.exists():
        emit(f"Resuming from embedding checkpoint: {embeddings_checkpoint_path}")
        manifest = {}
        try:
            import json

            manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to read checkpoint manifest: {checkpoint_manifest_path}") from exc

        checkpoint_fingerprint = manifest.get("document_fingerprint")
        if checkpoint_fingerprint != current_fingerprint:
            raise ValueError(
                "Checkpointed embeddings do not match the current filtered dataset. "
                "Delete the checkpoint directory or rerun without --resume."
            )
        embedding_start = time.perf_counter()
        embeddings = np.load(embeddings_checkpoint_path)
        if embeddings.shape[0] != len(texts):
            raise ValueError(
                "Embedding checkpoint row count does not match the filtered dataset. "
                "Delete the checkpoint or rerun without --resume."
            )
        emit(
            f"Loaded {embeddings.shape[0]} embeddings from checkpoint in {format_elapsed(embedding_start)}"
        )
    else:
        model_start = time.perf_counter()
        encoder, embedding_device = load_sentence_encoder(args, emit=emit)
        emit(f"Embedding model loading finished in {format_elapsed(model_start)}")

        embedding_start = time.perf_counter()
        emit(f"Encoding {len(texts)} documents into embeddings")
        embeddings = np.asarray(
            encoder.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        )
        np.save(embeddings_checkpoint_path, embeddings)
        emit(
            "Embedding encoding finished in "
            f"{format_elapsed(embedding_start)} and checkpointed to {embeddings_checkpoint_path}"
        )

    save_dataframe(filtered, filtered_checkpoint_path)
    save_json(
        checkpoint_manifest_path,
        {
            "document_count": int(len(filtered)),
            "document_fingerprint": current_fingerprint,
            "text_col": args.text_col,
            "keyword_col": args.keyword_col,
            "time_col": args.time_col,
        },
    )
    emit(f"Checkpointed filtered documents to {filtered_checkpoint_path}")

    fit_start = time.perf_counter()
    topic_model = build_bertopic_model(args, embedding_model=encoder, emit=emit)
    emit(
        f"Starting BERTopic fit_transform on {len(texts)} documents "
        f"(embedding_device={embedding_device})"
    )
    topics, probabilities = topic_model.fit_transform(texts, embeddings=embeddings)
    emit(f"BERTopic fit_transform finished in {format_elapsed(fit_start)}")

    postprocess_start = time.perf_counter()
    emit("Post-processing topic assignments")
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

    topic_info = topic_model.get_topic_info()
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
        topic_terms[topic_id_int] = topic_model.get_topic(topic_id_int) or []

    topic_terms_df = flatten_topic_terms(topic_terms)
    emit(
        "Collected topic outputs "
        f"with {int((topic_info['Topic'] >= 0).sum())} non-outlier topics"
    )

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

    ip_col = "ip_normalized"
    share_by_ip = (
        doc_topics[doc_topics["topic_id"] >= 0]
        .groupby([ip_col, "topic_id", "topic_label"], dropna=False)
        .size()
        .reset_index(name="doc_count")
    )
    if not share_by_ip.empty:
        ip_totals = share_by_ip.groupby(ip_col)["doc_count"].transform("sum")
        share_by_ip["doc_share"] = share_by_ip["doc_count"] / ip_totals
        ordered_ips = sort_ip_labels(share_by_ip[ip_col].astype(str).tolist())
        share_by_ip[ip_col] = pd.Categorical(
            share_by_ip[ip_col],
            categories=ordered_ips,
            ordered=True,
        )
        share_by_ip = share_by_ip.sort_values([ip_col, "doc_share", "topic_id"], ascending=[True, False, True])

    share_by_period_and_ip = (
        doc_topics[doc_topics["topic_id"] >= 0]
        .groupby([period_col, ip_col, "topic_id", "topic_label"], dropna=False)
        .size()
        .reset_index(name="doc_count")
    )
    if not share_by_period_and_ip.empty:
        grouped_totals = share_by_period_and_ip.groupby([period_col, ip_col])["doc_count"].transform("sum")
        share_by_period_and_ip["doc_share"] = share_by_period_and_ip["doc_count"] / grouped_totals
        ordered_periods = sort_period_labels(
            share_by_period_and_ip[period_col].astype(str).tolist(),
            args.time_granularity,
        )
        ordered_ips = sort_ip_labels(share_by_period_and_ip[ip_col].astype(str).tolist())
        share_by_period_and_ip[period_col] = pd.Categorical(
            share_by_period_and_ip[period_col],
            categories=ordered_periods,
            ordered=True,
        )
        share_by_period_and_ip[ip_col] = pd.Categorical(
            share_by_period_and_ip[ip_col],
            categories=ordered_ips,
            ordered=True,
        )
        share_by_period_and_ip = share_by_period_and_ip.sort_values(
            [period_col, ip_col, "doc_share", "topic_id"],
            ascending=[True, True, False, True],
        )

    share_by_period_ip_keyword = (
        doc_topics[doc_topics["topic_id"] >= 0]
        .groupby([args.keyword_col, period_col, ip_col, "topic_id", "topic_label"], dropna=False)
        .size()
        .reset_index(name="doc_count")
    )
    if not share_by_period_ip_keyword.empty:
        grouped_totals = share_by_period_ip_keyword.groupby([args.keyword_col, period_col, ip_col])["doc_count"].transform("sum")
        share_by_period_ip_keyword["doc_share"] = share_by_period_ip_keyword["doc_count"] / grouped_totals
        ordered_periods = sort_period_labels(
            share_by_period_ip_keyword[period_col].astype(str).tolist(),
            args.time_granularity,
        )
        ordered_ips = sort_ip_labels(share_by_period_ip_keyword[ip_col].astype(str).tolist())
        share_by_period_ip_keyword[period_col] = pd.Categorical(
            share_by_period_ip_keyword[period_col],
            categories=ordered_periods,
            ordered=True,
        )
        share_by_period_ip_keyword[ip_col] = pd.Categorical(
            share_by_period_ip_keyword[ip_col],
            categories=ordered_ips,
            ordered=True,
        )
        share_by_period_ip_keyword = share_by_period_ip_keyword.sort_values(
            [args.keyword_col, period_col, ip_col, "doc_share", "topic_id"],
            ascending=[True, True, True, False, True],
        )
    emit(f"Post-processing finished in {format_elapsed(postprocess_start)}")

    documents_path = output_dir / "document_topics.parquet"
    topic_info_path = output_dir / "topic_info.csv"
    topic_terms_path = output_dir / "topic_terms.csv"
    share_by_period_path = output_dir / "topic_share_by_period.csv"
    share_by_period_keyword_path = output_dir / "topic_share_by_period_and_keyword.csv"
    share_by_ip_path = output_dir / "topic_share_by_ip.csv"
    share_by_period_ip_path = output_dir / "topic_share_by_period_and_ip.csv"
    share_by_period_ip_keyword_path = output_dir / "topic_share_by_period_and_ip_and_keyword.csv"
    summary_path = output_dir / "topic_model_summary.json"

    save_start = time.perf_counter()
    emit(f"Saving outputs under {output_dir}")
    save_dataframe(doc_topics, documents_path)
    save_dataframe(topic_info, topic_info_path)
    save_dataframe(topic_terms_df, topic_terms_path)
    save_dataframe(share_by_period, share_by_period_path)
    save_dataframe(share_by_period_and_keyword, share_by_period_keyword_path)
    save_dataframe(share_by_ip, share_by_ip_path)
    save_dataframe(share_by_period_and_ip, share_by_period_ip_path)
    save_dataframe(share_by_period_ip_keyword, share_by_period_ip_keyword_path)

    if args.save_model:
        emit("Saving BERTopic model")
        topic_model.save(output_dir / "model")

    summary = {
        "input_path": args.input_path,
        "output_dir": str(output_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "selected_keywords": selected_keywords,
        "document_count": int(len(doc_topics)),
        "topic_count_excluding_outliers": int((topic_info["Topic"] >= 0).sum()),
        "embedding_device": embedding_device,
        "embedding_model": args.embedding_model,
        "resolved_ip_col": resolved_ip_col,
        "unique_ip_count_excluding_missing": int(doc_topics.loc[~doc_topics["ip_missing"], "ip_normalized"].nunique()),
        "missing_ip_document_count": int(doc_topics["ip_missing"].sum()),
        "missing_ip_rate": float(doc_topics["ip_missing"].mean()) if len(doc_topics) > 0 else 0.0,
        "embeddings_checkpoint_path": str(embeddings_checkpoint_path.resolve()),
        "filtered_documents_checkpoint_path": str(filtered_checkpoint_path.resolve()),
        "checkpoint_manifest_path": str(checkpoint_manifest_path.resolve()),
        "time_granularity": args.time_granularity,
        "period_column": period_column_name(args.time_granularity),
        "documents_path": str(documents_path.resolve()),
        "topic_info_path": str(topic_info_path.resolve()),
        "topic_terms_path": str(topic_terms_path.resolve()),
        "topic_share_by_period_path": str(share_by_period_path.resolve()),
        "topic_share_by_period_and_keyword_path": str(share_by_period_keyword_path.resolve()),
        "topic_share_by_ip_path": str(share_by_ip_path.resolve()),
        "topic_share_by_period_and_ip_path": str(share_by_period_ip_path.resolve()),
        "topic_share_by_period_and_ip_and_keyword_path": str(share_by_period_ip_keyword_path.resolve()),
    }
    save_json(summary_path, summary)
    emit(f"Saved outputs in {format_elapsed(save_start)}")
    emit(f"Saved BERTopic outputs to {output_dir}")
    emit(f"Total runtime: {format_elapsed(total_start)}")


if __name__ == "__main__":
    main()
