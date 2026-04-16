#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from lib.analysis_utils import (
    DEFAULT_ANALYSIS_KEYWORDS,
    MISSING_IP_LABEL,
    attach_ip_columns,
    detect_ip_column,
    flatten_topic_terms,
    load_tabular_files,
    load_term_list,
    normalize_cli_keywords,
    period_column_name,
    resolve_emit,
    save_dataframe,
    sort_period_labels,
)
from lib.io_utils import save_json

DEFAULT_TOPIC_STOPWORDS_PATH = "bert/config/topic_stopwords.txt"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_UMAP_N_NEIGHBORS = 30
TOPIC_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+")
TOPIC_SEGMENT_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


class ChineseTopicTokenizer:
    def __init__(self, *, stopwords: set[str], token_min_length: int, prefer_jieba: bool = True) -> None:
        self.stopwords = frozenset(stopwords)
        self.token_min_length = int(token_min_length)
        self.prefer_jieba = bool(prefer_jieba)

    def _normalize_token(self, candidate: str) -> str | None:
        normalized = candidate.strip().lower()
        if not normalized:
            return None
        if normalized in self.stopwords:
            return None
        if len(normalized) < self.token_min_length:
            return None
        if not TOPIC_TOKEN_RE.fullmatch(normalized):
            return None
        return normalized

    def _fallback_tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for chunk in TOPIC_SEGMENT_RE.findall(text):
            if not chunk:
                continue
            if chunk.isascii():
                normalized = self._normalize_token(chunk)
                if normalized:
                    tokens.append(normalized)
                continue

            if len(chunk) <= 4:
                normalized = self._normalize_token(chunk)
                if normalized:
                    tokens.append(normalized)
                continue

            for index in range(len(chunk) - 1):
                normalized = self._normalize_token(chunk[index : index + 2])
                if normalized:
                    tokens.append(normalized)
        return tokens

    def __call__(self, text: str) -> list[str]:
        if self.prefer_jieba:
            try:
                import jieba
            except ImportError:
                pass
            else:
                tokens: list[str] = []
                for token in jieba.lcut(text, cut_all=False):
                    normalized = self._normalize_token(token)
                    if normalized:
                        tokens.append(normalized)
                return tokens

        return self._fallback_tokenize(text)


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


def has_jieba() -> bool:
    try:
        import jieba  # noqa: F401
    except ImportError:
        return False
    return True


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


def load_checkpoint_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read checkpoint manifest: {path}") from exc


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
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--topic_language",
        choices=["multilingual", "english"],
        default="multilingual",
        help="BERTopic language setting used during text preprocessing and topic representation.",
    )
    parser.add_argument(
        "--topic_tokenizer",
        choices=["jieba", "default"],
        default="jieba",
        help="Tokenizer backend for topic term extraction. Defaults to jieba for Chinese text.",
    )
    parser.add_argument(
        "--topic_stopwords_path",
        default=DEFAULT_TOPIC_STOPWORDS_PATH,
        help="UTF-8 text file with one stopword per line for topic term extraction.",
    )
    parser.add_argument(
        "--topic_token_min_length",
        type=int,
        default=2,
        help="Minimum token length kept by the jieba topic tokenizer.",
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
        "--umap_verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show UMAP progress logs during BERTopic dimensionality reduction.",
    )
    parser.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=DEFAULT_UMAP_N_NEIGHBORS,
        help="UMAP n_neighbors parameter. Larger values encourage broader topic neighborhoods.",
    )
    parser.add_argument(
        "--umap_low_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use UMAP low-memory mode. Recommended for large corpora.",
    )
    parser.add_argument(
        "--calculate_probabilities",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Calculate full topic-probability vectors for every document. "
            "This is much more memory-intensive than assigned-topic probabilities."
        ),
    )
    parser.add_argument(
        "--hdbscan_core_dist_n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for HDBSCAN core-distance calculations. Lower values use less memory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpointed embeddings and dimensionality-reduction outputs if available.",
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


def build_topic_vectorizer(args: argparse.Namespace, *, emit):
    if args.topic_tokenizer == "default":
        emit("Using default CountVectorizer tokenization for topic term extraction")
        return None

    stopwords = set(load_term_list(args.topic_stopwords_path))
    jieba_available = has_jieba()
    emit(
        "Using Chinese CountVectorizer tokenization for topic term extraction "
        f"(backend={'jieba' if jieba_available else 'fallback_cjk'}, "
        f"stopwords={len(stopwords)}, token_min_length={args.topic_token_min_length})"
    )
    return CountVectorizer(
        tokenizer=ChineseTopicTokenizer(
            stopwords=stopwords,
            token_min_length=args.topic_token_min_length,
            prefer_jieba=jieba_available,
        ),
        token_pattern=None,
        lowercase=False,
    )


def build_bertopic_model(args: argparse.Namespace, *, embedding_model, emit):
    BERTopic = load_bertopic_class()
    umap_model = None
    hdbscan_model = None
    vectorizer_model = build_topic_vectorizer(args, emit=emit)
    try:
        from umap import UMAP
    except ImportError:
        emit("UMAP is not installed; BERTopic will fall back to its default reducer.")
    else:
        umap_model = UMAP(
            n_neighbors=args.umap_n_neighbors,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            low_memory=args.umap_low_memory,
            random_state=args.seed,
            verbose=args.umap_verbose,
        )
        emit(
            "Using explicit UMAP reducer "
            f"(n_neighbors={args.umap_n_neighbors}, n_components=5, min_dist=0.0, metric=cosine, "
            f"low_memory={args.umap_low_memory}, random_state={args.seed}, verbose={args.umap_verbose})"
        )
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        emit("hdbscan is not installed; BERTopic will fall back to its default clusterer.")
    else:
        hdbscan_model = HDBSCAN(
            min_cluster_size=args.min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=args.hdbscan_core_dist_n_jobs,
            prediction_data=args.calculate_probabilities,
        )
        emit(
            "Using explicit HDBSCAN clusterer "
            f"(min_cluster_size={args.min_topic_size}, metric=euclidean, "
            f"cluster_selection_method=eom, core_dist_n_jobs={args.hdbscan_core_dist_n_jobs}, "
            f"prediction_data={args.calculate_probabilities})"
        )
    topic_model = BERTopic(
        language=args.topic_language,
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=args.min_topic_size,
        top_n_words=args.top_n_words,
        nr_topics=args.nr_topics,
        low_memory=args.umap_low_memory,
        calculate_probabilities=args.calculate_probabilities,
        verbose=True,
    )
    emit(
        "BERTopic model initialized "
        f"(language={args.topic_language}, topic_tokenizer={args.topic_tokenizer}, "
        f"min_topic_size={args.min_topic_size}, top_n_words={args.top_n_words}, "
        f"nr_topics={args.nr_topics}, calculate_probabilities={args.calculate_probabilities})"
    )
    return topic_model


def build_reducer_signature(topic_model, args: argparse.Namespace) -> dict[str, object]:
    reducer_name = type(topic_model.umap_model).__name__ if topic_model.umap_model is not None else "None"
    signature: dict[str, object] = {"reducer_class": reducer_name}
    if reducer_name == "UMAP":
        signature.update(
            {
                "n_neighbors": int(args.umap_n_neighbors),
                "n_components": 5,
                "min_dist": 0.0,
                "metric": "cosine",
                "low_memory": bool(args.umap_low_memory),
                "seed": int(args.seed),
            }
        )
    return signature


def reduce_embeddings_with_checkpoint(
    topic_model,
    *,
    args: argparse.Namespace,
    embeddings: np.ndarray,
    checkpoint_dir: Path,
    previous_manifest: dict,
    current_fingerprint: str,
    emit,
) -> tuple[np.ndarray, Path, Path, bool]:
    reduced_embeddings_checkpoint_path = checkpoint_dir / "reduced_embeddings.npy"
    reducer_model_checkpoint_path = checkpoint_dir / "dimensionality_reduction_model.pkl"
    reducer_signature = build_reducer_signature(topic_model, args)

    can_resume_reducer = (
        args.resume
        and reduced_embeddings_checkpoint_path.exists()
        and reducer_model_checkpoint_path.exists()
        and previous_manifest.get("document_fingerprint") == current_fingerprint
        and previous_manifest.get("reducer_signature") == reducer_signature
        and previous_manifest.get("embedding_model") == args.embedding_model
    )

    if can_resume_reducer:
        emit(f"Resuming from dimensionality-reduction checkpoint: {reduced_embeddings_checkpoint_path}")
        reduce_start = time.perf_counter()
        reduced_embeddings = np.load(reduced_embeddings_checkpoint_path)
        if reduced_embeddings.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "Dimensionality-reduction checkpoint row count does not match the filtered dataset. "
                "Delete the checkpoint or rerun without --resume."
            )
        try:
            with reducer_model_checkpoint_path.open("rb") as handle:
                topic_model.umap_model = pickle.load(handle)
        except Exception as exc:
            raise ValueError(
                f"Failed to load dimensionality-reduction model checkpoint: {reducer_model_checkpoint_path}"
            ) from exc
        emit(
            "Loaded dimensionality-reduction checkpoint in "
            f"{format_elapsed(reduce_start)}"
        )
        return reduced_embeddings, reduced_embeddings_checkpoint_path, reducer_model_checkpoint_path, True

    reduce_start = time.perf_counter()
    emit(f"Reducing {embeddings.shape[0]} embeddings before clustering")
    reduced_embeddings = topic_model._reduce_dimensionality(embeddings, y=None)
    np.save(reduced_embeddings_checkpoint_path, reduced_embeddings)
    with reducer_model_checkpoint_path.open("wb") as handle:
        pickle.dump(topic_model.umap_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    emit(
        "Dimensionality reduction finished in "
        f"{format_elapsed(reduce_start)} and checkpointed to {reduced_embeddings_checkpoint_path}"
    )
    return reduced_embeddings, reduced_embeddings_checkpoint_path, reducer_model_checkpoint_path, False


def fit_topic_model_with_precomputed_reduction(
    topic_model,
    *,
    texts: list[str],
    embeddings: np.ndarray,
    reduced_embeddings: np.ndarray,
):
    documents = pd.DataFrame(
        {
            "Document": texts,
            "ID": range(len(texts)),
            "Topic": None,
            "Image": None,
        }
    )

    documents, probabilities = topic_model._cluster_embeddings(reduced_embeddings, documents, y=None)
    if not topic_model.nr_topics:
        documents = topic_model._sort_mappings_by_frequency(documents)

    topic_model._extract_topics(
        documents,
        embeddings=embeddings,
        verbose=topic_model.verbose,
        fine_tune_representation=not topic_model.nr_topics,
    )
    if topic_model.nr_topics:
        documents = topic_model._reduce_topics(documents)
    topic_model._save_representative_docs(documents)
    topic_model.probabilities_ = topic_model._map_probabilities(probabilities, original_topics=True)
    return documents.Topic.to_list(), topic_model.probabilities_


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
    reduced_embeddings_checkpoint_path = checkpoint_dir / "reduced_embeddings.npy"
    reducer_model_checkpoint_path = checkpoint_dir / "dimensionality_reduction_model.pkl"

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
    del df
    gc.collect()
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
    previous_manifest = load_checkpoint_manifest(checkpoint_manifest_path)

    texts = filtered[args.text_col].tolist()
    encoder = None
    embedding_device = "checkpoint"
    embeddings: np.ndarray

    if args.resume and embeddings_checkpoint_path.exists() and checkpoint_manifest_path.exists():
        checkpoint_fingerprint = previous_manifest.get("document_fingerprint")
        checkpoint_embedding_model = previous_manifest.get("embedding_model")
        if checkpoint_fingerprint != current_fingerprint:
            raise ValueError(
                "Checkpointed embeddings do not match the current filtered dataset. "
                "Delete the checkpoint directory or rerun without --resume."
            )
        if checkpoint_embedding_model != args.embedding_model:
            emit(
                "Embedding checkpoint metadata does not match the requested embedding model; "
                "recomputing embeddings instead of resuming."
            )
        else:
            emit(f"Resuming from embedding checkpoint: {embeddings_checkpoint_path}")
            embedding_start = time.perf_counter()
            embeddings = np.load(embeddings_checkpoint_path, mmap_mode="r")
            if embeddings.shape[0] != len(texts):
                raise ValueError(
                    "Embedding checkpoint row count does not match the filtered dataset. "
                    "Delete the checkpoint or rerun without --resume."
                )
            emit(
                f"Loaded {embeddings.shape[0]} embeddings from checkpoint in {format_elapsed(embedding_start)}"
            )
            checkpoint_embedding_model = args.embedding_model
    if "embeddings" not in locals():
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
        encoder = None
        gc.collect()

    topic_model = build_bertopic_model(args, embedding_model=None, emit=emit)
    reducer_signature = build_reducer_signature(topic_model, args)
    save_dataframe(filtered, filtered_checkpoint_path)
    save_json(
        checkpoint_manifest_path,
        {
            "document_count": int(len(filtered)),
            "document_fingerprint": current_fingerprint,
            "text_col": args.text_col,
            "keyword_col": args.keyword_col,
            "time_col": args.time_col,
            "embedding_model": args.embedding_model,
            "reducer_signature": reducer_signature,
            "embeddings_checkpoint_path": str(embeddings_checkpoint_path.resolve()),
            "reduced_embeddings_checkpoint_path": str(reduced_embeddings_checkpoint_path.resolve()),
            "reducer_model_checkpoint_path": str(reducer_model_checkpoint_path.resolve()),
        },
    )
    emit(f"Checkpointed filtered documents to {filtered_checkpoint_path}")

    fit_start = time.perf_counter()
    reduced_embeddings, reduced_embeddings_checkpoint_path, reducer_model_checkpoint_path, used_reducer_checkpoint = (
        reduce_embeddings_with_checkpoint(
            topic_model,
            args=args,
            embeddings=embeddings,
            checkpoint_dir=checkpoint_dir,
            previous_manifest=previous_manifest,
            current_fingerprint=current_fingerprint,
            emit=emit,
        )
    )
    if used_reducer_checkpoint:
        emit(
            f"Skipping dimensionality reduction and resuming BERTopic from clustering "
            f"(embedding_device={embedding_device})"
        )
    else:
        emit(
            f"Starting BERTopic clustering and topic extraction on {len(texts)} documents "
            f"(embedding_device={embedding_device})"
        )
    topics, probabilities = fit_topic_model_with_precomputed_reduction(
        topic_model,
        texts=texts,
        embeddings=embeddings,
        reduced_embeddings=reduced_embeddings,
    )
    emit(f"BERTopic pipeline finished in {format_elapsed(fit_start)}")

    postprocess_start = time.perf_counter()
    emit("Post-processing topic assignments")
    doc_topics = filtered.copy()
    doc_topics["topic_id"] = topics
    if probabilities is not None and getattr(probabilities, "ndim", 1) == 1:
        doc_topics["topic_probability"] = pd.Series(probabilities, copy=False).astype(float)
    elif probabilities is not None:
        topic_probabilities = []
        for row_index, topic_id in enumerate(topics):
            if topic_id < 0:
                topic_probabilities.append(float("nan"))
                continue
            topic_probabilities.append(float(probabilities[row_index][topic_id]))
        doc_topics["topic_probability"] = topic_probabilities
    else:
        doc_topics["topic_probability"] = float("nan")

    topic_info = topic_model.get_topic_info().copy()
    if "topic_label_machine" not in topic_info.columns:
        if "Name" in topic_info.columns:
            topic_info["topic_label_machine"] = topic_info["Name"].fillna("").astype("string")
        else:
            topic_info["topic_label_machine"] = topic_info["Topic"].map(lambda value: f"Topic {value}")
    if "topic_label_zh" not in topic_info.columns:
        topic_info["topic_label_zh"] = pd.Series([""] * len(topic_info), index=topic_info.index, dtype="string")

    topic_labels = {}
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        label = row.get("topic_label_machine")
        if pd.isna(label) or not str(label).strip():
            label = row.get("Name")
        if pd.isna(label) or not str(label).strip():
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
        topic_model.save(output_dir / "model", save_embedding_model=args.embedding_model)

    summary = {
        "input_path": args.input_path,
        "output_dir": str(output_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "selected_keywords": selected_keywords,
        "document_count": int(len(doc_topics)),
        "topic_count_excluding_outliers": int((topic_info["Topic"] >= 0).sum()),
        "embedding_device": embedding_device,
        "embedding_model": args.embedding_model,
        "topic_language": args.topic_language,
        "topic_tokenizer": args.topic_tokenizer,
        "topic_stopwords_path": args.topic_stopwords_path,
        "topic_token_min_length": int(args.topic_token_min_length),
        "calculate_probabilities": bool(args.calculate_probabilities),
        "umap_low_memory": bool(args.umap_low_memory),
        "umap_n_neighbors": int(args.umap_n_neighbors),
        "hdbscan_core_dist_n_jobs": int(args.hdbscan_core_dist_n_jobs),
        "reducer_signature": reducer_signature,
        "resolved_ip_col": resolved_ip_col,
        "unique_ip_count_excluding_missing": int(doc_topics.loc[~doc_topics["ip_missing"], "ip_normalized"].nunique()),
        "missing_ip_document_count": int(doc_topics["ip_missing"].sum()),
        "missing_ip_rate": float(doc_topics["ip_missing"].mean()) if len(doc_topics) > 0 else 0.0,
        "embeddings_checkpoint_path": str(embeddings_checkpoint_path.resolve()),
        "reduced_embeddings_checkpoint_path": str(reduced_embeddings_checkpoint_path.resolve()),
        "reducer_model_checkpoint_path": str(reducer_model_checkpoint_path.resolve()),
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
