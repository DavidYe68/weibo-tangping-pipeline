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
from lib.broad_analysis_layout import topic_model_output_paths
from lib.broad_analysis_overview import refresh_broad_analysis_overview
from lib.io_utils import save_json

DEFAULT_TOPIC_STOPWORDS_PATH = "bert/config/topic_stopwords.txt"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_UMAP_N_NEIGHBORS = 30
DEFAULT_OUTLIER_REDUCTION_STRATEGY = "none"
OUTLIER_REDUCTION_STRATEGIES = (
    "none",
    "c-tf-idf",
    "distributions",
    "embeddings",
    "probabilities",
    "c-tf-idf+distributions",
)
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
        default="bert/artifacts/broad_analysis/topic_model_BAAI",
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
        "--hdbscan_min_samples",
        type=int,
        default=None,
        help=(
            "Optional HDBSCAN min_samples. Defaults to min_topic_size. "
            "Lower values usually reduce outliers but can force noisier assignments."
        ),
    )
    parser.add_argument(
        "--outlier_reduction_strategy",
        choices=OUTLIER_REDUCTION_STRATEGIES,
        default=DEFAULT_OUTLIER_REDUCTION_STRATEGY,
        help=(
            "Optional BERTopic reduce_outliers strategy. "
            "Use 'c-tf-idf+distributions' to chain the two official strategies."
        ),
    )
    parser.add_argument(
        "--outlier_reduction_threshold",
        type=float,
        default=0.0,
        help=(
            "Similarity threshold passed to BERTopic reduce_outliers. "
            "Higher values make reassignment of outliers more conservative."
        ),
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


def resolve_hdbscan_min_samples(args: argparse.Namespace) -> int:
    if args.hdbscan_min_samples is None:
        return int(args.min_topic_size)
    if args.hdbscan_min_samples <= 0:
        raise ValueError("--hdbscan_min_samples must be a positive integer.")
    return int(args.hdbscan_min_samples)


def build_bertopic_model(args: argparse.Namespace, *, embedding_model, emit):
    BERTopic = load_bertopic_class()
    umap_model = None
    hdbscan_model = None
    vectorizer_model = build_topic_vectorizer(args, emit=emit)
    resolved_min_samples = resolve_hdbscan_min_samples(args)
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
            min_samples=resolved_min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=args.hdbscan_core_dist_n_jobs,
            prediction_data=args.calculate_probabilities,
        )
        emit(
            "Using explicit HDBSCAN clusterer "
            f"(min_cluster_size={args.min_topic_size}, min_samples={resolved_min_samples}, metric=euclidean, "
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


def reduce_topic_outliers(
    topic_model,
    *,
    args: argparse.Namespace,
    texts: list[str],
    topics: list[int],
    embeddings: np.ndarray,
    probabilities,
    emit,
) -> tuple[list[int], bool]:
    strategy = str(args.outlier_reduction_strategy or DEFAULT_OUTLIER_REDUCTION_STRATEGY).strip().lower()
    if strategy == "none":
        return list(topics), False

    original_topics = [int(topic_id) for topic_id in topics]
    original_outlier_count = int(sum(topic_id < 0 for topic_id in original_topics))
    if original_outlier_count <= 0:
        emit("Skipping outlier reduction because the model produced no outlier documents")
        return original_topics, False

    if strategy == "probabilities" and (probabilities is None or getattr(probabilities, "ndim", 1) != 2):
        raise ValueError(
            "--outlier_reduction_strategy probabilities requires --calculate_probabilities "
            "so BERTopic can produce a full document-topic probability matrix."
        )

    def run_single_strategy(base_topics: list[int], selected_strategy: str) -> list[int]:
        kwargs: dict[str, object] = {
            "strategy": selected_strategy,
            "threshold": float(args.outlier_reduction_threshold),
        }
        if selected_strategy == "embeddings":
            kwargs["embeddings"] = embeddings
        elif selected_strategy == "probabilities":
            kwargs["probabilities"] = probabilities
        return [int(topic_id) for topic_id in topic_model.reduce_outliers(texts, base_topics, **kwargs)]

    reduction_start = time.perf_counter()
    emit(
        "Reducing outliers with BERTopic "
        f"(strategy={strategy}, threshold={args.outlier_reduction_threshold}, "
        f"outliers_before={original_outlier_count})"
    )
    if strategy == "c-tf-idf+distributions":
        new_topics = run_single_strategy(original_topics, "c-tf-idf")
        remaining_outliers = int(sum(topic_id < 0 for topic_id in new_topics))
        emit(f"First-pass c-tf-idf outlier reduction left {remaining_outliers} outliers")
        if remaining_outliers > 0:
            new_topics = run_single_strategy(new_topics, "distributions")
    else:
        new_topics = run_single_strategy(original_topics, strategy)

    new_outlier_count = int(sum(topic_id < 0 for topic_id in new_topics))
    if new_topics == original_topics:
        emit(f"Outlier reduction made no assignment changes in {format_elapsed(reduction_start)}")
        return new_topics, False

    topic_model.update_topics(texts, topics=new_topics)
    emit(
        "Outlier reduction finished in "
        f"{format_elapsed(reduction_start)} (outliers: {original_outlier_count} -> {new_outlier_count})"
    )
    return new_topics, True


def clean_optional_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def build_topic_overview_table(
    topic_info: pd.DataFrame,
    topic_terms_df: pd.DataFrame,
    share_by_period: pd.DataFrame,
    share_by_period_and_keyword: pd.DataFrame,
    *,
    keyword_col: str,
    total_document_count: int,
) -> pd.DataFrame:
    info = topic_info.copy()
    info["Topic"] = pd.to_numeric(info["Topic"], errors="coerce").astype("Int64")
    info["Count"] = pd.to_numeric(info["Count"], errors="coerce").fillna(0).astype(int)

    clustered_document_count = int(info.loc[info["Topic"] >= 0, "Count"].sum())

    top_terms_lookup: dict[int, list[str]] = {}
    if not topic_terms_df.empty and {"topic_id", "term_rank", "term"}.issubset(topic_terms_df.columns):
        working_terms = topic_terms_df.copy()
        working_terms["topic_id"] = pd.to_numeric(working_terms["topic_id"], errors="coerce").astype("Int64")
        working_terms["term_rank"] = pd.to_numeric(working_terms["term_rank"], errors="coerce")
        working_terms = working_terms.dropna(subset=["topic_id", "term_rank", "term"]).copy()
        working_terms = working_terms.sort_values(["topic_id", "term_rank"])
        for topic_id, frame in working_terms.groupby("topic_id"):
            top_terms_lookup[int(topic_id)] = [
                clean_optional_text(term) for term in frame["term"].tolist() if clean_optional_text(term)
            ]

    peak_period_lookup: dict[int, dict[str, object]] = {}
    if not share_by_period.empty:
        working_period = share_by_period.copy()
        working_period["topic_id"] = pd.to_numeric(working_period["topic_id"], errors="coerce").astype("Int64")
        working_period["doc_count"] = pd.to_numeric(working_period["doc_count"], errors="coerce").fillna(0).astype(int)
        working_period["doc_share"] = pd.to_numeric(working_period["doc_share"], errors="coerce").fillna(0.0)
        working_period["period_label"] = working_period["period_label"].astype("string")
        working_period = working_period.dropna(subset=["topic_id"]).copy()
        working_period = working_period.sort_values(
            ["topic_id", "doc_count", "doc_share", "period_label"],
            ascending=[True, False, False, True],
        )
        for topic_id, frame in working_period.groupby("topic_id"):
            best_row = frame.iloc[0]
            peak_period_lookup[int(topic_id)] = {
                "peak_period": clean_optional_text(best_row.get("period_label")),
                "peak_doc_count": int(best_row.get("doc_count", 0)),
                "peak_doc_share_pct": float(best_row.get("doc_share", 0.0)) * 100.0,
            }

    dominant_keyword_lookup: dict[int, dict[str, object]] = {}
    if not share_by_period_and_keyword.empty and keyword_col in share_by_period_and_keyword.columns:
        working_keyword = share_by_period_and_keyword.copy()
        working_keyword["topic_id"] = pd.to_numeric(working_keyword["topic_id"], errors="coerce").astype("Int64")
        working_keyword["doc_count"] = pd.to_numeric(working_keyword["doc_count"], errors="coerce").fillna(0).astype(int)
        grouped_keyword = (
            working_keyword.dropna(subset=["topic_id"])
            .groupby(["topic_id", keyword_col], as_index=False)["doc_count"]
            .sum()
            .sort_values(["topic_id", "doc_count", keyword_col], ascending=[True, False, True])
        )
        for topic_id, frame in grouped_keyword.groupby("topic_id"):
            best_row = frame.iloc[0]
            dominant_keyword_lookup[int(topic_id)] = {
                "dominant_keyword": clean_optional_text(best_row.get(keyword_col)),
                "dominant_keyword_doc_count": int(best_row.get("doc_count", 0)),
            }

    rows: list[dict[str, object]] = []
    for row in info.itertuples(index=False):
        if pd.isna(row.Topic):
            continue
        topic_id = int(row.Topic)
        if topic_id < 0:
            continue

        topic_count = int(row.Count)
        label_machine = clean_optional_text(getattr(row, "topic_label_machine", "")) or clean_optional_text(
            getattr(row, "Name", "")
        )
        label_zh = clean_optional_text(getattr(row, "topic_label_zh", ""))
        peak = peak_period_lookup.get(topic_id, {})
        dominant_keyword = dominant_keyword_lookup.get(topic_id, {})
        dominant_keyword_doc_count = int(dominant_keyword.get("dominant_keyword_doc_count", 0))

        rows.append(
            {
                "topic_id": topic_id,
                "topic_name_raw": clean_optional_text(getattr(row, "Name", "")),
                "topic_label_machine": label_machine,
                "topic_label_zh": label_zh,
                "topic_label_display": label_zh or label_machine or f"Topic {topic_id}",
                "topic_count": topic_count,
                "share_of_all_docs_pct": (100.0 * topic_count / total_document_count) if total_document_count else 0.0,
                "share_of_clustered_docs_pct": (100.0 * topic_count / clustered_document_count)
                if clustered_document_count
                else 0.0,
                "top_terms": " / ".join(top_terms_lookup.get(topic_id, [])[:10]),
                "peak_period": clean_optional_text(peak.get("peak_period")),
                "peak_doc_count": int(peak.get("peak_doc_count", 0)),
                "peak_doc_share_pct": float(peak.get("peak_doc_share_pct", 0.0)),
                "dominant_keyword": clean_optional_text(dominant_keyword.get("dominant_keyword")),
                "dominant_keyword_doc_count": dominant_keyword_doc_count,
                "dominant_keyword_share_within_topic_pct": (100.0 * dominant_keyword_doc_count / topic_count)
                if topic_count
                else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values(["topic_count", "topic_id"], ascending=[False, True]).reset_index(drop=True)


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
    topics_before_outlier_reduction = [int(topic_id) for topic_id in topics]
    initial_outlier_document_count = int(sum(topic_id < 0 for topic_id in topics_before_outlier_reduction))
    initial_topic_count_excluding_outliers = len({topic_id for topic_id in topics_before_outlier_reduction if topic_id >= 0})
    topics, outlier_reduction_applied = reduce_topic_outliers(
        topic_model,
        args=args,
        texts=texts,
        topics=topics_before_outlier_reduction,
        embeddings=embeddings,
        probabilities=probabilities,
        emit=emit,
    )

    postprocess_start = time.perf_counter()
    emit("Post-processing topic assignments")
    doc_topics = filtered.copy()
    doc_topics["topic_id"] = topics
    if probabilities is not None and getattr(probabilities, "ndim", 1) == 1:
        topic_probabilities = []
        for row_index, topic_id in enumerate(topics):
            if topic_id < 0:
                topic_probabilities.append(float("nan"))
                continue
            if outlier_reduction_applied and topic_id != topics_before_outlier_reduction[row_index]:
                topic_probabilities.append(float("nan"))
                continue
            topic_probabilities.append(float(probabilities[row_index]))
        doc_topics["topic_probability"] = topic_probabilities
    elif probabilities is not None:
        topic_probabilities = []
        for row_index, topic_id in enumerate(topics):
            if topic_id < 0:
                topic_probabilities.append(float("nan"))
                continue
            if outlier_reduction_applied and topic_id != topics_before_outlier_reduction[row_index]:
                topic_probabilities.append(float("nan"))
                continue
            topic_probabilities.append(float(probabilities[row_index][topic_id]))
        doc_topics["topic_probability"] = topic_probabilities
    else:
        doc_topics["topic_probability"] = float("nan")
    if outlier_reduction_applied:
        reassigned_count = int(
            sum(
                new_topic_id != old_topic_id
                for new_topic_id, old_topic_id in zip(topics, topics_before_outlier_reduction)
            )
        )
        emit(
            "Outlier reassignment updated "
            f"{reassigned_count} documents; reassigned rows keep topic_id/topic_label but "
            "their topic_probability is set to NaN because BERTopic does not emit updated HDBSCAN probabilities here."
        )

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

    paths = topic_model_output_paths(output_dir)
    for directory_key in ["readouts_dir", "viz_inputs_dir"]:
        paths[directory_key].mkdir(parents=True, exist_ok=True)
    documents_path = paths["documents_path"]
    topic_info_path = paths["topic_info_path"]
    topic_terms_path = paths["topic_terms_path"]
    share_by_period_path = paths["topic_share_by_period_path"]
    share_by_period_keyword_path = paths["topic_share_by_period_and_keyword_path"]
    share_by_ip_path = paths["topic_share_by_ip_path"]
    share_by_period_ip_path = paths["topic_share_by_period_and_ip_path"]
    share_by_period_ip_keyword_path = paths["topic_share_by_period_and_ip_and_keyword_path"]
    topic_overview_path = paths["topic_overview_path"]
    summary_path = paths["summary_path"]

    save_start = time.perf_counter()
    emit(f"Saving outputs under {output_dir}")
    topic_overview_df = build_topic_overview_table(
        topic_info,
        topic_terms_df,
        share_by_period,
        share_by_period_and_keyword,
        keyword_col=args.keyword_col,
        total_document_count=len(doc_topics),
    )
    save_dataframe(doc_topics, documents_path)
    save_dataframe(topic_info, topic_info_path)
    save_dataframe(topic_terms_df, topic_terms_path)
    save_dataframe(share_by_period, share_by_period_path)
    save_dataframe(share_by_period_and_keyword, share_by_period_keyword_path)
    save_dataframe(share_by_ip, share_by_ip_path)
    save_dataframe(share_by_period_and_ip, share_by_period_ip_path)
    save_dataframe(share_by_period_ip_keyword, share_by_period_ip_keyword_path)
    save_dataframe(topic_overview_df, topic_overview_path)

    if args.save_model:
        emit("Saving BERTopic model")
        topic_model.save(paths["model_dir"], save_embedding_model=args.embedding_model)

    outlier_document_count = int(topic_info.loc[pd.to_numeric(topic_info["Topic"], errors="coerce") == -1, "Count"].sum())
    clustered_document_count = int(len(doc_topics) - outlier_document_count)
    summary = {
        "input_path": args.input_path,
        "output_dir": str(output_dir.resolve()),
        "readouts_dir": str(paths["readouts_dir"].resolve()),
        "viz_inputs_dir": str(paths["viz_inputs_dir"].resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "selected_keywords": selected_keywords,
        "document_count": int(len(doc_topics)),
        "clustered_document_count": clustered_document_count,
        "outlier_document_count": outlier_document_count,
        "outlier_share": float(outlier_document_count / len(doc_topics)) if len(doc_topics) > 0 else 0.0,
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
        "hdbscan_min_samples": int(resolve_hdbscan_min_samples(args)),
        "nr_topics": args.nr_topics,
        "initial_topic_count_excluding_outliers": int(initial_topic_count_excluding_outliers),
        "initial_outlier_document_count": int(initial_outlier_document_count),
        "initial_outlier_share": (
            float(initial_outlier_document_count / len(doc_topics)) if len(doc_topics) > 0 else 0.0
        ),
        "outlier_reduction_strategy": str(args.outlier_reduction_strategy),
        "outlier_reduction_threshold": float(args.outlier_reduction_threshold),
        "outlier_reduction_applied": bool(outlier_reduction_applied),
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
        "topic_overview_path": str(topic_overview_path.resolve()),
    }
    save_json(summary_path, summary)
    try:
        refresh_broad_analysis_overview(output_dir)
    except Exception as exc:
        emit(f"Skipped broad-analysis overview refresh: {exc}")
    emit(f"Saved outputs in {format_elapsed(save_start)}")
    emit(f"Saved BERTopic outputs to {output_dir}")
    emit(f"Total runtime: {format_elapsed(total_start)}")


if __name__ == "__main__":
    main()
