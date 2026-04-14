#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from lib.analysis_utils import (
    DEFAULT_ANALYSIS_KEYWORDS,
    coerce_period_series,
    load_tabular_files,
    normalize_cli_keywords,
    resolve_emit,
    save_dataframe,
    sort_period_labels,
)
from lib.io_utils import save_json


DEFAULT_STOPWORDS = {
    "我们",
    "你们",
    "他们",
    "就是",
    "一个",
    "没有",
    "这个",
    "那个",
    "真的",
    "自己",
    "现在",
    "因为",
    "但是",
    "还是",
    "已经",
    "非常",
    "有点",
    "一下",
    "一下子",
    "感觉",
    "觉得",
    "时候",
    "大家",
    "可以",
    "不是",
    "如果",
    "什么",
    "怎么",
    "为什么",
    "而且",
    "然后",
    "还有",
    "一个人",
}

TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze keyword co-occurrence and semantic neighborhoods over time."
    )
    parser.add_argument(
        "--input_path",
        default="bert/artifacts/broad_analysis/analysis_base.parquet",
        help="Path or glob pattern for the analysis base table.",
    )
    parser.add_argument(
        "--output_dir",
        default="bert/artifacts/broad_analysis/semantic_analysis",
        help="Directory for co-occurrence and semantic-neighbor outputs.",
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
        help="Granularity used for temporal outputs.",
    )
    parser.add_argument(
        "--stopwords_path",
        default=None,
        help="Optional newline-delimited stopword list.",
    )
    parser.add_argument(
        "--token_min_length",
        type=int,
        default=2,
        help="Minimum token length after tokenization.",
    )
    parser.add_argument(
        "--min_doc_freq",
        type=int,
        default=5,
        help="Minimum document frequency for a candidate term.",
    )
    parser.add_argument(
        "--top_k_terms",
        type=int,
        default=50,
        help="Number of top co-occurrence terms to keep per keyword/period.",
    )
    parser.add_argument(
        "--top_k_neighbors",
        type=int,
        default=20,
        help="Number of top embedding neighbors to keep per keyword/period.",
    )
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=200,
        help="Candidate-term pool size before embedding ranking.",
    )
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
    return parser.parse_args()


def load_stopwords(path: str | None) -> set[str]:
    stopwords = set(DEFAULT_STOPWORDS)
    if not path:
        return stopwords

    content = Path(path).read_text(encoding="utf-8")
    for line in content.splitlines():
        normalized = line.strip()
        if normalized:
            stopwords.add(normalized)
    return stopwords


def load_tokenizer():
    try:
        import jieba
    except ImportError as exc:
        raise ImportError("jieba is not installed. Please install requirements and retry.") from exc
    return jieba


def load_sentence_encoder(args: argparse.Namespace):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. Please install requirements and retry."
        ) from exc

    return SentenceTransformer(
        args.embedding_model,
        local_files_only=bool(args.local_files_only),
    )


def tokenize_text(text: str, *, jieba_module, stopwords: set[str], token_min_length: int) -> list[str]:
    tokens: list[str] = []
    for token in jieba_module.lcut(text, cut_all=False):
        candidate = token.strip().lower()
        if not candidate:
            continue
        if candidate in stopwords:
            continue
        if len(candidate) < token_min_length:
            continue
        if not TOKEN_RE.fullmatch(candidate):
            continue
        tokens.append(candidate)
    return tokens


def build_reference_doc_freq(token_sets: list[set[str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for token_set in token_sets:
        counter.update(token_set)
    return counter


def score_terms(
    *,
    keyword: str,
    period_label: str,
    docs: pd.DataFrame,
    reference_docs: pd.DataFrame,
    top_k_terms: int,
    min_doc_freq: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if docs.empty or reference_docs.empty:
        return rows

    keyword_token_sets = docs["token_set"].tolist()
    reference_token_sets = reference_docs["token_set"].tolist()
    keyword_df = build_reference_doc_freq(keyword_token_sets)
    reference_df = build_reference_doc_freq(reference_token_sets)

    keyword_tf: Counter[str] = Counter()
    for tokens in docs["tokens"]:
        keyword_tf.update(tokens)

    docs_count = len(docs)
    reference_count = len(reference_docs)
    for term, doc_freq in keyword_df.items():
        if term == keyword:
            continue
        if term in normalize_cli_keywords(DEFAULT_ANALYSIS_KEYWORDS):
            continue
        if doc_freq < min_doc_freq:
            continue
        corpus_doc_freq = reference_df.get(term, 0)
        if corpus_doc_freq <= 0:
            continue
        pmi = math.log2((doc_freq * reference_count) / (docs_count * corpus_doc_freq))
        rows.append(
            {
                "keyword": keyword,
                "period_label": period_label,
                "doc_count_in_keyword": docs_count,
                "doc_count_in_reference": reference_count,
                "term": term,
                "term_doc_freq": int(doc_freq),
                "term_tf": int(keyword_tf.get(term, 0)),
                "term_doc_rate": float(doc_freq / docs_count),
                "reference_doc_freq": int(corpus_doc_freq),
                "reference_doc_rate": float(corpus_doc_freq / reference_count),
                "pmi": float(pmi),
                "lift": float((doc_freq / docs_count) / (corpus_doc_freq / reference_count)),
            }
        )

    rows.sort(key=lambda item: (item["pmi"], item["term_doc_freq"], item["term_tf"]), reverse=True)
    for rank, row in enumerate(rows[:top_k_terms], start=1):
        row["term_rank"] = rank
    return rows[:top_k_terms]


def build_semantic_neighbors(
    cooccurrence_df: pd.DataFrame,
    *,
    encoder,
    top_k_neighbors: int,
    candidate_pool_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if cooccurrence_df.empty:
        return pd.DataFrame(rows)

    grouped = cooccurrence_df.groupby(["keyword", "period_label"], dropna=False)
    for (keyword, period_label), frame in grouped:
        candidate_frame = frame.sort_values(
            ["pmi", "term_doc_freq", "term_tf"],
            ascending=[False, False, False],
        ).head(candidate_pool_size)
        candidates = candidate_frame["term"].astype(str).tolist()
        if not candidates:
            continue

        embeddings = encoder.encode([str(keyword)] + candidates, normalize_embeddings=True)
        keyword_vector = embeddings[0]
        candidate_vectors = embeddings[1:]
        similarities = candidate_vectors @ keyword_vector
        similarities = np.asarray(similarities, dtype=float)

        ranked = candidate_frame.copy()
        ranked["embedding_similarity"] = similarities
        ranked = ranked.sort_values(
            ["embedding_similarity", "pmi", "term_doc_freq"],
            ascending=[False, False, False],
        ).head(top_k_neighbors)

        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "keyword": keyword,
                    "period_label": period_label,
                    "neighbor_rank": rank,
                    "neighbor_term": row["term"],
                    "embedding_similarity": float(row["embedding_similarity"]),
                    "term_doc_freq": int(row["term_doc_freq"]),
                    "term_doc_rate": float(row["term_doc_rate"]),
                    "pmi": float(row["pmi"]),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    emit = resolve_emit("semantic", None)
    selected_keywords = normalize_cli_keywords(args.keywords)

    df, _ = load_tabular_files(args.input_path, emit=emit)
    for column in (args.text_col, args.keyword_col, args.time_col):
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found in analysis base.")

    working = df[df[args.keyword_col].isin(selected_keywords)].copy()
    working[args.text_col] = working[args.text_col].fillna("").astype("string").str.strip()
    working = working[working[args.text_col].ne("")].reset_index(drop=True)
    if working.empty:
        raise ValueError("No rows left after keyword filtering.")

    working["period_label"] = coerce_period_series(working[args.time_col], args.time_granularity)

    stopwords = load_stopwords(args.stopwords_path)
    jieba_module = load_tokenizer()

    emit("Tokenizing texts")
    working["tokens"] = working[args.text_col].map(
        lambda text: tokenize_text(
            str(text),
            jieba_module=jieba_module,
            stopwords=stopwords,
            token_min_length=args.token_min_length,
        )
    )
    working["token_set"] = working["tokens"].map(lambda tokens: set(tokens))

    cooccurrence_rows: list[dict[str, object]] = []
    period_labels = sort_period_labels(working["period_label"].astype(str).unique().tolist(), args.time_granularity)
    for period_label in ["ALL"] + period_labels:
        reference_docs = working if period_label == "ALL" else working[working["period_label"] == period_label]
        if reference_docs.empty:
            continue
        for keyword in selected_keywords:
            keyword_docs = reference_docs[reference_docs[args.keyword_col] == keyword]
            cooccurrence_rows.extend(
                score_terms(
                    keyword=keyword,
                    period_label=period_label,
                    docs=keyword_docs,
                    reference_docs=reference_docs,
                    top_k_terms=args.top_k_terms,
                    min_doc_freq=args.min_doc_freq,
                )
            )

    cooccurrence_df = pd.DataFrame(cooccurrence_rows)
    if not cooccurrence_df.empty:
        cooccurrence_df = cooccurrence_df.sort_values(
            ["keyword", "period_label", "term_rank"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    emit("Computing embedding neighbors")
    encoder = load_sentence_encoder(args)
    semantic_neighbors_df = build_semantic_neighbors(
        cooccurrence_df,
        encoder=encoder,
        top_k_neighbors=args.top_k_neighbors,
        candidate_pool_size=args.candidate_pool_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cooccurrence_path = output_dir / "keyword_cooccurrence.csv"
    neighbors_path = output_dir / "keyword_semantic_neighbors.csv"
    tokenized_path = output_dir / "tokenized_analysis_base.parquet"
    summary_path = output_dir / "semantic_analysis_summary.json"

    save_dataframe(cooccurrence_df, cooccurrence_path)
    save_dataframe(semantic_neighbors_df, neighbors_path)
    save_dataframe(working.drop(columns=["token_set"]), tokenized_path)

    summary = {
        "input_path": args.input_path,
        "output_dir": str(output_dir.resolve()),
        "selected_keywords": selected_keywords,
        "time_granularity": args.time_granularity,
        "cooccurrence_path": str(cooccurrence_path.resolve()),
        "semantic_neighbors_path": str(neighbors_path.resolve()),
        "tokenized_analysis_base_path": str(tokenized_path.resolve()),
        "doc_count": int(len(working)),
    }
    save_json(summary_path, summary)
    emit(f"Saved semantic-analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
