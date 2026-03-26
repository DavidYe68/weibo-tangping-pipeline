#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TIME_CANDIDATES = ["发布时间", "created_at", "publish_time", "timestamp"]
KEYWORD_CANDIDATES = ["keyword", "hit_keyword", "query_keyword"]
TEXT_CANDIDATES = [
    "cleaned_text",
    "cleaned_text_with_emoji",
    "text_raw",
    "微博正文",
    "text",
    "content",
    "body",
    "message",
    "post_text",
    "desc",
    "description",
    "title",
]
def emit_progress(message: str) -> None:
    print(f"[sample] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified sampling from parquet files with exact sample size."
    )
    parser.add_argument("--input", default="data/processed/text_dedup/*.parquet", help="Input parquet glob pattern")
    parser.add_argument("--output", default="data/bert/sample.csv", help="Output CSV path")
    parser.add_argument("--n", type=int, default=6000, help="Target sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--k_min",
        type=int,
        default=20,
        help="Minimum stratum population to guarantee at least one sample; 0 disables the guarantee",
    )
    parser.add_argument("--text_col", default=None, help="Optional forced text column name")
    parser.add_argument(
        "--report_path",
        default="data/bert/sampling_report.json",
        help="Sampling report JSON path",
    )
    return parser.parse_args()


def resolve_input_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    if not files and pattern.startswith("/"):
        files = sorted(glob.glob(pattern.lstrip("/")))
    if not files:
        raise FileNotFoundError(f"No parquet files matched input pattern: {pattern}")
    return files


def load_parquet_files(pattern: str) -> pd.DataFrame:
    files = resolve_input_files(pattern)
    frames = []
    for fp in files:
        try:
            frames.append(pd.read_parquet(fp))
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet file: {fp}. Error: {e}") from e
    if not frames:
        raise RuntimeError("No data loaded from parquet files.")
    return pd.concat(frames, axis=0, ignore_index=True)


def pick_text_col(df: pd.DataFrame, forced: Optional[str]) -> str:
    if forced:
        if forced not in df.columns:
            raise ValueError(f"--text_col '{forced}' not found in data columns.")
        return forced

    for col in TEXT_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "Cannot detect text column automatically. Please provide --text_col explicitly."
    )


def detect_month_series(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    for col in TIME_CANDIDATES:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid = parsed.notna().sum()
        if valid > 0:
            month = parsed.dt.to_period("M").astype("string")
            month = month.fillna("NA")
            return month, col
    return None, None


def detect_keyword_series(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    for col in KEYWORD_CANDIDATES:
        if col not in df.columns:
            continue
        s = df[col].astype("string")
        usable = ((s.notna()) & (s.str.strip() != "")).sum()
        if usable > 0:
            normalized = s.fillna("").str.strip()
            return normalized.mask(normalized == "", "NA"), col
    return None, None


def build_len_bin(text_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    lengths = text_series.fillna("").astype(str).str.len()
    bins = np.select(
        [lengths <= 20, (lengths >= 21) & (lengths <= 60), (lengths >= 61) & (lengths <= 140), lengths > 140],
        ["<=20", "21-60", "61-140", ">140"],
        default=">140",
    )
    return pd.Series(bins, index=text_series.index), lengths


def allocate_samples(
    counts: pd.Series,
    n: int,
    k_min: int,
) -> pd.Series:
    total = int(counts.sum())
    if n <= 0:
        raise ValueError("--n must be a positive integer.")
    if total < n:
        raise ValueError(
            f"Not enough rows to sample exactly n={n}. Available rows after dedup: {total}."
        )

    exact = counts / total * n
    alloc = np.floor(exact).astype(int)

    remainder = int(n - alloc.sum())
    if remainder > 0:
        frac = (exact - alloc).sort_values(ascending=False)
        for idx in frac.index[:remainder]:
            alloc.loc[idx] += 1

    eligible_min1 = counts >= k_min if k_min > 0 else pd.Series(False, index=counts.index)
    alloc = alloc.copy()
    alloc.loc[(eligible_min1) & (alloc == 0)] = 1

    alloc = np.minimum(alloc, counts)

    min_allowed = pd.Series(0, index=counts.index, dtype=int)
    min_allowed.loc[eligible_min1] = 1

    diff = int(n - alloc.sum())

    if diff > 0:
        order = counts.sort_values(ascending=False).index.tolist()
        i = 0
        max_iters = len(order) * (diff + 2)
        iters = 0
        while diff > 0 and iters < max_iters:
            stratum = order[i % len(order)]
            if alloc.loc[stratum] < counts.loc[stratum]:
                alloc.loc[stratum] += 1
                diff -= 1
            i += 1
            iters += 1
        if diff != 0:
            raise RuntimeError("Failed to distribute positive tail difference to reach target n.")

    elif diff < 0:
        remove = -diff
        order = alloc.sort_values(ascending=False).index.tolist()
        i = 0
        max_iters = len(order) * (remove + 2)
        iters = 0
        while remove > 0 and iters < max_iters:
            stratum = order[i % len(order)]
            if alloc.loc[stratum] > min_allowed.loc[stratum]:
                alloc.loc[stratum] -= 1
                remove -= 1
            i += 1
            iters += 1
        if remove != 0:
            raise RuntimeError(
                "Failed to reduce allocation to target n under k_min constraints. "
                "Try --k_min 0 to disable the guarantee, or increase --k_min / --n."
            )

    if int(alloc.sum()) != n:
        raise RuntimeError(
            f"Allocation sum mismatch: expected {n}, got {int(alloc.sum())}."
        )

    return alloc.astype(int)


def sample_by_stratum(
    df: pd.DataFrame,
    stratum_col: str,
    alloc: pd.Series,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    picked_indices: List[int] = []

    groups = df.groupby(stratum_col, sort=False)
    for stratum, group in groups:
        k = int(alloc.get(stratum, 0))
        if k <= 0:
            continue
        idx = group.index.to_numpy()
        if k >= len(idx):
            picked_indices.extend(idx.tolist())
        else:
            chosen = rng.choice(idx, size=k, replace=False)
            picked_indices.extend(chosen.tolist())

    if len(picked_indices) != int(alloc.sum()):
        raise RuntimeError(
            f"Sampled row count mismatch: expected {int(alloc.sum())}, got {len(picked_indices)}"
        )

    sampled = df.loc[picked_indices].copy()
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled


def series_count_dict(s: pd.Series) -> Dict[str, int]:
    vc = s.astype("string").fillna("NA").value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    args = parse_args()

    try:
        emit_progress(f"load input pattern={args.input}")
        df_raw = load_parquet_files(args.input)
        raw_rows = int(len(df_raw))

        emit_progress("detect text column")
        text_col = pick_text_col(df_raw, args.text_col)

        emit_progress("build stratification dimensions")
        month_series, month_col = detect_month_series(df_raw)
        keyword_series, keyword_col = detect_keyword_series(df_raw)
        len_bin_series, lengths = build_len_bin(df_raw[text_col])

        work = df_raw.copy()
        work["__month"] = month_series if month_series is not None else "NA"
        work["__keyword"] = keyword_series if keyword_series is not None else "NA"
        work["__len_bin"] = len_bin_series
        work["__text_len"] = lengths

        work["__stratum_id"] = (
            work["__month"].astype("string").fillna("NA")
            + "|"
            + work["__keyword"].astype("string").fillna("NA")
            + "|"
            + work["__len_bin"].astype("string").fillna("NA")
        )

        counts = work["__stratum_id"].value_counts(sort=False)

        emit_progress("allocate sample counts")
        alloc = allocate_samples(counts=counts, n=args.n, k_min=args.k_min)

        emit_progress("sample rows by stratum")
        sampled = sample_by_stratum(
            df=work,
            stratum_col="__stratum_id",
            alloc=alloc,
            seed=args.seed,
        )

        if len(sampled) != args.n:
            raise RuntimeError(
                f"Final sampled size mismatch: expected {args.n}, got {len(sampled)}"
            )

        output_df = sampled[df_raw.columns].copy()

        ensure_parent_dir(args.output)
        emit_progress(f"write sampled csv={args.output}")
        output_df.to_csv(args.output, index=False, encoding="utf-8")

        stratum_report = {}
        for sid, total_cnt in counts.items():
            stratum_report[str(sid)] = {
                "population_count": int(total_cnt),
                "sample_count": int(alloc.get(sid, 0)),
            }

        used_dims = ["len_bin"]
        if month_col is not None:
            used_dims.insert(0, "month")
        if keyword_col is not None:
            if "month" in used_dims:
                used_dims.insert(1, "keyword")
            else:
                used_dims.insert(0, "keyword")

        report = {
            "input_pattern": args.input,
            "output_csv": args.output,
            "input_rows": raw_rows,
            "dedup_applied_in_sampling": False,
            "sample_n": int(args.n),
            "seed": int(args.seed),
            "k_min": int(args.k_min),
            "text_col": text_col,
            "used_stratification_dims": used_dims,
            "detected_columns": {
                "month_source_col": month_col,
                "keyword_source_col": keyword_col,
            },
            "strata": stratum_report,
            "sample_distributions": {
                "len_bin": series_count_dict(sampled["__len_bin"]),
            },
        }

        if month_col is not None:
            report["sample_distributions"]["month"] = series_count_dict(sampled["__month"])
        if keyword_col is not None:
            report["sample_distributions"]["keyword"] = series_count_dict(sampled["__keyword"])

        ensure_parent_dir(args.report_path)
        emit_progress(f"write sampling report={args.report_path}")
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Done. Saved sampled CSV to: {args.output}")
        print(f"Done. Saved sampling report to: {args.report_path}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
