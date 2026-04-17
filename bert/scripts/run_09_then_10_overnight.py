#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_ANALYSIS_BASE = "bert/artifacts/broad_analysis/analysis_base.parquet"
DEFAULT_TOPIC_MODEL_DIR = "bert/artifacts/broad_analysis/topic_model_BAAI"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def emit(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[overnight-09-10] {timestamp} {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim incomplete trailing months, then run bert 09 and 10 sequentially."
    )
    parser.add_argument(
        "--analysis_base",
        default=DEFAULT_ANALYSIS_BASE,
        help="Path to analysis_base.parquet.",
    )
    parser.add_argument(
        "--topic_model_dir",
        default=DEFAULT_TOPIC_MODEL_DIR,
        help="Directory containing 08 topic-share outputs.",
    )
    parser.add_argument(
        "--max_month",
        required=True,
        help="Latest complete month to keep, formatted as YYYY-MM.",
    )
    parser.add_argument(
        "--python_bin",
        default=".venv/bin/python",
        help="Python interpreter used to launch downstream scripts.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Embedding device passed to 09.",
    )
    parser.add_argument(
        "--embedding_model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers model name or local path for 09.",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Directory for filtered inputs and state files. Defaults to a max_month-specific folder.",
    )
    parser.add_argument(
        "--semantic_output_dir",
        default=None,
        help="Output directory for 09. Defaults to a max_month-specific folder.",
    )
    parser.add_argument(
        "--drift_output_dir",
        default=None,
        help="Output directory for 10. Defaults to a max_month-specific folder.",
    )
    parser.add_argument(
        "--state_path",
        default=None,
        help="Optional JSON state file path. Defaults under run_dir.",
    )
    parser.add_argument(
        "--log_path",
        default=None,
        help="Optional log file path recorded in state metadata.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="top_n passed to 10_concept_drift_analysis.py.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_max_month(max_month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        month_start = pd.Timestamp(f"{max_month}-01")
    except Exception as exc:
        raise ValueError(f"Invalid --max_month '{max_month}'. Expected YYYY-MM.") from exc
    next_month = month_start + pd.offsets.MonthBegin(1)
    return month_start, next_month


def ensure_path(path_str: str | None, fallback: Path) -> Path:
    if not path_str:
        return fallback
    return Path(path_str)


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def save_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def count_rows_by_month(df: pd.DataFrame, time_col: str) -> dict[str, int]:
    months = pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").astype("string")
    counts = months.value_counts(dropna=True).sort_index()
    return {str(index): int(value) for index, value in counts.items()}


def filter_analysis_base(
    *,
    source_path: Path,
    output_path: Path,
    max_month: str,
    month_exclusive_upper: pd.Timestamp,
) -> dict[str, object]:
    emit(f"Loading analysis base from {source_path}")
    df = pd.read_parquet(source_path)
    if "publish_time" not in df.columns:
        raise ValueError("analysis base is missing required column 'publish_time'.")

    original_rows = len(df)
    original_months = count_rows_by_month(df, "publish_time")
    publish_time = pd.to_datetime(df["publish_time"], errors="coerce")
    kept_mask = publish_time.lt(month_exclusive_upper)
    filtered = df.loc[kept_mask].copy()
    if filtered.empty:
        raise ValueError("No rows remain after trimming incomplete months.")

    dropped_months = sorted(
        month for month in original_months.keys() if month >= str(month_exclusive_upper.to_period("M"))
    )
    filtered_rows = len(filtered)
    filtered_months = count_rows_by_month(filtered, "publish_time")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    emit(
        "Saving filtered analysis base "
        f"({filtered_rows}/{original_rows} rows kept, max_month={max_month}) to {output_path}"
    )
    filtered.to_parquet(output_path, index=False)

    return {
        "source_path": source_path,
        "output_path": output_path,
        "rows_before": int(original_rows),
        "rows_after": int(filtered_rows),
        "rows_dropped": int(original_rows - filtered_rows),
        "months_before": original_months,
        "months_after": filtered_months,
        "dropped_months": dropped_months,
    }


def filter_topic_share_file(source_path: Path, output_path: Path, max_month: str) -> dict[str, object]:
    emit(f"Filtering topic-share input {source_path.name}")
    df = pd.read_csv(source_path)
    if "period_label" not in df.columns:
        raise ValueError(f"{source_path} is missing required column 'period_label'.")

    original_rows = len(df)
    filtered = df[df["period_label"].astype(str) <= max_month].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    kept_periods = sorted(filtered["period_label"].astype(str).dropna().unique().tolist())
    return {
        "source_path": source_path,
        "output_path": output_path,
        "rows_before": int(original_rows),
        "rows_after": int(len(filtered)),
        "rows_dropped": int(original_rows - len(filtered)),
        "kept_periods": kept_periods,
    }


def build_topic_input_map(topic_model_dir: Path, run_dir: Path, max_month: str) -> dict[str, dict[str, object]]:
    mapping = {
        "topic_share_by_period": "topic_share_by_period.csv",
        "topic_share_by_period_and_keyword": "topic_share_by_period_and_keyword.csv",
        "topic_share_by_period_and_ip": "topic_share_by_period_and_ip.csv",
        "topic_share_by_period_and_ip_and_keyword": "topic_share_by_period_and_ip_and_keyword.csv",
    }
    outputs: dict[str, dict[str, object]] = {}
    trimmed_dir = run_dir / "trimmed_topic_inputs"
    for key, filename in mapping.items():
        source_path = topic_model_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Required topic-share input not found: {source_path}")
        output_path = trimmed_dir / filename
        outputs[key] = filter_topic_share_file(source_path, output_path, max_month)
    return outputs


def run_command(
    *,
    cmd: list[str],
    cwd: Path,
    state: dict,
    state_path: Path,
    stage_key: str,
) -> None:
    state["stage"] = stage_key
    state["current_command"] = cmd
    state["status"] = "running"
    save_state(state_path, state)
    emit(f"Running {' '.join(cmd)}")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    completed = subprocess.run(cmd, cwd=cwd, env=env, check=False)
    state["last_return_code"] = int(completed.returncode)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root()
    suffix = f"through_{args.max_month.replace('-', '_')}"

    run_dir = ensure_path(args.run_dir, repo_root / "bert" / "artifacts" / "broad_analysis" / f"overnight_09_10_{suffix}")
    semantic_output_dir = ensure_path(
        args.semantic_output_dir,
        repo_root / "bert" / "artifacts" / "broad_analysis" / f"semantic_analysis_{suffix}",
    )
    drift_output_dir = ensure_path(
        args.drift_output_dir,
        repo_root / "bert" / "artifacts" / "broad_analysis" / f"drift_analysis_{suffix}",
    )
    state_path = ensure_path(args.state_path, run_dir / "state.json")
    summary_path = run_dir / "summary.json"
    filtered_analysis_base = run_dir / f"analysis_base_{suffix}.parquet"

    analysis_base = (repo_root / args.analysis_base).resolve() if not Path(args.analysis_base).is_absolute() else Path(args.analysis_base)
    topic_model_dir = (repo_root / args.topic_model_dir).resolve() if not Path(args.topic_model_dir).is_absolute() else Path(args.topic_model_dir)
    if Path(args.python_bin).is_absolute():
        python_bin = Path(args.python_bin)
    else:
        # Keep the venv interpreter path intact instead of resolving the symlink
        # back to the system Python, otherwise the child process can lose the venv.
        python_bin = (repo_root / args.python_bin).absolute()

    month_start, month_exclusive_upper = parse_max_month(args.max_month)

    state = {
        "status": "starting",
        "stage": "initializing",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "max_month": args.max_month,
        "month_exclusive_upper": month_exclusive_upper,
        "analysis_base": analysis_base,
        "filtered_analysis_base": filtered_analysis_base,
        "topic_model_dir": topic_model_dir,
        "semantic_output_dir": semantic_output_dir,
        "drift_output_dir": drift_output_dir,
        "log_path": args.log_path,
        "time_granularity": "month",
        "device": args.device,
        "embedding_model": args.embedding_model,
    }
    save_state(state_path, state)

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        semantic_output_dir.mkdir(parents=True, exist_ok=True)
        drift_output_dir.mkdir(parents=True, exist_ok=True)

        if not analysis_base.exists():
            raise FileNotFoundError(f"analysis base not found: {analysis_base}")
        if not topic_model_dir.exists():
            raise FileNotFoundError(f"topic model dir not found: {topic_model_dir}")
        if not python_bin.exists():
            raise FileNotFoundError(f"python interpreter not found: {python_bin}")

        state["stage"] = "filtering_inputs"
        state["status"] = "running"
        save_state(state_path, state)

        analysis_base_summary = filter_analysis_base(
            source_path=analysis_base,
            output_path=filtered_analysis_base,
            max_month=args.max_month,
            month_exclusive_upper=month_exclusive_upper,
        )
        topic_input_summary = build_topic_input_map(topic_model_dir, run_dir, args.max_month)

        state["trimmed_inputs"] = {
            "analysis_base": analysis_base_summary,
            "topic_share_files": topic_input_summary,
        }
        save_state(state_path, state)

        semantic_cmd = [
            str(python_bin),
            str((repo_root / "bert" / "09_keyword_semantic_analysis.py").resolve()),
            "--input_path",
            str(filtered_analysis_base),
            "--output_dir",
            str(semantic_output_dir),
            "--time_granularity",
            "month",
            "--embedding_model",
            args.embedding_model,
            "--device",
            args.device,
            "--local_files_only",
        ]
        run_command(
            cmd=semantic_cmd,
            cwd=repo_root,
            state=state,
            state_path=state_path,
            stage_key="running_09_semantic_analysis",
        )

        drift_cmd = [
            str(python_bin),
            str((repo_root / "bert" / "10_concept_drift_analysis.py").resolve()),
            "--cooccurrence_path",
            str(semantic_output_dir / "keyword_cooccurrence.csv"),
            "--neighbor_path",
            str(semantic_output_dir / "keyword_semantic_neighbors.csv"),
            "--topic_share_path",
            str((run_dir / "trimmed_topic_inputs" / "topic_share_by_period_and_keyword.csv").resolve()),
            "--overall_topic_share_path",
            str((run_dir / "trimmed_topic_inputs" / "topic_share_by_period.csv").resolve()),
            "--topic_share_by_period_and_ip_path",
            str((run_dir / "trimmed_topic_inputs" / "topic_share_by_period_and_ip.csv").resolve()),
            "--topic_share_by_period_and_ip_and_keyword_path",
            str((run_dir / "trimmed_topic_inputs" / "topic_share_by_period_and_ip_and_keyword.csv").resolve()),
            "--output_dir",
            str(drift_output_dir),
            "--time_granularity",
            "month",
            "--top_n",
            str(args.top_n),
        ]
        run_command(
            cmd=drift_cmd,
            cwd=repo_root,
            state=state,
            state_path=state_path,
            stage_key="running_10_concept_drift",
        )

        state["status"] = "completed"
        state["stage"] = "completed"
        state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        state["current_command"] = None
        save_state(state_path, state)

        summary = {
            "status": "completed",
            "started_at": state["started_at"],
            "finished_at": state["finished_at"],
            "max_month": args.max_month,
            "month_start": month_start,
            "month_exclusive_upper": month_exclusive_upper,
            "run_dir": run_dir,
            "state_path": state_path,
            "log_path": args.log_path,
            "filtered_analysis_base": filtered_analysis_base,
            "semantic_output_dir": semantic_output_dir,
            "drift_output_dir": drift_output_dir,
            "trimmed_inputs": state["trimmed_inputs"],
        }
        save_state(summary_path, summary)
        emit("Overnight pipeline completed successfully")
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["stage"] = "failed"
        state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        state["current_command"] = None
        state["error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_state(state_path, state)
        emit(f"Pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
