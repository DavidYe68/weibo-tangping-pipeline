#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent


def bootstrap_local_site_packages() -> None:
    lib_dirs = [
        ROOT / "scripts" / "AI" / "lib",
        ROOT / "AI" / "lib",
    ]
    for lib_dir in lib_dirs:
        if not lib_dir.exists():
            continue
        for candidate in sorted(lib_dir.glob("python*/site-packages")):
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)


bootstrap_local_site_packages()

import pandas as pd

try:
    import emoji
except ImportError:
    emoji = None

RAW_DIR = ROOT / "raw"
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MERGED_DIR = PROCESSED_DIR / "merged_dedup"
PREPROCESSED_DIR = PROCESSED_DIR / "preprocessed"
TEXT_DEDUP_DIR = PROCESSED_DIR / "text_dedup"
EXPORTS_DIR = DATA_DIR / "exports"
MERGED_EXPORT_PATH = EXPORTS_DIR / "merged_dedup.csv"
TEXT_DEDUP_EXPORT_PATH = EXPORTS_DIR / "text_dedup.csv"
STATE_DIR = DATA_DIR / "state"
MANIFEST_PATH = STATE_DIR / "raw_manifest.json"
ID_HASH_PATH = STATE_DIR / "id_hashes.txt"
TEXT_HASH_PATH = STATE_DIR / "text_hashes.txt"
REPORTS_DIR = DATA_DIR / "reports"
RUN_REPORT_PATH = REPORTS_DIR / "pipeline_last_run.json"

CHUNK_SIZE = 200_000
WRITE_BUFFER_ROWS = 200_000
MIN_LEN = 2
HARD_CUT = 2000

MERGED_COLUMNS = [
    "id",
    "微博正文",
    "发布时间",
    "话题",
    "keyword",
    "转发数",
    "评论数",
    "点赞数",
    "ip",
    "source_file",
]

PREPROCESSED_COLUMNS = [
    "id",
    "cleaned_text",
    "cleaned_text_with_emoji",
    "text_raw",
    "发布时间",
    "话题",
    "keyword",
    "转发数",
    "评论数",
    "点赞数",
    "ip",
    "source_file",
]

_EMOJI_RANGES = [
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FA6F),
    (0x1FA70, 0x1FAFF),
    (0xFE00, 0xFE0F),
]


@dataclass(frozen=True)
class RawCsvFile:
    path: Path
    rel_path: str
    keyword: str
    size: int
    mtime_ns: int
    sort_key: tuple[int, int, int, str, str]


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_progress(message: str) -> None:
    print(f"[pipeline] {message}", file=sys.stderr, flush=True)


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DEDUP_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_fancy_unicode_to_ascii_table() -> dict[int, int]:
    table: dict[int, int] = {}
    for i in range(0xFF01, 0xFF5F):
        table[i] = 0x21 + (i - 0xFF01)

    blocks = [
        (0x1D400, 26, 0x41),
        (0x1D41A, 26, 0x61),
        (0x1D434, 26, 0x41),
        (0x1D44E, 26, 0x61),
        (0x1D468, 26, 0x41),
        (0x1D482, 26, 0x61),
        (0x1D49C, 26, 0x41),
        (0x1D4B6, 26, 0x61),
        (0x1D4D0, 26, 0x41),
        (0x1D4EA, 26, 0x61),
        (0x1D504, 26, 0x41),
        (0x1D51E, 26, 0x61),
        (0x1D538, 26, 0x41),
        (0x1D552, 26, 0x61),
        (0x1D56C, 26, 0x41),
        (0x1D586, 26, 0x61),
        (0x1D5A0, 26, 0x41),
        (0x1D5BA, 26, 0x61),
        (0x1D5D4, 26, 0x41),
        (0x1D5EE, 26, 0x61),
        (0x1D608, 26, 0x41),
        (0x1D622, 26, 0x61),
        (0x1D63C, 26, 0x41),
        (0x1D656, 26, 0x61),
        (0x1D670, 26, 0x41),
        (0x1D68A, 26, 0x61),
        (0x1D7CE, 10, 0x30),
        (0x1D7D8, 10, 0x30),
        (0x1D7E2, 10, 0x30),
        (0x1D7EC, 10, 0x30),
    ]
    for start, count, base in blocks:
        for i in range(count):
            table[start + i] = base + i

    superscript_map = (0x2070, 0x00B9, 0x00B2, 0x00B3, 0x2074, 0x2075, 0x2076, 0x2077, 0x2078, 0x2079)
    for i, code in enumerate(superscript_map):
        table[code] = 0x30 + i
    for i in range(10):
        table[0x2080 + i] = 0x30 + i
    return table


_FANCY_UNICODE_TABLE = _build_fancy_unicode_to_ascii_table()


def normalize_font(s: str) -> str:
    if not s:
        return s
    return s.translate(_FANCY_UNICODE_TABLE)


def _is_emoji_codepoint(code: int) -> bool:
    return any(lo <= code <= hi for lo, hi in _EMOJI_RANGES)


def _emoji_fallback(emj: str) -> str:
    if not emj:
        return "[emoji]"
    c = emj[0]
    try:
        name = unicodedata.name(c, "").lower().replace(" ", "_")
        return f":{name}:" if name else "[emoji]"
    except Exception:
        return "[emoji]"


def _replace_emoji_ranges(s: str, replacement_fn) -> str:
    if not s:
        return s
    result: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        code = ord(c)
        if code in (0xFE0F, 0xFE0E):
            i += 1
            continue
        if _is_emoji_codepoint(code):
            end = i + 1
            while end < len(s):
                n = ord(s[end])
                if n in (0xFE0F, 0xFE0E) or 0x1F3FB <= n <= 0x1F3FF:
                    end += 1
                else:
                    break
            result.append(replacement_fn(s[i:end]))
            i = end
        else:
            result.append(c)
            i += 1
    return "".join(result)


def emoji_to_placeholder(s: str) -> str:
    if not s:
        return s
    if emoji is not None:
        s = emoji.replace_emoji(s, replace="[emoji]")
    return _replace_emoji_ranges(s, lambda _: "[emoji]")


def emoji_to_text(s: str) -> str:
    if not s:
        return s
    if emoji is not None:

        def _replace(emj: str, data: dict[str, Any] | None) -> str:
            if data and "en" in data:
                return data["en"]
            return _emoji_fallback(emj)

        s = emoji.replace_emoji(s, replace=_replace)
    return _replace_emoji_ranges(s, _emoji_fallback)


def clean_weibo_text_base(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    s = normalize_font(raw.strip())
    s = re.sub(r"https?://\S+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"L[^\s]*的微博视频\s*", "", s)
    s = re.sub(r"网页链接\s*", "", s)
    s = re.sub(r"全文\s*$", "", s)
    s = re.sub(r"[\s\u3000]+", " ", s).strip()
    return s


def extract_keyword_and_sort(rel_to_raw: Path) -> tuple[str, tuple[int, int, int, str, str]] | None:
    parts = rel_to_raw.parts
    csv_idx = -1
    for idx, part in enumerate(parts):
        if part.lower() == "csv":
            csv_idx = idx
            break
    if csv_idx < 0 or csv_idx + 1 >= len(parts):
        return None

    keyword = parts[csv_idx + 1]
    year = 9999
    month = 99
    day = 99

    if csv_idx + 2 < len(parts) and parts[csv_idx + 2].isdigit():
        year = int(parts[csv_idx + 2])
    if csv_idx + 3 < len(parts) and parts[csv_idx + 3].isdigit():
        month = int(parts[csv_idx + 3])

    stem = Path(parts[-1]).stem
    if stem.isdigit():
        day = int(stem)

    return keyword, (year, month, day, keyword, str(rel_to_raw))


def discover_raw_csv_files(raw_dir: Path = RAW_DIR) -> list[RawCsvFile]:
    if not raw_dir.exists():
        return []

    discovered: list[RawCsvFile] = []
    for f in raw_dir.rglob("*.csv"):
        if ".backup" in f.name:
            continue
        rel_to_raw = f.relative_to(raw_dir)
        parsed = extract_keyword_and_sort(rel_to_raw)
        if parsed is None:
            continue
        keyword, sort_key = parsed
        stat = f.stat()
        rel_to_root = f.relative_to(ROOT)
        discovered.append(
            RawCsvFile(
                path=f,
                rel_path=str(rel_to_root),
                keyword=keyword,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sort_key=sort_key,
            )
        )
    discovered.sort(key=lambda x: x.sort_key)
    return discovered


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload.get("files", {})
        if isinstance(files, dict):
            return files
    except Exception:
        return {}
    return {}


def save_manifest(files_map: dict[str, dict[str, Any]], path: Path = MANIFEST_PATH) -> None:
    payload = {
        "version": 1,
        "updated_at": now_iso_utc(),
        "files": files_map,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def delta_files(discovered: list[RawCsvFile], old_manifest: dict[str, dict[str, Any]]) -> tuple[list[RawCsvFile], set[str]]:
    delta: list[RawCsvFile] = []
    unchanged: set[str] = set()
    for f in discovered:
        prev = old_manifest.get(f.rel_path)
        if prev and prev.get("size") == f.size and prev.get("mtime_ns") == f.mtime_ns:
            unchanged.add(f.rel_path)
            continue
        delta.append(f)
    return delta, unchanged


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, on_bad_lines="skip", low_memory=False)
        except Exception as e:
            errors.append(f"{enc}: {e}")
    raise RuntimeError("; ".join(errors))


def _has_parquet_support() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        bootstrap_local_site_packages()
        try:
            import pyarrow  # noqa: F401

            return True
        except Exception:
            pass
        try:
            import fastparquet  # noqa: F401

            return True
        except Exception:
            return False


def require_parquet_support() -> None:
    if not _has_parquet_support():
        raise RuntimeError("当前流程已切换为全 parquet 存储，请先安装 `pyarrow` 或 `fastparquet`。")


def parquet_parts(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        return []
    return sorted(dataset_dir.glob("part-*.parquet"))


def next_part_index(dataset_dir: Path) -> int:
    max_idx = -1
    pattern = re.compile(r"^part-(\d+)\.parquet$")
    for f in parquet_parts(dataset_dir):
        m = pattern.match(f.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def write_parquet_parts(df: pd.DataFrame, dataset_dir: Path, start_idx: int | None = None) -> tuple[list[str], int]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        next_idx = start_idx if start_idx is not None else next_part_index(dataset_dir)
        return [], next_idx

    part_idx = next_part_index(dataset_dir) if start_idx is None else start_idx
    written: list[str] = []
    for start in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[start : start + CHUNK_SIZE].copy()
        out = dataset_dir / f"part-{part_idx:05d}.parquet"
        chunk.to_parquet(out, index=False)
        written.append(str(out.relative_to(ROOT)))
        part_idx += 1
    return written, part_idx


def flush_parquet_buffer(
    frames: list[pd.DataFrame],
    dataset_dir: Path,
    part_idx: int,
) -> tuple[list[str], int]:
    if not frames:
        return [], part_idx
    merged = pd.concat(frames, ignore_index=True)
    frames.clear()
    return write_parquet_parts(merged, dataset_dir, part_idx)


def export_parquet_dataset_to_csv(dataset_dir: Path, csv_path: Path) -> int:
    parts = parquet_parts(dataset_dir)
    if not parts:
        return 0

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()

    total_rows = 0
    first = True
    for part in parts:
        df = pd.read_parquet(part)
        if df.empty:
            continue
        df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
            mode="w" if first else "a",
            header=first,
        )
        first = False
        total_rows += len(df)
    return total_rows


def load_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(line)
    return seen


def append_hashes(hashes: list[str], path: Path) -> None:
    if not hashes:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for h in hashes:
            f.write(h + "\n")


def _text_hash(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def normalize_raw_df(df: pd.DataFrame, source_keyword: str, source_file: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=MERGED_COLUMNS)
    work = df.copy()
    work["keyword"] = source_keyword
    work["source_file"] = source_file
    for col in MERGED_COLUMNS:
        if col not in work.columns:
            work[col] = ""
    return work[MERGED_COLUMNS].copy()


def dedup_new_rows_by_id_hash(df: pd.DataFrame, seen_id_hashes: set[str]) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    stats = {
        "incoming_rows": int(len(df)),
        "missing_id_rows": 0,
        "batch_dedup_removed": 0,
        "existing_dedup_removed": 0,
        "appended_rows": 0,
    }
    if df.empty:
        return pd.DataFrame(columns=MERGED_COLUMNS), stats, []

    work = df.copy()
    work["id"] = work["id"].fillna("").astype(str).str.strip()

    missing_mask = work["id"] == ""
    stats["missing_id_rows"] = int(missing_mask.sum())
    work = work[~missing_mask].copy()
    if work.empty:
        return pd.DataFrame(columns=MERGED_COLUMNS), stats, []

    before_batch = len(work)
    work = work.drop_duplicates(subset=["id"], keep="first")
    stats["batch_dedup_removed"] = int(before_batch - len(work))

    keep_index: list[int] = []
    new_id_hashes: list[str] = []
    for idx, val in work["id"].items():
        h = _text_hash(val)
        if not h or h in seen_id_hashes:
            continue
        seen_id_hashes.add(h)
        keep_index.append(idx)
        new_id_hashes.append(h)

    deduped = work.loc[keep_index].copy()
    stats["existing_dedup_removed"] = int(len(work) - len(deduped))
    stats["appended_rows"] = int(len(deduped))
    return deduped, stats, new_id_hashes


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PREPROCESSED_COLUMNS)

    work = df.copy()
    work["text_raw"] = work["微博正文"].fillna("").astype(str)
    work = work[work["text_raw"].str.strip() != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=PREPROCESSED_COLUMNS)

    base = work["text_raw"].map(clean_weibo_text_base)
    work["cleaned_text"] = base.map(emoji_to_placeholder)
    work["cleaned_text_with_emoji"] = base.map(emoji_to_text)

    work = work[work["cleaned_text"].str.len() >= MIN_LEN].copy()
    if work.empty:
        return pd.DataFrame(columns=PREPROCESSED_COLUMNS)

    work["cleaned_text"] = work["cleaned_text"].str.slice(0, HARD_CUT)
    work["cleaned_text_with_emoji"] = work["cleaned_text_with_emoji"].str.slice(0, HARD_CUT)

    for col in PREPROCESSED_COLUMNS:
        if col not in work.columns:
            work[col] = ""
    return work[PREPROCESSED_COLUMNS].copy()


def dedup_text_with_seen(pre_df: pd.DataFrame, seen: set[str]) -> tuple[pd.DataFrame, list[str], int, int]:
    if pre_df.empty:
        return pd.DataFrame(columns=PREPROCESSED_COLUMNS), [], 0, 0

    keep_idx: list[int] = []
    new_hashes: list[str] = []
    cleaned = pre_df["cleaned_text"].fillna("").astype(str)
    for idx, txt in cleaned.items():
        h = _text_hash(txt)
        if not h or h in seen:
            continue
        seen.add(h)
        keep_idx.append(idx)
        new_hashes.append(h)
    out_df = pre_df.loc[keep_idx].copy()
    return out_df, new_hashes, int(len(pre_df)), int(len(out_df))


def write_run_report(report: dict[str, Any], path: Path = RUN_REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_last_run_report(path: Path = RUN_REPORT_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def reset_outputs() -> None:
    for path in [MERGED_DIR, PREPROCESSED_DIR, TEXT_DEDUP_DIR, EXPORTS_DIR]:
        if path.exists():
            shutil.rmtree(path)

    for path in [ID_HASH_PATH, TEXT_HASH_PATH, MANIFEST_PATH, RUN_REPORT_PATH]:
        if path.exists():
            path.unlink()

    ensure_dirs()


def run_pipeline(full_rebuild: bool = False) -> dict[str, Any]:
    require_parquet_support()
    ensure_dirs()
    started_at = now_iso_utc()
    mode_name = "full" if full_rebuild else "incremental"
    emit_progress(f"start mode={mode_name}")

    if full_rebuild:
        emit_progress("reset processed/state/reports/exports")
        reset_outputs()

    previous_report = {} if full_rebuild else load_last_run_report(RUN_REPORT_PATH)
    previous_totals = previous_report.get("totals", {}) if isinstance(previous_report, dict) else {}
    previous_merged_rows = int(previous_totals.get("merged_rows", 0) or 0)
    previous_text_rows = int(previous_totals.get("text_dedup_rows", 0) or 0)

    discovered = discover_raw_csv_files(RAW_DIR)
    old_manifest = load_manifest(MANIFEST_PATH)
    emit_progress(f"discovered raw_files={len(discovered)}")
    if full_rebuild:
        target_files = discovered
        unchanged: set[str] = set()
    else:
        target_files, unchanged = delta_files(discovered, old_manifest)
    emit_progress(f"target raw_files={len(target_files)} unchanged={len(unchanged)}")

    seen_id_hashes = load_hashes(ID_HASH_PATH)
    seen_text_hashes = load_hashes(TEXT_HASH_PATH)

    merged_part_idx = next_part_index(MERGED_DIR)
    preprocessed_part_idx = next_part_index(PREPROCESSED_DIR)
    text_part_idx = next_part_index(TEXT_DEDUP_DIR)

    ok_set: set[str] = set()
    rows_read = 0
    id_stats_total = {
        "incoming_rows": 0,
        "missing_id_rows": 0,
        "batch_dedup_removed": 0,
        "existing_dedup_removed": 0,
        "appended_rows": 0,
    }
    merged_parts_written: list[str] = []
    preprocessed_parts_written: list[str] = []
    text_parts_written: list[str] = []
    rows_preprocessed_written = 0
    text_before_total = 0
    text_after_total = 0
    merged_buffer: list[pd.DataFrame] = []
    preprocessed_buffer: list[pd.DataFrame] = []
    text_buffer: list[pd.DataFrame] = []
    merged_buffer_rows = 0
    preprocessed_buffer_rows = 0
    text_buffer_rows = 0

    processed_files = 0
    progress_every = max(1, len(target_files) // 20) if target_files else 1

    for raw_file in target_files:
        try:
            raw_df = read_csv_with_fallback(raw_file.path)
        except Exception:
            continue

        ok_set.add(raw_file.rel_path)
        if raw_df.empty:
            processed_files += 1
            if processed_files == len(target_files) or processed_files % progress_every == 0:
                emit_progress(
                    "processed "
                    f"files={processed_files}/{len(target_files)} "
                    f"rows_read={rows_read} merged_appended={id_stats_total['appended_rows']} "
                    f"text_dedup_appended={text_after_total}"
                )
            continue

        normalized_df = normalize_raw_df(raw_df, raw_file.keyword, raw_file.rel_path)
        rows_read += len(normalized_df)

        dedup_df, id_stats, new_id_hashes = dedup_new_rows_by_id_hash(normalized_df, seen_id_hashes)
        for key in id_stats_total:
            id_stats_total[key] += int(id_stats.get(key, 0))
        if dedup_df.empty:
            processed_files += 1
            if processed_files == len(target_files) or processed_files % progress_every == 0:
                emit_progress(
                    "processed "
                    f"files={processed_files}/{len(target_files)} "
                    f"rows_read={rows_read} merged_appended={id_stats_total['appended_rows']} "
                    f"text_dedup_appended={text_after_total}"
                )
            continue

        merged_buffer.append(dedup_df)
        merged_buffer_rows += len(dedup_df)
        if merged_buffer_rows >= WRITE_BUFFER_ROWS:
            written, merged_part_idx = flush_parquet_buffer(merged_buffer, MERGED_DIR, merged_part_idx)
            merged_parts_written.extend(written)
            merged_buffer_rows = 0
        append_hashes(new_id_hashes, ID_HASH_PATH)

        pre_df = preprocess_dataframe(dedup_df)
        rows_preprocessed_written += int(len(pre_df))
        if not pre_df.empty:
            preprocessed_buffer.append(pre_df)
            preprocessed_buffer_rows += len(pre_df)
            if preprocessed_buffer_rows >= WRITE_BUFFER_ROWS:
                written, preprocessed_part_idx = flush_parquet_buffer(preprocessed_buffer, PREPROCESSED_DIR, preprocessed_part_idx)
                preprocessed_parts_written.extend(written)
                preprocessed_buffer_rows = 0

        dedup_text_df, new_text_hashes, text_before, text_after = dedup_text_with_seen(pre_df, seen_text_hashes)
        text_before_total += text_before
        text_after_total += text_after
        if not dedup_text_df.empty:
            text_buffer.append(dedup_text_df)
            text_buffer_rows += len(dedup_text_df)
            if text_buffer_rows >= WRITE_BUFFER_ROWS:
                written, text_part_idx = flush_parquet_buffer(text_buffer, TEXT_DEDUP_DIR, text_part_idx)
                text_parts_written.extend(written)
                text_buffer_rows = 0
        append_hashes(new_text_hashes, TEXT_HASH_PATH)
        processed_files += 1
        if processed_files == len(target_files) or processed_files % progress_every == 0:
            emit_progress(
                "processed "
                f"files={processed_files}/{len(target_files)} "
                f"rows_read={rows_read} merged_appended={id_stats_total['appended_rows']} "
                f"text_dedup_appended={text_after_total}"
            )

    emit_progress("flushing parquet buffers")
    written, merged_part_idx = flush_parquet_buffer(merged_buffer, MERGED_DIR, merged_part_idx)
    merged_parts_written.extend(written)
    written, preprocessed_part_idx = flush_parquet_buffer(preprocessed_buffer, PREPROCESSED_DIR, preprocessed_part_idx)
    preprocessed_parts_written.extend(written)
    written, text_part_idx = flush_parquet_buffer(text_buffer, TEXT_DEDUP_DIR, text_part_idx)
    text_parts_written.extend(written)

    discovered_map = {f.rel_path: f for f in discovered}
    new_manifest: dict[str, dict[str, Any]] = {}
    for rel, f in discovered_map.items():
        if rel in unchanged or rel in ok_set:
            new_manifest[rel] = {
                "size": f.size,
                "mtime_ns": f.mtime_ns,
                "keyword": f.keyword,
            }
        elif rel in old_manifest:
            new_manifest[rel] = old_manifest[rel]
    emit_progress("writing manifest and run report")
    save_manifest(new_manifest, MANIFEST_PATH)

    report = {
        "mode": "full" if full_rebuild else "incremental",
        "started_at": started_at,
        "ended_at": now_iso_utc(),
        "raw_files_total": len(discovered),
        "raw_files_target": len(target_files),
        "raw_files_processed_ok": len(ok_set),
        "raw_files_skipped_unchanged": len(unchanged),
        "rows_read_from_target_files": rows_read,
        "id_dedup": id_stats_total,
        "rows_preprocessed_written": rows_preprocessed_written,
        "parts_written": {
            "merged_dedup": merged_parts_written,
            "preprocessed": preprocessed_parts_written,
            "text_dedup": text_parts_written,
        },
        "text_dedup": {
            "rows_before": text_before_total,
            "rows_after": text_after_total,
            "rows_removed": int(text_before_total - text_after_total),
        },
        "paths": {
            "merged_dedup_dir": str(MERGED_DIR),
            "preprocessed_dir": str(PREPROCESSED_DIR),
            "text_dedup_dir": str(TEXT_DEDUP_DIR),
            "exports_dir": str(EXPORTS_DIR),
            "manifest": str(MANIFEST_PATH),
            "run_report": str(RUN_REPORT_PATH),
        },
        "totals": {
            "merged_rows": previous_merged_rows + int(id_stats_total["appended_rows"]),
            "text_dedup_rows": previous_text_rows + text_after_total,
            "merged_part_files": len(parquet_parts(MERGED_DIR)),
            "preprocessed_part_files": len(parquet_parts(PREPROCESSED_DIR)),
            "text_dedup_part_files": len(parquet_parts(TEXT_DEDUP_DIR)),
        },
    }
    write_run_report(report, RUN_REPORT_PATH)
    emit_progress(
        "done "
        f"merged_rows={report['totals']['merged_rows']} "
        f"text_dedup_rows={report['totals']['text_dedup_rows']}"
    )
    return report


def export_csv(target: str = "all") -> dict[str, Any]:
    require_parquet_support()
    ensure_dirs()
    emit_progress(f"export start target={target}")

    targets = {
        "merged": (MERGED_DIR, MERGED_EXPORT_PATH),
        "text": (TEXT_DEDUP_DIR, TEXT_DEDUP_EXPORT_PATH),
    }
    selected = targets.items() if target == "all" else [(target, targets[target])]

    exported: dict[str, Any] = {}
    for name, (dataset_dir, csv_path) in selected:
        emit_progress(f"exporting {name} from {dataset_dir}")
        rows = export_parquet_dataset_to_csv(dataset_dir, csv_path)
        exported[name] = {
            "rows": rows,
            "csv_path": str(csv_path),
            "dataset_dir": str(dataset_dir),
        }
        emit_progress(f"exported {name} rows={rows} csv={csv_path}")

    return {
        "mode": "export-csv",
        "exported_at": now_iso_utc(),
        "target": target,
        "exports": exported,
    }


def get_status() -> dict[str, Any]:
    ensure_dirs()
    last_run = load_last_run_report(RUN_REPORT_PATH)
    totals = last_run.get("totals", {}) if isinstance(last_run, dict) else {}

    return {
        "raw_files_indexed": len(load_manifest(MANIFEST_PATH)),
        "merged_rows": int(totals.get("merged_rows", 0) or 0),
        "merged_part_files": len(parquet_parts(MERGED_DIR)),
        "preprocessed_part_files": len(parquet_parts(PREPROCESSED_DIR)),
        "text_dedup_rows": int(totals.get("text_dedup_rows", 0) or 0),
        "text_dedup_part_files": len(parquet_parts(TEXT_DEDUP_DIR)),
        "last_run": {
            "mode": last_run.get("mode"),
            "ended_at": last_run.get("ended_at"),
        },
        "paths": {
            "merged_dedup_dir": str(MERGED_DIR),
            "preprocessed_dir": str(PREPROCESSED_DIR),
            "text_dedup_dir": str(TEXT_DEDUP_DIR),
            "exports_dir": str(EXPORTS_DIR),
            "manifest": str(MANIFEST_PATH),
            "run_report": str(RUN_REPORT_PATH),
        },
    }


def print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="统一数据流水线：run/full/status/export-csv")
    parser.add_argument("command", nargs="?", choices=["run", "full", "status", "export-csv"], default="run")
    parser.add_argument("target", nargs="?", choices=["all", "merged", "text"], default="all")
    args = parser.parse_args()

    if args.command == "status":
        print_report(get_status())
    elif args.command == "full":
        print_report(run_pipeline(full_rebuild=True))
    elif args.command == "export-csv":
        print_report(export_csv(args.target))
    else:
        print_report(run_pipeline(full_rebuild=False))
