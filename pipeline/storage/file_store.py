"""
File-based article storage.

Output layout under NEWS_DATA_DIR (default: ../src/news/):

    src/news/
    ├── 2024/
    │   ├── 01/
    │   │   ├── 2024-01-01.json    ← list of article dicts
    │   │   └── 2024-01-02.json
    │   └── 02/
    │       └── 2024-02-01.json
    └── .collection_log.json       ← {date_str: {status, count, ...}}

The collection log is the resume mechanism for backfill: any date whose
status is not "done" is re-collected when backfill runs again.
"""

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import config

logger = logging.getLogger(__name__)

_ROOT = Path(config.NEWS_DATA_DIR)
_LOG_FILE = _ROOT / ".collection_log.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _article_path(d: date) -> Path:
    return _ROOT / f"{d.year}" / f"{d.month:02d}" / f"{d.isoformat()}.json"


def _read_log() -> dict:
    if _LOG_FILE.exists():
        try:
            return json.loads(_LOG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read collection log; starting fresh")
    return {}


def _write_log(log: dict) -> None:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE.write_text(
        json.dumps(log, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# ── Article I/O ───────────────────────────────────────────────────────────────

def save_articles(target_date: date, articles: Iterable[dict]) -> int:
    """
    Write articles to src/news/YYYY/MM/YYYY-MM-DD.json.
    Merges with any existing file (deduplicates by news_id).
    Returns the number of articles written.
    """
    rows = list(articles)
    if not rows:
        return 0

    path = _article_path(target_date)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing records and merge (upsert by news_id)
    existing: dict[str, dict] = {}
    if path.exists():
        try:
            for rec in json.loads(path.read_text(encoding="utf-8")):
                existing[rec["news_id"]] = rec
        except (json.JSONDecodeError, KeyError, OSError):
            existing = {}

    before = len(existing)
    for row in rows:
        existing[row["news_id"]] = row

    merged = list(existing.values())
    merged.sort(key=lambda r: (r.get("provider_code", ""), r.get("news_id", "")))

    path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    new_count = len(existing) - before
    logger.info("Saved %s → %s (%d new, %d total)", target_date, path, new_count, len(merged))
    return new_count


def load_articles(target_date: date) -> list[dict]:
    path = _article_path(target_date)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


# ── Collection log ────────────────────────────────────────────────────────────

def log_start(target_date: date) -> None:
    log = _read_log()
    log[target_date.isoformat()] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_log(log)


def log_done(target_date: date, count: int) -> None:
    log = _read_log()
    entry = log.get(target_date.isoformat(), {})
    entry.update({
        "status": "done",
        "count": count,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    log[target_date.isoformat()] = entry
    _write_log(log)


def log_failed(target_date: date, error: str) -> None:
    log = _read_log()
    entry = log.get(target_date.isoformat(), {})
    entry.update({
        "status": "failed",
        "error": error,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    log[target_date.isoformat()] = entry
    _write_log(log)


# ── Query helpers used by backfill ────────────────────────────────────────────

def get_uncollected_dates(start: date, end: date) -> list[date]:
    """Return every date in [start, end] that is not yet status='done'."""
    from datetime import timedelta
    log = _read_log()
    out = []
    d = start
    while d <= end:
        if log.get(d.isoformat(), {}).get("status") != "done":
            out.append(d)
        d += timedelta(days=1)
    return out


def get_failed_dates() -> list[date]:
    log = _read_log()
    return sorted(
        date.fromisoformat(k)
        for k, v in log.items()
        if v.get("status") == "failed"
    )
