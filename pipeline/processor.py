"""
Normalizes raw BigKinds article dicts (from any source) into a consistent
schema that matches the articles table.

Both the unofficial POST API and the official Open API return overlapping but
not identical field names. This module handles both.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any

from bigkinds.media_codes import category_for_code, outlet_for_code

logger = logging.getLogger(__name__)

# ── Date parsing ──────────────────────────────────────────────────────────────

_DATE_PATTERNS = [
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%Y%m%d",
]


def _parse_date(raw: Any) -> date | None:
    if not raw:
        return None
    s = str(raw).strip()[:19]
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    logger.debug("Could not parse date: %r", raw)
    return None


# ── List fields ───────────────────────────────────────────────────────────────

def _to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    if isinstance(value, str):
        # May be comma-separated or pipe-separated
        sep = "|" if "|" in value else ","
        return [v.strip() for v in value.split(sep) if v.strip()]
    return [str(value).strip()]


# ── Main normalizer ───────────────────────────────────────────────────────────

def normalize(raw: dict) -> dict | None:
    """
    Convert a raw BigKinds dict to the articles table schema.
    Returns None if essential fields are missing.
    """
    # news_id — try multiple field names
    news_id = (
        raw.get("newsId")
        or raw.get("news_id")
        or raw.get("_id")
        or raw.get("id")
    )
    if not news_id:
        logger.debug("Skipping record with no news_id: %s", list(raw.keys()))
        return None

    # pub_date
    pub_date = _parse_date(
        raw.get("publishDate")
        or raw.get("published_at")
        or raw.get("pub_date")
        or raw.get("date")
    )
    if not pub_date:
        logger.debug("Skipping %s: no parseable date", news_id)
        return None

    # Provider info
    provider_code = raw.get("providerCode") or raw.get("provider_code") or ""
    provider_name = (
        raw.get("providerName")
        or raw.get("provider_name")
        or outlet_for_code(provider_code)
        or ""
    )
    media_category = category_for_code(provider_code) or ""

    # Content
    title   = (raw.get("title") or "").strip()
    body    = (raw.get("content") or raw.get("body") or "").strip()
    summary = (raw.get("summaryContent") or raw.get("summary") or "").strip()

    # Categories
    cat_large = raw.get("categoryCode") or raw.get("category_large") or ""
    cat_small = raw.get("categoryName")  or raw.get("category_small") or ""

    # Arrays
    keywords = _to_list(raw.get("keyword") or raw.get("keywords"))
    persons  = _to_list(raw.get("person")  or raw.get("persons"))
    orgs     = _to_list(raw.get("organization") or raw.get("organizations"))
    places   = _to_list(raw.get("place")   or raw.get("places"))

    reporter    = (raw.get("reporter") or raw.get("byLine") or "").strip()
    article_url = (raw.get("newsLink") or raw.get("article_url") or "").strip()

    return {
        "news_id":        str(news_id),
        "pub_date":       pub_date,
        "provider_code":  provider_code,
        "provider_name":  provider_name,
        "media_category": media_category,
        "title":          title,
        "body":           body,
        "summary":        summary,
        "category_large": cat_large,
        "category_small": cat_small,
        "keywords":       keywords,
        "persons":        persons,
        "organizations":  orgs,
        "places":         places,
        "reporter":       reporter,
        "article_url":    article_url,
    }


def normalize_batch(raw_articles: list[dict]) -> list[dict]:
    results = []
    for raw in raw_articles:
        normalized = normalize(raw)
        if normalized:
            results.append(normalized)
    return results


# ── Excel file normalizer (for Playwright downloads) ─────────────────────────

def normalize_excel(path: str) -> list[dict]:
    """Parse a BigKinds Excel download and return normalized article dicts."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required for Excel parsing: pip install pandas openpyxl")

    df = pd.read_excel(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # BigKinds Excel column names (Korean) → internal field mapping
    col_map = {
        "뉴스식별자":   "newsId",
        "일자":          "publishDate",
        "언론사":        "providerName",
        "제목":          "title",
        "본문":          "content",
        "요약문":        "summaryContent",
        "언론사코드":    "providerCode",
        "통합분류1":     "categoryCode",
        "통합분류2":     "categoryName",
        "키워드":        "keyword",
        "인물":          "person",
        "기관":          "organization",
        "장소":          "place",
        "기자":          "reporter",
        "URL":           "newsLink",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return normalize_batch(df.where(df.notna(), None).to_dict("records"))
