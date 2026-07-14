"""
BigKinds provider codes grouped by the 5 media categories required for this project.

Provider codes are the filterProviderCode values used in the BigKinds search API.
Each outlet has a numeric code; categories are logical groupings used internally.

Source: BigKinds provider list (bigkinds.or.kr 수집 언론사 목록)
"""

from __future__ import annotations

from typing import Dict, List

# ── Category constants ────────────────────────────────────────────────────────

NATIONAL_DAILY   = "전국일간지"
ECONOMIC_DAILY   = "경제일간지"
SPECIALIZED      = "전문지"
BROADCAST        = "방송사"
ONLINE           = "인터넷신문"

ALL_CATEGORIES = [NATIONAL_DAILY, ECONOMIC_DAILY, SPECIALIZED, BROADCAST, ONLINE]

# ── Provider codes by category ────────────────────────────────────────────────
# Format: { category: [(code, outlet_name), ...] }

PROVIDERS: Dict[str, List[tuple]] = {
    NATIONAL_DAILY: [
        ("01100101", "경향신문"),
        ("01100201", "국민일보"),
        ("01100301", "내일신문"),
        ("01100401", "동아일보"),
        ("01100501", "문화일보"),
        ("01100601", "서울신문"),
        ("01100701", "세계일보"),
        ("01100801", "조선일보"),
        ("01100901", "중앙일보"),
        ("01101001", "한겨레"),
        ("01101101", "한국일보"),
        ("01101201", "헤럴드경제"),
    ],
    ECONOMIC_DAILY: [
        ("01200101", "매일경제"),
        ("01200201", "머니투데이"),
        ("01200301", "서울경제"),
        ("01200401", "아시아경제"),
        ("01200501", "이데일리"),
        ("01200601", "파이낸셜뉴스"),
        ("01200701", "한국경제"),
        ("01200801", "헤럴드경제(경제)"),
    ],
    SPECIALIZED: [
        ("01300101", "코리아중앙데일리"),
        ("01300201", "코리아타임스"),
        ("01300301", "코리아헤럴드"),
        ("01300401", "한겨레21"),
        ("01300501", "시사IN"),
        ("01300601", "주간경향"),
        ("01300701", "주간조선"),
    ],
    BROADCAST: [
        ("02100101", "KBS"),
        ("02100201", "MBC"),
        ("02100301", "SBS"),
        ("02100401", "YTN"),
        ("02100501", "MBN"),
        ("02100601", "채널A"),
        ("02100701", "TV조선"),
        ("02100801", "JTBC"),
    ],
    ONLINE: [
        ("02300101", "노컷뉴스"),
        ("02300201", "뉴시스"),
        ("02300301", "뉴스1"),
        ("02300401", "연합뉴스"),
        ("02300501", "오마이뉴스"),
        ("02300601", "프레시안"),
        ("02300701", "미디어오늘"),
        ("02300801", "데일리안"),
        ("02300901", "뉴스웨이"),
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def codes_for_category(category: str) -> List[str]:
    """Return provider codes for a given category."""
    return [code for code, _ in PROVIDERS.get(category, [])]

def codes_for_all_categories() -> List[str]:
    """Return all provider codes across every category."""
    return [code for entries in PROVIDERS.values() for code, _ in entries]

def category_for_code(code: str) -> str | None:
    """Reverse-lookup: provider code → category name."""
    for cat, entries in PROVIDERS.items():
        if any(c == code for c, _ in entries):
            return cat
    return None

def outlet_for_code(code: str) -> str | None:
    """Reverse-lookup: provider code → outlet name."""
    for entries in PROVIDERS.values():
        for c, name in entries:
            if c == code:
                return name
    return None

# Comma-separated strings ready for the API filterProviderCode parameter
FILTER_CODE_ALL = ",".join(codes_for_all_categories())
FILTER_CODES_BY_CATEGORY = {
    cat: ",".join(codes_for_category(cat)) for cat in ALL_CATEGORIES
}
