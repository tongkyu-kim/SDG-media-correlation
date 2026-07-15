"""
Korean-language development/ODA vocabulary dictionary.

Twelve categories — international organizations, Korean development actors,
development/SDG/humanitarian/sectoral/climate/trade/initiative/event
vocabulary, and aid verbs — supplementing the per-SDG keyword lists in
keywords_ko.py and the country list in countries_ko.py. This backs the
candidate rule in classify/candidate_filter.py.

Korean terms are standard translations of well-known institution names
(UNDP, WHO, World Bank, etc.); a manual spot-check against an authoritative
Korean ODA glossary (e.g. odakorea.go.kr) is recommended before treating
this list as fully validated.

Design decisions:

1. Development-sector terms are split into two tiers:
   - SECTOR_TERMS_KO: standalone trigger-eligible (irrigation, rural
     development, renewable energy, digital/financial inclusion, etc.) —
     specific enough that false-positive risk is acceptable.
   - SECTOR_TERMS_COOCCUR_ONLY_KO: 학교(school), 교사(teacher), 병원(hospital),
     중소기업(SMEs), etc. are excluded from standalone triggering. These are
     the same class of generic Korean-domestic terms already excluded from
     the SDG3 keyword list in keywords_ko.py, for the same reason — they
     fire on ordinary domestic news with no development relevance. They
     only count as a signal when a country/org/actor mention also appears
     in the same article.

2. Aid verbs (AID_VERBS_KO) are recorded for documentation and possible
   future co-occurrence-weighted scoring, but are not wired into the
   candidate trigger. Korean equivalents of "support," "provide," "invest,"
   "expand," "cooperate," "implement," "strengthen," "construct"
   (지원하다/제공하다/투자하다/확대하다/협력하다/시행하다/강화하다/건설하다)
   are near-universal verbs in Korean news of any kind — domestic policy,
   corporate earnings, sports coverage all use them routinely. Treating
   them as an independent trigger would push the candidate rate toward the
   full corpus, defeating the purpose of a compute-bounding pre-filter.

3. "KF" (Korea Foundation) and bare "UN"/"KSP" as Latin-letter tokens carry
   real homonym risk in Korean news (KF94/KF80 face masks are extremely
   common; short Latin acronyms can appear inside unrelated strings). Full
   Korean names are used as the primary form; bare English acronyms are
   included only where they are distinctive multi-letter tokens (UNDP,
   UNESCO, UNICEF, KOICA, EDCF). Bare "KF" and "UN" are omitted.
"""

from __future__ import annotations

import re
from typing import Dict, List

# ── Rule 2: International organizations ───────────────────────────────────
INTL_ORGS_KO: Dict[str, List[str]] = {
    "UN":       ["유엔", "국제연합"],
    "UNDP":     ["유엔개발계획", "UNDP"],
    "UNICEF":   ["유니세프", "유엔아동기금", "UNICEF"],
    "UNHCR":    ["유엔난민기구", "UNHCR"],
    "UNESCO":   ["유네스코", "유엔교육과학문화기구", "UNESCO"],
    "WHO":      ["세계보건기구", "WHO"],
    "FAO":      ["유엔식량농업기구", "FAO"],
    "ILO":      ["국제노동기구", "ILO"],
    "UN Women": ["유엔여성기구"],
    "UNEP":     ["유엔환경계획", "UNEP"],
    "UNIDO":    ["유엔공업개발기구", "UNIDO"],
    "UNFPA":    ["유엔인구기금", "UNFPA"],
    "WFP":      ["세계식량계획", "WFP"],
    "IFAD":     ["국제농업개발기금", "IFAD"],
    "UNOPS":    ["유엔프로젝트조달기구", "UNOPS"],
    "World Bank": ["세계은행"],
    "IBRD":     ["국제부흥개발은행", "IBRD"],
    "IDA":      ["국제개발협회"],
    "IFC":      ["국제금융공사", "IFC"],
    "MIGA":     ["국제투자보증기구", "MIGA"],
    "ADB":      ["아시아개발은행", "ADB"],
    "AfDB":     ["아프리카개발은행", "AfDB"],
    "AIIB":     ["아시아인프라투자은행", "AIIB"],
    "EBRD":     ["유럽부흥개발은행", "EBRD"],
    "OECD":     ["경제협력개발기구", "OECD"],
    "OECD DAC": ["개발원조위원회", "OECD DAC"],
    "IMF":      ["국제통화기금", "IMF"],
    "IOM":      ["국제이주기구", "IOM"],
    "GAVI":     ["세계백신면역연합", "가비"],
    "Global Fund":       ["글로벌펀드", "세계기금"],
    "Green Climate Fund": ["녹색기후기금"],
    "GEF":      ["지구환경기금"],
    "CGIAR":    ["국제농업연구자문그룹"],
}

# ── Rule 3: Korean development actors ─────────────────────────────────────
# KOICA/EDCF/공적개발원조 already exist in classify/keyword_classifier.py's
# POLICY_ACTORS list — repeated here so this module is self-contained, but
# candidate_filter.py should not double-count; see integration notes there.
KOREAN_DEV_ACTORS_KO: Dict[str, List[str]] = {
    "KOICA":  ["코이카", "한국국제협력단", "KOICA"],
    "EDCF":   ["대외경제협력기금", "EDCF"],
    "KEXIM":  ["한국수출입은행"],
    "MOFA":   ["외교부"],
    "ODA":    ["공적개발원조", "ODA"],
    "CIDC":   ["국제개발협력위원회"],
    "KSP":    ["경제발전경험공유사업", "지식공유사업"],
    "KOAFEC": ["한-아프리카 경제협력", "한아프리카경제협력회의", "KOAFEC"],
    "KOFIH":  ["한국국제보건의료재단"],
    "KF":     ["한국국제교류재단"],  # bare "KF" deliberately excluded — see module docstring
}

# ── Rule 4: Development vocabulary ────────────────────────────────────────
DEV_VOCAB_KO: List[str] = [
    "국제개발", "지속가능한 개발", "지속가능개발", "역량강화", "역량개발",
    "기술협력", "기술지원", "무상원조", "유상원조", "양허성차관",
    "남남협력", "삼각협력", "개발협력", "정책대화", "지식공유",
    "개발금융", "기후금융", "적응금융", "혼합금융", "공여국", "수원국",
]

# ── Rule 5: SDG vocabulary (top-level/agenda phrases, not per-SDG keywords —
# those live in keywords_ko.py) ────────────────────────────────────────────
SDG_AGENDA_VOCAB_KO: List[str] = [
    "지속가능발전목표", "지속가능한 개발목표", "SDGs", "SDG",
    "2030 의제", "2030의제",
]

# ── Rule 6: Humanitarian vocabulary ───────────────────────────────────────
HUMANITARIAN_VOCAB_KO: List[str] = [
    "긴급구호", "구호활동", "국내실향민", "식량불안", "기근",
    "콜레라", "에볼라", "감염병 대유행", "재난대응", "재해복구",
    "이재민", "구호물자", "난민캠프", "난민촌",
]

# ── Rule 7: Development sectors ────────────────────────────────────────────
# Standalone trigger-eligible — specific enough to carry real signal alone.
SECTOR_TERMS_KO: List[str] = [
    "모성보건", "관개", "관개시설", "농촌개발", "재생에너지",
    "마이크로그리드", "전력화", "디지털 포용", "인터넷 접근성",
    "여성역량강화", "청년고용", "직업훈련", "금융포용", "소액금융",
    "농업기술", "농업생산성",
]
# NOT standalone — too generic in Korean domestic news (school/teacher/
# hospital/SME stories are constant even with zero development relevance).
# Only counts when co-occurring with a country/org/actor mention.
SECTOR_TERMS_COOCCUR_ONLY_KO: List[str] = [
    "학교", "교사", "병원", "중소기업", "농업", "위생", "영양",
]

# ── Rule 8: Climate & environment ─────────────────────────────────────────
CLIMATE_VOCAB_KO: List[str] = [
    "탄소중립", "국가결정기여", "파리협정", "생물다양성", "산림복원",
    "맹그로브", "해양보전", "기후회복력", "자연기반해법", "산림전용",
    "기후적응", "기후완화",
]

# ── Rule 9: Trade & infrastructure ────────────────────────────────────────
TRADE_INFRA_VOCAB_KO: List[str] = [
    "경제자유구역", "경제특구", "산업단지", "전력망", "송전망",
    "교통연계성", "통관현대화", "국경간 인프라", "경제회랑",
]

# ── Rule 10: Major development initiatives ────────────────────────────────
INITIATIVES_KO: List[str] = [
    "일대일로", "글로벌 게이트웨이", "더 나은 세계 재건", "글로벌인프라투자파트너십",
    "메콩강", "아세안 연계성", "아프리카대륙자유무역지대", "NEPAD",
    "아프리카개발을위한새로운파트너십",
]

# ── Rule 11: Development events ───────────────────────────────────────────
EVENTS_KO: List[str] = [
    "기후변화당사국총회", "유엔총회", "고위급정치포럼", "세계은행 연차총회",
    "IMF·세계은행 춘계회의", "개발재원총회", "부산세계개발원조총회",
    "효과적개발협력을위한글로벌파트너십",
]

# ── Rule 12: Aid verbs — documented, NOT wired into the candidate trigger ──
# (see module docstring, point 2)
AID_VERBS_KO: List[str] = [
    "지원하다", "제공하다", "투자하다", "확대하다", "강화하다",
    "협력하다", "시행하다", "건설하다", "복구하다", "재건하다",
    "기부하다", "공여하다",
]

# ── Combined standalone-trigger set ───────────────────────────────────────
_STANDALONE_CATEGORIES: Dict[str, List[str]] = {
    "intl_org":       [term for terms in INTL_ORGS_KO.values() for term in terms],
    "korean_actor":   [term for terms in KOREAN_DEV_ACTORS_KO.values() for term in terms if term != "KF"],
    "dev_vocab":      DEV_VOCAB_KO,
    "sdg_agenda":     SDG_AGENDA_VOCAB_KO,
    "humanitarian":   HUMANITARIAN_VOCAB_KO,
    "sector":         SECTOR_TERMS_KO,
    "climate":        CLIMATE_VOCAB_KO,
    "trade_infra":    TRADE_INFRA_VOCAB_KO,
    "initiative":     INITIATIVES_KO,
    "event":          EVENTS_KO,
}

_ALL_TERMS: set[str] = {t for terms in _STANDALONE_CATEGORIES.values() for t in terms}
_COOCCUR_ONLY_TERMS: set[str] = set(SECTOR_TERMS_COOCCUR_ONLY_KO)


def has_development_vocab_hit(text: str) -> bool:
    """True if text contains any Rule 2/3/4/5/6/7(standalone)/8/9/10/11 term."""
    if not text:
        return False
    return any(term in text for term in _ALL_TERMS)


def development_vocab_categories(text: str) -> list[str]:
    """Which categories matched — for diagnostics/spot-checking, not scoring."""
    if not text:
        return []
    return [cat for cat, terms in _STANDALONE_CATEGORIES.items()
            if any(term in text for term in terms)]


def has_cooccur_only_sector_hit(text: str) -> bool:
    """
    True if text contains a Rule-7 cooccur-only term (학교/교사/병원/중소기업/etc).
    Callers should AND this with a country/org/actor signal before treating
    it as a candidate trigger — see module docstring point 1.
    """
    if not text:
        return False
    return any(term in text for term in _COOCCUR_ONLY_TERMS)
