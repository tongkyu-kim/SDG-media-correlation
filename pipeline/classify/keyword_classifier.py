"""
Pure keyword-based classifier for Korean development news articles.

No ML dependencies — uses curated Korean keyword dictionaries.
All operations are vectorized over pandas Series for speed (~1-2s per 40K-article file).

Output columns (prefix kw_ = keyword-derived, used as fallback when BERT unavailable):
  kw_sdg_label        int  Primary SDG 1-17 (0 = not SDG-relevant)
  kw_sdg_secondary    int  Secondary SDG 1-17 (0 = none)
  kw_sdg_score        float Keyword density proxy 0.0-1.0
  kw_sdg_intensity    int  0-3 confidence proxy (matches BERT output schema)
  issue_intensity     int  0-5 severity/urgency (0 = not SDG-relevant)
  kw_sdg_favorability str  positive | neutral | negative  (backward-compat alias)
  aid_stance          str  supportive | neutral | opposed
  issue_frame         str  humanitarian | economic | security | environmental | governance | mixed | ""
  problem_solution    str  problem | solution | mixed | neutral
  crisis_type         str  comma-separated subset of: disaster,climate,conflict,health,food,refugee
  policy_actor        int  1 if Korean ODA actor explicitly mentioned, else 0

Text inputs:
  text_short = title + keywords + top_keywords   (SDG scoring, aid stance, problem/solution)
  text_long  = text_short + body[:300]            (crisis type, issue frame, policy actor)
"""

from __future__ import annotations

import re
import pandas as pd
import numpy as np
from typing import Dict

from .keywords_ko import SDG_KEYWORDS_KO


# ── Severity amplifiers ────────────────────────────────────────────────────────
# Words indicating high issue severity — used to boost issue_intensity score.

SEVERITY_AMPLIFIERS: list[str] = [
    # Scale of impact
    "수백만명", "수십만명", "수만명", "수천명", "수백명",
    # Mortality / catastrophe
    "사망자", "희생자", "아사", "사망", "참사", "학살", "재앙",
    # Emergency declarations
    "비상사태", "긴급사태", "재난 선포", "국가 비상", "위기 선포",
    # Collapse / destruction
    "붕괴", "파괴", "폐허", "완전 파괴",
    # Famine / extreme deprivation
    "기근", "극심한 기아", "극심한 빈곤",
]

# ── Aid support stance ─────────────────────────────────────────────────────────
# Supportive: article argues for, justifies, or encourages aid spending.
# Opposed: article criticises, questions, or discourages aid spending.

AID_SUPPORTIVE: list[str] = [
    "원조 필요", "원조가 필요", "지원이 필요", "지원 필요",
    "원조 확대", "원조 증가", "원조 강화", "원조 촉구",
    "ODA 증액", "ODA 확대", "원조 지속",
    "지원 호소", "지원 촉구", "국제 지원 요청",
    "인도적 지원 촉구", "원조 효과적", "원조 성과",
    "원조 성공", "개발원조 필요", "지원 확대 필요",
]

AID_OPPOSED: list[str] = [
    "원조 낭비", "원조 비효율", "원조 효과 없",
    "원조 실패", "원조 문제", "원조 비판",
    "원조 삭감", "ODA 삭감", "원조 축소", "원조 중단",
    "세금 낭비", "원조에 반대", "원조 반대",
    "원조 부패", "원조 착복", "지원금 유용",
    "원조 무용", "원조 회의",
]

# ── Issue frames ───────────────────────────────────────────────────────────────
# Based on standard media framing literature (Entman 1993; Semetko & Valkenburg 2000).

FRAMES: Dict[str, list[str]] = {
    "humanitarian": [
        "인도적", "인도주의", "구호", "긴급구호", "인도지원",
        "난민", "피해자", "이재민", "취약계층", "구호물자",
        "생존", "고통 받는", "구호 활동",
    ],
    "economic": [
        "경제개발", "경제성장", "빈곤감소", "소득증가",
        "일자리 창출", "무역협력", "투자", "경제협력",
        "성장 동력", "시장 개방", "민간 투자", "경제 회복",
    ],
    "security": [
        "안보", "평화유지", "분쟁 지역", "테러",
        "무장단체", "평화협정", "안정화", "치안",
        "PKO", "평화구축", "무력 충돌",
    ],
    "environmental": [
        "기후변화", "환경파괴", "생태계", "탄소 배출",
        "온실가스", "기후위기", "환경오염", "생물다양성",
        "산림파괴", "기후적응", "기후재앙",
    ],
    "governance": [
        "민주주의", "법치", "부패", "선거", "인권",
        "거버넌스", "시민사회", "투명성", "책임성",
        "사법", "법 집행", "제도 개혁",
    ],
}

# ── Problem / solution framing ─────────────────────────────────────────────────

PROBLEM_KW: list[str] = [
    "피해", "사망", "부족", "결핍", "위기", "고통",
    "취약", "위협", "파괴", "붕괴", "기아", "가난",
    "실업", "박해", "학대", "차별", "악화", "심화",
    "부족", "굶주림", "결핍", "미충족",
]

SOLUTION_KW: list[str] = [
    "원조", "지원", "구호", "개발 사업", "지원 사업",
    "원조 프로그램", "성공", "개선", "달성",
    "해결", "복구", "재건", "회복", "향상",
    "지원 완료", "사업 완료", "목표 달성",
]

# ── Crisis types ───────────────────────────────────────────────────────────────
# Multi-label: an article may match multiple crisis types.

CRISIS_TYPES: Dict[str, list[str]] = {
    "disaster": [
        "지진", "홍수", "태풍", "허리케인", "사이클론",
        "가뭄", "산사태", "쓰나미", "화산", "폭풍",
        "자연재해", "폭우 피해", "지진 피해", "홍수 피해",
    ],
    "climate": [
        "기후변화", "폭염", "이상기후", "해수면 상승",
        "기후재앙", "기후난민", "기후위기", "이상 고온",
    ],
    "conflict": [
        "전쟁", "내전", "무력분쟁", "교전",
        "폭격", "전투", "무장충돌", "군사작전",
        "분쟁 지역", "전투 지역",
    ],
    "health": [
        "전염병", "감염병", "팬데믹", "에피데믹",
        "바이러스", "코로나", "에볼라", "콜레라",
        "보건 위기", "의료 붕괴", "방역",
    ],
    "food": [
        "식량위기", "기아", "기근", "식량부족",
        "영양실조", "아사", "식량 불안", "식량 위기",
    ],
    "refugee": [
        "난민", "이주민", "강제이주", "피란민",
        "실향민", "망명", "난민 위기", "난민 캠프",
    ],
}

# ── Policy actors ──────────────────────────────────────────────────────────────
# Korean ODA-specific actors. Presence indicates direct policy relevance.

POLICY_ACTORS: list[str] = [
    "KOICA", "코이카", "한국국제협력단",
    "EDCF", "대외경제협력기금",
    "ODA 사업", "공적개발원조", "국제개발협력",
    "한국 원조", "한국 개발협력", "한국 ODA",
    "개발원조위원회", "한국 지원 사업",
]


def _compile(terms: list[str], flags: int = 0) -> re.Pattern:
    """Compile a list of terms into a single alternation regex."""
    return re.compile("|".join(re.escape(t) for t in terms), flags)


class KeywordClassifier:
    """
    Classifies Korean news articles across 10 dimensions using keyword matching.
    Designed for vectorized batch processing — call classify_dataframe() on a
    full file's DataFrame rather than classify() on individual rows.
    """

    def __init__(self) -> None:
        # SDG patterns — one per SDG
        self._sdg_patterns: Dict[int, re.Pattern] = {
            sdg: _compile(kws)
            for sdg, kws in SDG_KEYWORDS_KO.items()
        }
        # Other variable patterns
        self._severity_re     = _compile(SEVERITY_AMPLIFIERS)
        self._support_re      = _compile(AID_SUPPORTIVE)
        self._oppose_re       = _compile(AID_OPPOSED)
        self._frame_patterns  = {f: _compile(kws) for f, kws in FRAMES.items()}
        self._problem_re      = _compile(PROBLEM_KW)
        self._solution_re     = _compile(SOLUTION_KW)
        self._crisis_patterns = {ct: _compile(kws) for ct, kws in CRISIS_TYPES.items()}
        self._policy_re       = _compile(POLICY_ACTORS, re.IGNORECASE)

    # ── Text extraction ────────────────────────────────────────────────────────

    @staticmethod
    def _text_short(df: pd.DataFrame) -> pd.Series:
        """title + keywords + top_keywords — primary classification text."""
        cols = ["title", "keywords", "top_keywords"]
        parts = [df[c].fillna("") for c in cols if c in df.columns]
        if not parts:
            return pd.Series("", index=df.index)
        return parts[0].str.cat(parts[1:], sep=" ", na_rep="")

    @staticmethod
    def _text_long(df: pd.DataFrame, short: pd.Series) -> pd.Series:
        """short + first 300 chars of body — for frame/crisis/policy detection."""
        if "body" not in df.columns:
            return short
        body = df["body"].fillna("").str[:300]
        return short + " " + body

    # ── Main classifier ────────────────────────────────────────────────────────

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify an entire file's DataFrame in one pass.

        Returns a DataFrame (same index as df) with all keyword output columns.
        Typically takes 1-3 seconds for a 40K-row file.
        """
        short = self._text_short(df)
        long_ = self._text_long(df, short)

        # ── SDG keyword hit counts ─────────────────────────────────────────────
        sdg_hits = pd.DataFrame(
            {sdg: short.str.count(pat) for sdg, pat in self._sdg_patterns.items()},
            index=df.index,
        )
        max_hits  = sdg_hits.max(axis=1)
        total_hits = sdg_hits.sum(axis=1)

        # Primary SDG — column with most hits; 0 if no matches
        primary = sdg_hits.idxmax(axis=1).where(max_hits > 0, 0).astype(int)

        # Secondary SDG — highest remaining after masking primary
        temp = sdg_hits.copy().astype(float)
        for sdg_num in SDG_KEYWORDS_KO:
            mask = primary == sdg_num
            if mask.any():
                temp.loc[mask, sdg_num] = -1.0
        secondary = temp.idxmax(axis=1).where(temp.max(axis=1) > 0, 0).astype(int)

        # SDG score: asymptotic proxy for confidence (0.0–1.0)
        sdg_score = (max_hits / (max_hits + 3)).round(4).where(max_hits > 0, 0.0)

        # SDG intensity (0-3) — keyword density proxy matching BERT output schema
        sdg_intensity = np.select(
            [max_hits == 0, max_hits < 3, max_hits < 6],
            [0, 1, 2],
            default=3,
        )

        # ── Issue intensity (0-5 severity scale) ──────────────────────────────
        severity_hits = long_.str.count(self._severity_re)
        base = np.select(
            [max_hits == 0, max_hits == 1, max_hits <= 3, max_hits <= 6],
            [0, 1, 2, 3],
            default=4,
        )
        issue_intensity = np.clip(base + severity_hits.clip(0, 1).values, 0, 5).astype(int)
        issue_intensity[max_hits.values == 0] = 0

        # ── Aid support stance ─────────────────────────────────────────────────
        support_hits = long_.str.count(self._support_re)
        oppose_hits  = long_.str.count(self._oppose_re)
        aid_stance = pd.Series("neutral", index=df.index)
        aid_stance[support_hits > oppose_hits] = "supportive"
        aid_stance[oppose_hits  > support_hits] = "opposed"

        # Backward-compatible mapping for aggregate_media.py
        sdg_favorability = aid_stance.map(
            {"supportive": "positive", "neutral": "neutral", "opposed": "negative"}
        )

        # ── Issue frame ────────────────────────────────────────────────────────
        frame_hits = pd.DataFrame(
            {f: long_.str.count(pat) for f, pat in self._frame_patterns.items()},
            index=df.index,
        )
        n_active_frames = (frame_hits > 0).sum(axis=1)
        dominant_frame  = frame_hits.idxmax(axis=1).where(frame_hits.max(axis=1) > 0, "")
        dominant_frame[n_active_frames >= 3] = "mixed"

        # ── Problem / solution framing ─────────────────────────────────────────
        prob_hits = short.str.count(self._problem_re)
        soln_hits = short.str.count(self._solution_re)

        prob_sol = pd.Series("neutral", index=df.index)
        prob_sol[(prob_hits > 0) & (soln_hits == 0)] = "problem"
        prob_sol[(soln_hits > 0) & (prob_hits == 0)] = "solution"
        prob_sol[(prob_hits > 0) & (soln_hits > 0)]  = "mixed"
        # Dominant sub-type when both present
        dominant_prob = (prob_hits > soln_hits * 2) & (soln_hits > 0)
        dominant_soln = (soln_hits > prob_hits * 2) & (prob_hits > 0)
        prob_sol[dominant_prob] = "problem"
        prob_sol[dominant_soln] = "solution"

        # ── Crisis type (multi-label) ──────────────────────────────────────────
        crisis_flags = pd.DataFrame(
            {ct: long_.str.contains(pat, na=False) for ct, pat in self._crisis_patterns.items()},
            index=df.index,
        )
        crisis_type = crisis_flags.apply(
            lambda row: ",".join(ct for ct in crisis_flags.columns if row[ct]),
            axis=1,
        )

        # ── Policy actor mention ───────────────────────────────────────────────
        policy_actor = long_.str.contains(self._policy_re, na=False).astype(int)

        return pd.DataFrame(
            {
                "kw_sdg_label":        primary,
                "kw_sdg_secondary":    secondary,
                "kw_sdg_hits":         max_hits,              # raw hit count for BERT threshold
                "kw_sdg_score":        sdg_score,
                "kw_sdg_intensity":    pd.Series(sdg_intensity, index=df.index, dtype=int),
                "issue_intensity":     pd.Series(issue_intensity, index=df.index, dtype=int),
                "kw_sdg_favorability": sdg_favorability,
                "aid_stance":          aid_stance,
                "issue_frame":         dominant_frame,
                "problem_solution":    prob_sol,
                "crisis_type":         crisis_type,
                "policy_actor":        policy_actor,
            },
            index=df.index,
        )
