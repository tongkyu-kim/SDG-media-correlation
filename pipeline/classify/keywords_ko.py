"""
Korean SDG keyword lists for rule-based pre-filtering and confidence boosting.

Sources: SDSN keyword taxonomy, Aurora SDG query system, UN SDG metadata —
translated and extended for Korean news context.

Each list contains high-precision terms. Breadth is deliberately limited to
reduce false positives; coverage is handled by the ML classifier.
"""

from typing import Dict, List

# Format: { sdg_number: [keyword, ...] }
# Keywords are matched as substrings (case-insensitive in Korean).

SDG_KEYWORDS_KO: Dict[int, List[str]] = {
    1: [
        "빈곤", "극빈", "기초생활수급", "생계급여", "취약계층", "저소득층",
        "사회안전망", "빈곤율", "빈곤층", "절대빈곤", "상대적 빈곤",
        "기초생활보장", "사각지대", "극빈층",
    ],
    2: [
        "식량안보", "기아", "굶주림", "영양결핍", "식량부족", "영양부족",
        "식량위기", "식량지원", "농업개발", "식량원조", "식량자급",
        "식량불안", "영양실조", "아동 영양",
    ],
    3: [
        "보건의료", "공중보건", "전염병", "감염병", "의료접근성",
        "건강보험", "의료체계", "백신", "예방접종", "사망률", "모성사망",
        "신생아 사망", "에이즈", "결핵", "말라리아", "정신건강",
        "의약품 접근", "유니버설 헬스케어", "건강검진", "의료비",
        "비만", "당뇨", "고혈압", "암 검진", "병원", "의원",
        "건강수명", "건강관리", "만성질환",
    ],
    4: [
        "교육기회", "교육격차", "교육접근성", "문해력", "초등교육",
        "중등교육", "직업훈련", "성인교육", "교육 불평등", "학교 중퇴",
        "교육비 지원", "장학금", "교육 인프라", "교사 부족",
    ],
    5: [
        "성평등", "젠더 평등", "여성 권리", "성차별", "여성 폭력",
        "가정폭력", "성폭력", "여성 경제참여", "유리천장", "성별 임금격차",
        "여성 대표성", "생리대", "재생산권", "여성 리더십",
    ],
    6: [
        "수자원", "식수", "깨끗한 물", "물 부족", "상하수도", "위생시설",
        "수질오염", "물 관리", "물 안보", "홍수 관리", "지하수",
        "하수처리", "수도 인프라",
    ],
    7: [
        "재생에너지", "신재생에너지", "태양광", "풍력", "에너지전환",
        "에너지 빈곤", "에너지 접근", "전력화", "에너지효율",
        "탄소중립 에너지", "수소에너지", "전기차", "에너지 안보",
        "청정에너지",
    ],
    8: [
        "일자리 창출", "고용률", "청년 실업", "노동권", "최저임금",
        "비정규직", "경제성장", "포용성장", "노동환경", "산업재해",
        "아동노동", "강제노동", "공정무역", "중소기업",
    ],
    9: [
        "인프라", "디지털 전환", "혁신기술", "연구개발", "산업화",
        "스마트 제조", "4차 산업혁명", "인터넷 접근", "디지털 격차",
        "기술이전", "첨단산업", "제조업 혁신",
    ],
    10: [
        "불평등", "소득격차", "양극화", "경제 불평등", "사회 이동성",
        "기회 불평등", "취약계층 지원", "차별 금지", "사회적 포용",
        "이민자 권리", "이주노동자",
    ],
    11: [
        "지속가능한 도시", "도시 빈민", "슬럼", "주거권", "적정 주택",
        "공공교통", "도시 교통", "스마트시티", "도시계획", "재난 위험",
        "문화유산 보존", "도시 녹지",
    ],
    12: [
        "지속가능한 소비", "지속가능한 생산", "자원 효율", "음식물 낭비",
        "플라스틱 폐기물", "재활용", "순환경제", "친환경 소비",
        "과소비", "탄소발자국", "ESG", "녹색 구매",
    ],
    13: [
        "기후변화", "기후위기", "온실가스", "탄소배출", "탄소중립",
        "탄소세", "파리협정", "기후협약", "기후적응", "기후재앙",
        "이상기후", "폭염", "홍수", "기후난민", "넷제로",
    ],
    14: [
        "해양오염", "해양생태계", "어업", "수산자원", "산호초",
        "해양산성화", "플라스틱 해양오염", "불법 어업", "해양보호구역",
        "연안 관리", "해양 생물다양성",
    ],
    15: [
        "생물다양성", "산림 파괴", "사막화", "토지 황폐화", "멸종위기종",
        "야생동물", "생태계 복원", "삼림벌채", "산림 보전",
        "육상 생태계", "습지 보전", "외래종",
    ],
    16: [
        "법치주의", "부패 척결", "투명성", "인권", "민주주의",
        "평화구축", "분쟁 해결", "테러리즘", "아동 학대", "인신매매",
        "사법접근성", "언론 자유", "시민사회",
    ],
    17: [
        "공적개발원조", "ODA", "개발협력", "국제원조", "다자협력",
        "파트너십", "개발재원", "기술협력", "남남협력", "민관협력",
        "무역 원조", "개발도상국 지원",
    ],
}


def keyword_scores(text: str) -> Dict[int, int]:
    """
    Count keyword hits per SDG for the given text.
    Returns {sdg: hit_count} for SDGs with at least one match.
    """
    lower = text.lower()
    return {
        sdg: sum(1 for kw in keywords if kw in lower)
        for sdg, keywords in SDG_KEYWORDS_KO.items()
        if any(kw in lower for kw in keywords)
    }


def top_sdg_from_keywords(text: str) -> tuple[int, int] | tuple[None, int]:
    """
    Return (top_sdg, hit_count) based purely on keyword matching.
    Returns (None, 0) if no keywords found.
    """
    scores = keyword_scores(text)
    if not scores:
        return None, 0
    top = max(scores, key=scores.__getitem__)
    return top, scores[top]
