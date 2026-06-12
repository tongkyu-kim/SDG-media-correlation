"""
Korean country name dictionary.

Maps Korean country names (as they appear in news and ODA data) to
standardised English names and ISO 3166-1 alpha-3 codes.

Organised by UN region. Covers ~170 countries + common regional aggregates.

Sources:
  - UN Term Portal (unterm.un.org) — official Korean UN names
  - MOFA Korea — Korean Ministry of Foreign Affairs country name standards
  - ODA dataset 수원국 field (BigKinds/KOICA naming conventions)
"""

from __future__ import annotations

# Format: "Korean name": ("ISO3", "English name", "UN region")
# ISO3 = "---" for regional/unallocated entries

COUNTRY_MAP: dict[str, tuple[str, str, str]] = {

    # ── East Asia ─────────────────────────────────────────────────────────────
    "중국":           ("CHN", "China",              "E_ASIA"),
    "일본":           ("JPN", "Japan",              "E_ASIA"),
    "몽골":           ("MNG", "Mongolia",           "E_ASIA"),
    "대만":           ("TWN", "Taiwan",             "E_ASIA"),
    "북한":           ("PRK", "North Korea",        "E_ASIA"),
    "홍콩":           ("HKG", "Hong Kong",          "E_ASIA"),

    # ── Southeast Asia ────────────────────────────────────────────────────────
    "베트남":         ("VNM", "Vietnam",            "SE_ASIA"),
    "캄보디아":       ("KHM", "Cambodia",           "SE_ASIA"),
    "미얀마":         ("MMR", "Myanmar",            "SE_ASIA"),
    "라오스":         ("LAO", "Laos",               "SE_ASIA"),
    "태국":           ("THA", "Thailand",           "SE_ASIA"),
    "인도네시아":     ("IDN", "Indonesia",          "SE_ASIA"),
    "필리핀":         ("PHL", "Philippines",        "SE_ASIA"),
    "말레이시아":     ("MYS", "Malaysia",           "SE_ASIA"),
    "싱가포르":       ("SGP", "Singapore",          "SE_ASIA"),
    "동티모르":       ("TLS", "Timor-Leste",        "SE_ASIA"),
    "브루나이":       ("BRN", "Brunei",             "SE_ASIA"),
    "미얀마(버마)":   ("MMR", "Myanmar",            "SE_ASIA"),

    # ── South Asia ────────────────────────────────────────────────────────────
    "방글라데시":     ("BGD", "Bangladesh",         "S_ASIA"),
    "파키스탄":       ("PAK", "Pakistan",           "S_ASIA"),
    "인도":           ("IND", "India",              "S_ASIA"),
    "네팔":           ("NPL", "Nepal",              "S_ASIA"),
    "스리랑카":       ("LKA", "Sri Lanka",          "S_ASIA"),
    "아프가니스탄":   ("AFG", "Afghanistan",        "S_ASIA"),
    "부탄":           ("BTN", "Bhutan",             "S_ASIA"),
    "몰디브":         ("MDV", "Maldives",           "S_ASIA"),

    # ── Central Asia ──────────────────────────────────────────────────────────
    "우즈베키스탄":   ("UZB", "Uzbekistan",         "C_ASIA"),
    "카자흐스탄":     ("KAZ", "Kazakhstan",         "C_ASIA"),
    "키르기스스탄":   ("KGZ", "Kyrgyzstan",         "C_ASIA"),
    "타지키스탄":     ("TJK", "Tajikistan",         "C_ASIA"),
    "투르크메니스탄": ("TKM", "Turkmenistan",       "C_ASIA"),
    "아제르바이잔":   ("AZE", "Azerbaijan",         "C_ASIA"),
    "조지아":         ("GEO", "Georgia",            "C_ASIA"),
    "아르메니아":     ("ARM", "Armenia",            "C_ASIA"),

    # ── Middle East & North Africa ────────────────────────────────────────────
    "이라크":         ("IRQ", "Iraq",               "MENA"),
    "예멘":           ("YEM", "Yemen",              "MENA"),
    "이란":           ("IRN", "Iran",               "MENA"),
    "시리아":         ("SYR", "Syria",              "MENA"),
    "요르단":         ("JOR", "Jordan",             "MENA"),
    "레바논":         ("LBN", "Lebanon",            "MENA"),
    "팔레스타인":     ("PSE", "Palestine",          "MENA"),
    "이스라엘":       ("ISR", "Israel",             "MENA"),
    "사우디아라비아": ("SAU", "Saudi Arabia",       "MENA"),
    "쿠웨이트":       ("KWT", "Kuwait",             "MENA"),
    "아랍에미리트":   ("ARE", "UAE",                "MENA"),
    "카타르":         ("QAT", "Qatar",              "MENA"),
    "오만":           ("OMN", "Oman",               "MENA"),
    "바레인":         ("BHR", "Bahrain",            "MENA"),
    "튀르키예":       ("TUR", "Türkiye",            "MENA"),
    "터키":           ("TUR", "Türkiye",            "MENA"),
    "이집트":         ("EGY", "Egypt",              "N_AFRICA"),
    "리비아":         ("LBY", "Libya",              "N_AFRICA"),
    "튀니지":         ("TUN", "Tunisia",            "N_AFRICA"),
    "알제리":         ("DZA", "Algeria",            "N_AFRICA"),
    "모로코":         ("MAR", "Morocco",            "N_AFRICA"),

    # ── Sub-Saharan Africa — East ──────────────────────────────────────────────
    "에티오피아":     ("ETH", "Ethiopia",           "E_AFRICA"),
    "케냐":           ("KEN", "Kenya",              "E_AFRICA"),
    "탄자니아":       ("TZA", "Tanzania",           "E_AFRICA"),
    "우간다":         ("UGA", "Uganda",             "E_AFRICA"),
    "르완다":         ("RWA", "Rwanda",             "E_AFRICA"),
    "소말리아":       ("SOM", "Somalia",            "E_AFRICA"),
    "남수단":         ("SSD", "South Sudan",        "E_AFRICA"),
    "수단":           ("SDN", "Sudan",              "E_AFRICA"),
    "지부티":         ("DJI", "Djibouti",           "E_AFRICA"),
    "에리트레아":     ("ERI", "Eritrea",            "E_AFRICA"),
    "마다가스카르":   ("MDG", "Madagascar",         "E_AFRICA"),
    "잠비아":         ("ZMB", "Zambia",             "E_AFRICA"),
    "짐바브웨":       ("ZWE", "Zimbabwe",           "E_AFRICA"),
    "말라위":         ("MWI", "Malawi",             "E_AFRICA"),
    "모잠비크":       ("MOZ", "Mozambique",         "SE_AFRICA"),
    "마다가스카르":   ("MDG", "Madagascar",         "SE_AFRICA"),

    # ── Sub-Saharan Africa — West ──────────────────────────────────────────────
    "나이지리아":     ("NGA", "Nigeria",            "W_AFRICA"),
    "가나":           ("GHA", "Ghana",              "W_AFRICA"),
    "세네갈":         ("SEN", "Senegal",            "W_AFRICA"),
    "말리":           ("MLI", "Mali",               "W_AFRICA"),
    "부르키나파소":   ("BFA", "Burkina Faso",       "W_AFRICA"),
    "코트디부아르":   ("CIV", "Côte d'Ivoire",      "W_AFRICA"),
    "기니":           ("GIN", "Guinea",             "W_AFRICA"),
    "시에라리온":     ("SLE", "Sierra Leone",       "W_AFRICA"),
    "라이베리아":     ("LBR", "Liberia",            "W_AFRICA"),
    "카메룬":         ("CMR", "Cameroon",           "C_AFRICA"),
    "콩고민주공화국": ("COD", "DRC",                "C_AFRICA"),
    "DR콩고":         ("COD", "DRC",                "C_AFRICA"),
    "콩고민주공화국(DR콩고)": ("COD", "DRC",        "C_AFRICA"),
    "콩고공화국":     ("COG", "Congo",              "C_AFRICA"),
    "니제르":         ("NER", "Niger",              "W_AFRICA"),
    "차드":           ("TCD", "Chad",               "C_AFRICA"),
    "중앙아프리카공화국": ("CAF", "CAR",            "C_AFRICA"),

    # ── Sub-Saharan Africa — South ─────────────────────────────────────────────
    "남아프리카공화국":("ZAF", "South Africa",      "S_AFRICA"),
    "남아공":         ("ZAF", "South Africa",       "S_AFRICA"),
    "앙골라":         ("AGO", "Angola",             "S_AFRICA"),
    "나미비아":       ("NAM", "Namibia",            "S_AFRICA"),
    "보츠와나":       ("BWA", "Botswana",           "S_AFRICA"),
    "레소토":         ("LSO", "Lesotho",            "S_AFRICA"),
    "에스와티니":     ("SWZ", "Eswatini",           "S_AFRICA"),

    # ── Americas ──────────────────────────────────────────────────────────────
    "아이티":         ("HTI", "Haiti",              "AMERICAS"),
    "쿠바":           ("CUB", "Cuba",               "AMERICAS"),
    "도미니카공화국": ("DOM", "Dominican Republic", "AMERICAS"),
    "자메이카":       ("JAM", "Jamaica",            "AMERICAS"),
    "과테말라":       ("GTM", "Guatemala",          "AMERICAS"),
    "온두라스":       ("HND", "Honduras",           "AMERICAS"),
    "엘살바도르":     ("SLV", "El Salvador",        "AMERICAS"),
    "니카라과":       ("NIC", "Nicaragua",          "AMERICAS"),
    "코스타리카":     ("CRI", "Costa Rica",         "AMERICAS"),
    "파나마":         ("PAN", "Panama",             "AMERICAS"),
    "콜롬비아":       ("COL", "Colombia",           "AMERICAS"),
    "베네수엘라":     ("VEN", "Venezuela",          "AMERICAS"),
    "에콰도르":       ("ECU", "Ecuador",            "AMERICAS"),
    "페루":           ("PER", "Peru",               "AMERICAS"),
    "볼리비아":       ("BOL", "Bolivia",            "AMERICAS"),
    "브라질":         ("BRA", "Brazil",             "AMERICAS"),
    "파라과이":       ("PRY", "Paraguay",           "AMERICAS"),
    "우루과이":       ("URY", "Uruguay",            "AMERICAS"),
    "아르헨티나":     ("ARG", "Argentina",          "AMERICAS"),
    "칠레":           ("CHL", "Chile",              "AMERICAS"),
    "멕시코":         ("MEX", "Mexico",             "AMERICAS"),
    "미국":           ("USA", "United States",      "AMERICAS"),

    # ── Pacific ───────────────────────────────────────────────────────────────
    "파푸아뉴기니":   ("PNG", "Papua New Guinea",   "PACIFIC"),
    "피지":           ("FJI", "Fiji",               "PACIFIC"),
    "솔로몬제도":     ("SLB", "Solomon Islands",    "PACIFIC"),
    "바누아투":       ("VUT", "Vanuatu",            "PACIFIC"),
    "사모아":         ("WSM", "Samoa",              "PACIFIC"),
    "통가":           ("TON", "Tonga",              "PACIFIC"),
    "마이크로네시아연방": ("FSM", "Micronesia",     "PACIFIC"),
    "팔라우":         ("PLW", "Palau",              "PACIFIC"),
    "마셜제도":       ("MHL", "Marshall Islands",   "PACIFIC"),
    "키리바시":       ("KIR", "Kiribati",           "PACIFIC"),
    "투발루":         ("TUV", "Tuvalu",             "PACIFIC"),
    "나우루":         ("NRU", "Nauru",              "PACIFIC"),

    # ── Europe (ODA recipients) ───────────────────────────────────────────────
    "우크라이나":     ("UKR", "Ukraine",            "E_EUROPE"),
    "몰도바":         ("MDA", "Moldova",            "E_EUROPE"),
    "벨라루스":       ("BLR", "Belarus",            "E_EUROPE"),
    "세르비아":       ("SRB", "Serbia",             "SE_EUROPE"),
    "북마케도니아":   ("MKD", "North Macedonia",    "SE_EUROPE"),
    "보스니아헤르체고비나": ("BIH", "Bosnia",       "SE_EUROPE"),
    "코소보":         ("XKX", "Kosovo",             "SE_EUROPE"),
    "알바니아":       ("ALB", "Albania",            "SE_EUROPE"),
    "몬테네그로":     ("MNE", "Montenegro",         "SE_EUROPE"),

    # ── Regional / unallocated (from ODA dataset) ─────────────────────────────
    "그외 지역 또는 다수국가 (미배분)": ("---", "Unallocated/Multi",   "MULTI"),
    "아시아 지역 (미배분)":            ("---", "Asia Unallocated",     "ASIA"),
    "아프리카 지역 (미배분)":          ("---", "Africa Unallocated",   "AFRICA"),
    "아메리카 지역 (미배분)":          ("---", "Americas Unallocated", "AMERICAS"),
    "오세아니아 지역 (미배분)":        ("---", "Oceania Unallocated",  "PACIFIC"),
    "유럽 지역 (미배분)":              ("---", "Europe Unallocated",   "EUROPE"),
    "중동 지역 (미배분)":              ("---", "MENA Unallocated",     "MENA"),
}

# Reverse lookup: ISO3 → Korean names (list, since multiple may map to same ISO3)
ISO3_TO_KO: dict[str, list[str]] = {}
for ko, (iso3, en, region) in COUNTRY_MAP.items():
    ISO3_TO_KO.setdefault(iso3, []).append(ko)

# Set of all Korean country strings for fast membership test
ALL_KO_NAMES: frozenset[str] = frozenset(COUNTRY_MAP.keys())


def detect_countries(text: str) -> list[str]:
    """
    Return ISO3 codes for all countries mentioned in text.
    Uses substring matching on Korean country names.
    Returns unique codes (an article mentioning 파키스탄 twice → ['PAK'] once).
    """
    if not text:
        return []
    found: set[str] = set()
    for ko, (iso3, _, _) in COUNTRY_MAP.items():
        if iso3 != "---" and ko in text:
            found.add(iso3)
    return sorted(found)


def ko_to_iso3(ko_name: str) -> str | None:
    """Look up a Korean country name → ISO3 code."""
    entry = COUNTRY_MAP.get(ko_name)
    return entry[0] if entry else None


def ko_to_english(ko_name: str) -> str | None:
    """Look up a Korean country name → English name."""
    entry = COUNTRY_MAP.get(ko_name)
    return entry[1] if entry else None
