"""
Global crisis and shock event database (2007-2025).

Each event is coded with:
  year, month        — event onset (first full reporting month)
  event_type         — health | conflict | food | disaster
  label              — unique identifier
  severity           — 1=moderate, 2=major, 3=catastrophic (OCHA/EM-DAT scale)
  affected_sdgs      — list of primarily affected SDGs
  region             — UN region code
  description        — brief description

Sources:
  - EM-DAT International Disaster Database (emdat.be)
  - OCHA Financial Tracking Service (fts.unocha.org)
  - WHO Global Health Observatory
  - UNHCR Refugee Data Finder
  - FAO GIEWS food emergency reports
  - UCDP/PRIO Armed Conflict Dataset
"""

from __future__ import annotations
import pandas as pd


CRISIS_EVENTS: list[dict] = [

    # ── Health crises ─────────────────────────────────────────────────────────
    {"year": 2009, "month": 4,  "event_type": "health",   "severity": 2,
     "label": "H1N1_PANDEMIC",          "affected_sdgs": [3],
     "region": "GLOBAL",   "description": "H1N1 influenza pandemic; WHO declared PHE June 2009"},

    {"year": 2012, "month": 9,  "event_type": "health",   "severity": 1,
     "label": "MERS_COV_EMERGENCE",     "affected_sdgs": [3],
     "region": "MENA",     "description": "MERS-CoV first identified, Saudi Arabia"},

    {"year": 2014, "month": 3,  "event_type": "health",   "severity": 3,
     "label": "EBOLA_WEST_AFRICA",      "affected_sdgs": [3, 1],
     "region": "W_AFRICA",  "description": "Ebola epidemic Guinea/Sierra Leone/Liberia; 11,300 deaths"},

    {"year": 2015, "month": 5,  "event_type": "health",   "severity": 2,
     "label": "MERS_KOREA_OUTBREAK",    "affected_sdgs": [3],
     "region": "E_ASIA",   "description": "MERS outbreak in South Korea; 186 cases, 38 deaths"},

    {"year": 2016, "month": 2,  "event_type": "health",   "severity": 2,
     "label": "ZIKA_PHE",               "affected_sdgs": [3, 2],
     "region": "AMERICAS", "description": "Zika virus; WHO declared PHEIC Feb 2016"},

    {"year": 2019, "month": 7,  "event_type": "health",   "severity": 2,
     "label": "EBOLA_DRC_2019",         "affected_sdgs": [3],
     "region": "C_AFRICA",  "description": "Ebola outbreak DRC; WHO declared PHEIC Jul 2019"},

    {"year": 2020, "month": 1,  "event_type": "health",   "severity": 3,
     "label": "COVID19_PANDEMIC",       "affected_sdgs": [3, 1, 8, 4, 2],
     "region": "GLOBAL",   "description": "COVID-19; WHO PHEIC Jan 2020, pandemic declared Mar 2020"},

    {"year": 2022, "month": 5,  "event_type": "health",   "severity": 1,
     "label": "MPOX_OUTBREAK",          "affected_sdgs": [3],
     "region": "GLOBAL",   "description": "Monkeypox (mpox) global outbreak; WHO PHEIC Jul 2022"},

    # ── Conflict & displacement crises ────────────────────────────────────────
    {"year": 2008, "month": 12, "event_type": "conflict",  "severity": 2,
     "label": "GAZA_WAR_2008",          "affected_sdgs": [16, 1, 3],
     "region": "MENA",     "description": "Israel-Gaza conflict (Operation Cast Lead)"},

    {"year": 2011, "month": 2,  "event_type": "conflict",  "severity": 2,
     "label": "LIBYA_CIVIL_WAR",        "affected_sdgs": [16, 1],
     "region": "N_AFRICA",  "description": "Libyan civil war; NATO intervention Mar 2011"},

    {"year": 2011, "month": 3,  "event_type": "conflict",  "severity": 3,
     "label": "SYRIA_CIVIL_WAR",        "affected_sdgs": [16, 1, 3, 4],
     "region": "MENA",     "description": "Syrian civil war onset; major displacement crisis"},

    {"year": 2013, "month": 12, "event_type": "conflict",  "severity": 2,
     "label": "SOUTH_SUDAN_CIVIL_WAR",  "affected_sdgs": [16, 1, 2],
     "region": "E_AFRICA",  "description": "South Sudan civil war; major famine risk"},

    {"year": 2014, "month": 2,  "event_type": "conflict",  "severity": 2,
     "label": "UKRAINE_CRIMEA",         "affected_sdgs": [16, 17],
     "region": "E_EUROPE",  "description": "Russia annexation of Crimea; Ukraine conflict begins"},

    {"year": 2014, "month": 6,  "event_type": "conflict",  "severity": 3,
     "label": "ISIS_EXPANSION",         "affected_sdgs": [16, 1],
     "region": "MENA",     "description": "ISIS territorial expansion Iraq/Syria; mass displacement"},

    {"year": 2015, "month": 3,  "event_type": "conflict",  "severity": 2,
     "label": "YEMEN_CIVIL_WAR",        "affected_sdgs": [16, 1, 2, 3],
     "region": "MENA",     "description": "Yemen civil war; Saudi-led coalition intervention"},

    {"year": 2015, "month": 9,  "event_type": "conflict",  "severity": 2,
     "label": "EUROPE_REFUGEE_CRISIS",  "affected_sdgs": [10, 1, 16],
     "region": "EUROPE",   "description": "European refugee/migration crisis; peak Syrian displacement"},

    {"year": 2017, "month": 8,  "event_type": "conflict",  "severity": 3,
     "label": "ROHINGYA_CRISIS",        "affected_sdgs": [16, 1, 10],
     "region": "SE_ASIA",  "description": "Rohingya genocide; 700,000+ fled Myanmar to Bangladesh"},

    {"year": 2022, "month": 2,  "event_type": "conflict",  "severity": 3,
     "label": "UKRAINE_WAR_2022",       "affected_sdgs": [16, 1, 2, 7, 13],
     "region": "E_EUROPE",  "description": "Russia full-scale invasion of Ukraine; global shockwaves"},

    {"year": 2023, "month": 4,  "event_type": "conflict",  "severity": 2,
     "label": "SUDAN_CONFLICT_2023",    "affected_sdgs": [16, 1, 2],
     "region": "NE_AFRICA",  "description": "Sudan RSF-SAF conflict; major humanitarian crisis"},

    {"year": 2023, "month": 10, "event_type": "conflict",  "severity": 3,
     "label": "GAZA_WAR_2023",          "affected_sdgs": [16, 1, 3, 6],
     "region": "MENA",     "description": "Israel-Gaza war; large-scale civilian casualties"},

    # ── Food security crises ──────────────────────────────────────────────────
    {"year": 2008, "month": 1,  "event_type": "food",      "severity": 2,
     "label": "GLOBAL_FOOD_PRICE_2008", "affected_sdgs": [2, 1],
     "region": "GLOBAL",   "description": "Global food price crisis; commodity prices doubled"},

    {"year": 2011, "month": 7,  "event_type": "food",      "severity": 3,
     "label": "HORN_OF_AFRICA_FAMINE",  "affected_sdgs": [2, 1, 3],
     "region": "E_AFRICA",  "description": "Famine in Somalia, Ethiopia, Kenya; UN declared famine"},

    {"year": 2012, "month": 1,  "event_type": "food",      "severity": 2,
     "label": "SAHEL_FOOD_CRISIS",      "affected_sdgs": [2, 1],
     "region": "W_AFRICA",  "description": "Sahel regional food crisis; 18 million affected"},

    {"year": 2017, "month": 2,  "event_type": "food",      "severity": 3,
     "label": "E_AFRICA_FOOD_2017",     "affected_sdgs": [2, 1, 3],
     "region": "E_AFRICA",  "description": "East Africa famine; UN declared famine in S. Sudan"},

    {"year": 2022, "month": 3,  "event_type": "food",      "severity": 3,
     "label": "GLOBAL_FOOD_CRISIS_2022","affected_sdgs": [2, 1],
     "region": "GLOBAL",   "description": "Global food crisis driven by Ukraine war; fertilizer/grain shock"},

    {"year": 2023, "month": 1,  "event_type": "food",      "severity": 2,
     "label": "HORN_DROUGHT_2023",      "affected_sdgs": [2, 1, 13],
     "region": "E_AFRICA",  "description": "Prolonged drought Horn of Africa; worst in 40 years"},

    # ── Natural disasters ─────────────────────────────────────────────────────
    {"year": 2008, "month": 5,  "event_type": "disaster",  "severity": 3,
     "label": "MYANMAR_CYCLONE_NARGIS", "affected_sdgs": [13, 1, 11],
     "region": "SE_ASIA",  "description": "Cyclone Nargis Myanmar; 138,000 deaths"},

    {"year": 2008, "month": 5,  "event_type": "disaster",  "severity": 3,
     "label": "CHINA_SICHUAN_EQ",       "affected_sdgs": [11, 1],
     "region": "E_ASIA",   "description": "Sichuan earthquake China; 68,000 deaths"},

    {"year": 2010, "month": 1,  "event_type": "disaster",  "severity": 3,
     "label": "HAITI_EARTHQUAKE",       "affected_sdgs": [11, 1, 3, 16],
     "region": "AMERICAS", "description": "Haiti earthquake Mw 7.0; 200,000+ deaths"},

    {"year": 2010, "month": 7,  "event_type": "disaster",  "severity": 3,
     "label": "PAKISTAN_FLOODS_2010",   "affected_sdgs": [13, 1, 2, 6],
     "region": "S_ASIA",   "description": "Pakistan monsoon floods; 20M affected, 1/5 of country submerged"},

    {"year": 2011, "month": 3,  "event_type": "disaster",  "severity": 3,
     "label": "JAPAN_TOHOKU",           "affected_sdgs": [11, 7, 13],
     "region": "E_ASIA",   "description": "Tohoku earthquake/tsunami/Fukushima; 18,000 deaths"},

    {"year": 2011, "month": 7,  "event_type": "disaster",  "severity": 2,
     "label": "THAILAND_FLOODS_2011",   "affected_sdgs": [11, 9, 2],
     "region": "SE_ASIA",  "description": "Thailand floods; major supply chain disruption"},

    {"year": 2013, "month": 11, "event_type": "disaster",  "severity": 3,
     "label": "PHILIPPINES_HAIYAN",     "affected_sdgs": [13, 11, 1],
     "region": "SE_ASIA",  "description": "Typhoon Haiyan (Yolanda) Philippines; 6,300 deaths"},

    {"year": 2015, "month": 4,  "event_type": "disaster",  "severity": 3,
     "label": "NEPAL_EARTHQUAKE",       "affected_sdgs": [11, 1, 3],
     "region": "S_ASIA",   "description": "Nepal earthquake Mw 7.8; 9,000 deaths"},

    {"year": 2016, "month": 4,  "event_type": "disaster",  "severity": 2,
     "label": "ECUADOR_EARTHQUAKE",     "affected_sdgs": [11, 1],
     "region": "AMERICAS", "description": "Ecuador earthquake Mw 7.8; 650 deaths"},

    {"year": 2018, "month": 9,  "event_type": "disaster",  "severity": 3,
     "label": "INDONESIA_SULAWESI",     "affected_sdgs": [11, 1, 13],
     "region": "SE_ASIA",  "description": "Sulawesi earthquake-tsunami Indonesia; 4,300 deaths"},

    {"year": 2019, "month": 3,  "event_type": "disaster",  "severity": 2,
     "label": "CYCLONE_IDAI",           "affected_sdgs": [13, 11, 1],
     "region": "SE_AFRICA", "description": "Cyclone Idai Mozambique/Zimbabwe/Malawi; 1,300 deaths"},

    {"year": 2021, "month": 8,  "event_type": "disaster",  "severity": 2,
     "label": "HAITI_EARTHQUAKE_2021",  "affected_sdgs": [11, 1],
     "region": "AMERICAS", "description": "Haiti earthquake Mw 7.2; 2,200 deaths"},

    {"year": 2022, "month": 6,  "event_type": "disaster",  "severity": 3,
     "label": "PAKISTAN_FLOODS_2022",   "affected_sdgs": [13, 1, 2, 6],
     "region": "S_ASIA",   "description": "Pakistan floods; 1,700 deaths, 33M affected, 1/3 submerged"},

    {"year": 2023, "month": 2,  "event_type": "disaster",  "severity": 3,
     "label": "TURKEY_SYRIA_EQ",        "affected_sdgs": [11, 1, 3],
     "region": "MENA",     "description": "Turkey-Syria earthquake Mw 7.8; 56,000 deaths"},

    {"year": 2023, "month": 9,  "event_type": "disaster",  "severity": 2,
     "label": "MOROCCO_EQ_LIBYA_FLOOD", "affected_sdgs": [11, 1, 6],
     "region": "N_AFRICA",  "description": "Morocco earthquake (Mw 6.8) + Libya floods (Derna) within 1 week"},
]


# ── Public API ────────────────────────────────────────────────────────────────

def build_crisis_panel(start_year: int = 2007, end_year: int = 2025) -> pd.DataFrame:
    """
    Return a month-level DataFrame with crisis indicator columns.

    Columns (one per event_type × severity combination, plus SDG-specific flags):
      crisis_health_{1,2,3}     — health crisis of that severity active this month
      crisis_conflict_{1,2,3}
      crisis_food_{1,2,3}
      crisis_disaster_{1,2,3}
      crisis_sdg_{1..17}        — any crisis event affecting this SDG this month
      crisis_any                — any crisis of any type/severity
      crisis_severity_max       — maximum severity active this month (0 if none)
      crisis_count              — number of distinct events active this month

    Each event is active only in its onset month (binary indicator).
    For regression use, consider extending to duration windows if needed.
    """
    months = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-01",
        freq="MS",
    )
    df = pd.DataFrame({"year": months.year, "month": months.month})

    for et in ["health", "conflict", "food", "disaster"]:
        for sv in [1, 2, 3]:
            df[f"crisis_{et}_{sv}"] = 0

    for sdg in range(1, 18):
        df[f"crisis_sdg_{sdg}"] = 0

    df["crisis_any"]          = 0
    df["crisis_severity_max"] = 0
    df["crisis_count"]        = 0

    for ev in CRISIS_EVENTS:
        row_mask = (df["year"] == ev["year"]) & (df["month"] == ev["month"])
        et = ev["event_type"]
        sv = ev["severity"]
        df.loc[row_mask, f"crisis_{et}_{sv}"]   = 1
        df.loc[row_mask, "crisis_any"]           = 1
        df.loc[row_mask, "crisis_count"]        += 1
        df.loc[row_mask, "crisis_severity_max"]  = df.loc[row_mask, "crisis_severity_max"].clip(lower=sv)
        for sdg in ev.get("affected_sdgs", []):
            df.loc[row_mask, f"crisis_sdg_{sdg}"] = 1

    return df


def get_events_df() -> pd.DataFrame:
    """Return the raw events list as a DataFrame."""
    return pd.DataFrame(CRISIS_EVENTS)
