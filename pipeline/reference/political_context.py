"""
Korean political context reference data.

Covers 2003-01 through 2025-12. Generates a month-level DataFrame with
one row per year-month containing all political indicators.

Sources:
  - Presidential terms: National Archives of Korea (국가기록원)
  - Elections: National Election Commission (중앙선거관리위원회), info.nec.go.kr
  - DAC peer reviews: OECD DAC, oecd.org/dac/peer-reviews/korea
  - ODA policy: KOICA, MOFA Korea, Korean Official Gazette
  - International events: UN, UNFCCC, OECD official records
"""

from __future__ import annotations
from datetime import date
import pandas as pd


# ── Presidential administrations ─────────────────────────────────────────────

ADMINISTRATIONS: list[dict] = [
    {
        "president":      "Roh Moo-hyun",
        "president_ko":   "노무현",
        "party":          "Uri Party",
        "ideology":       "centre-left",
        "start":          date(2003, 2, 25),
        "end":            date(2008, 2, 24),
        "acting_periods": [],
    },
    {
        "president":      "Lee Myung-bak",
        "president_ko":   "이명박",
        "party":          "GNP",
        "ideology":       "centre-right",
        "start":          date(2008, 2, 25),
        "end":            date(2013, 2, 24),
        "acting_periods": [],
    },
    {
        "president":      "Park Geun-hye",
        "president_ko":   "박근혜",
        "party":          "Saenuri",
        "ideology":       "right",
        "start":          date(2013, 2, 25),
        "end":            date(2017, 5, 9),
        # Acting president (Hwang Kyo-ahn) from impeachment vote to Constitutional Court ruling
        "acting_periods": [(date(2016, 12, 9), date(2017, 3, 10))],
        "impeached":      date(2017, 3, 10),
    },
    {
        "president":      "Moon Jae-in",
        "president_ko":   "문재인",
        "party":          "Democratic Party",
        "ideology":       "centre-left",
        "start":          date(2017, 5, 10),
        "end":            date(2022, 5, 9),
        "acting_periods": [],
    },
    {
        "president":      "Yoon Suk-yeol",
        "president_ko":   "윤석열",
        "party":          "PPP",
        "ideology":       "right",
        "start":          date(2022, 5, 10),
        "end":            date(2027, 5, 9),   # scheduled end
        "acting_periods": [],
    },
]

# ── Elections ─────────────────────────────────────────────────────────────────

NATIONAL_ASSEMBLY_ELECTIONS: list[date] = [
    date(2008, 4, 9),   # 18th National Assembly
    date(2012, 4, 11),  # 19th National Assembly
    date(2016, 4, 13),  # 20th National Assembly
    date(2020, 4, 15),  # 21st National Assembly
    date(2024, 4, 10),  # 22nd National Assembly
]

PRESIDENTIAL_ELECTIONS: list[date] = [
    date(2007, 12, 19),   # Lee Myung-bak
    date(2012, 12, 19),   # Park Geun-hye
    date(2017, 5, 9),     # Moon Jae-in (special election)
    date(2022, 3, 9),     # Yoon Suk-yeol
]

# ── ODA policy events ─────────────────────────────────────────────────────────
# (year, month, label)
KOREA_ODA_EVENTS: list[tuple[int, int, str]] = [
    (2010,  1, "DAC_ACCESSION"),          # Korea became 24th DAC member
    (2010, 11, "ODA_BASIC_ACT"),          # 국제개발협력기본법 enacted
    (2011,  1, "ODA_STRATEGY_1_START"),   # 1st mid-term strategy 2011-2015
    (2012,  1, "DAC_PEER_REVIEW"),        # OECD DAC peer review
    (2015,  7, "ODA_STRATEGY_2_START"),   # 2nd mid-term strategy 2016-2020
    (2017,  7, "KOREA_VNR_1"),            # 1st Voluntary National Review (HLPF)
    (2018,  4, "DAC_PEER_REVIEW"),        # OECD DAC peer review
    (2018,  7, "SDG_NATIONAL_PLAN"),      # Korea national SDG action plan
    (2020,  7, "ODA_STRATEGY_3_START"),   # 3rd mid-term strategy 2021-2025
    (2021,  7, "KOREA_VNR_2"),            # 2nd Voluntary National Review (HLPF)
    (2022,  6, "DAC_PEER_REVIEW"),        # OECD DAC peer review
]

# ── International development/climate events ──────────────────────────────────

INTERNATIONAL_EVENTS: list[tuple[int, int, str]] = [
    (2010, 11, "G20_SEOUL"),              # G20 Seoul Summit (Korea hosted)
    (2015,  9, "SDG_ADOPTION"),           # UN adopted 2030 Agenda / 17 SDGs
    (2015, 12, "COP21_PARIS"),            # Paris Agreement adopted
    (2016,  7, "HLPF_HIGH_LEVEL"),        # First HLPF under ECOSOC
    (2016, 11, "PARIS_AGREEMENT_FORCE"),  # Paris Agreement entered into force
    (2017,  7, "HLPF_HIGH_LEVEL"),
    (2018,  7, "HLPF_HIGH_LEVEL"),
    (2019,  7, "HLPF_HIGH_LEVEL"),
    (2019,  9, "UN_SDG_SUMMIT"),          # UN SDG Summit (mid-term review)
    (2021, 11, "COP26_GLASGOW"),
    (2022, 11, "COP27_EGYPT"),
    (2023, 12, "COP28_UAE"),
    (2024,  9, "UN_SUMMIT_FUTURE"),
    (2024, 11, "COP29_BAKU"),
]


# ── Public API ────────────────────────────────────────────────────────────────

def build_political_panel(start_year: int = 2007, end_year: int = 2025) -> pd.DataFrame:
    """
    Return a DataFrame indexed by (year, month) with political context variables.

    Columns:
      president          str   — surname, e.g. "Lee Myung-bak"
      ideology           str   — "centre-left" | "centre-right" | "right"
      transition_month   bool  — first month of new administration
      first_year         bool  — first 12 months of administration
      acting_president   bool  — period when acting president governed
      na_election_year   bool  — national assembly election year
      presidential_election_year bool
      oda_event          str   — label if ODA event occurred, else ""
      intl_event         str   — label if international event occurred, else ""
    """
    months = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-01",
        freq="MS",
    )
    df = pd.DataFrame({"year": months.year, "month": months.month})
    df["ym"] = pd.to_datetime(df[["year", "month"]].assign(day=1))

    # Presidential variables
    df["president"]   = ""
    df["ideology"]    = ""
    df["transition_month"] = False
    df["first_year"]       = False
    df["acting_president"] = False

    for adm in ADMINISTRATIONS:
        s = pd.Timestamp(adm["start"])
        e = pd.Timestamp(adm["end"])
        mask = (df["ym"] >= s.to_period("M").to_timestamp()) & \
               (df["ym"] <= e.to_period("M").to_timestamp())
        df.loc[mask, "president"] = adm["president"]
        df.loc[mask, "ideology"]  = adm["ideology"]

        # First month of term
        trans_mask = (df["year"] == s.year) & (df["month"] == s.month)
        df.loc[trans_mask, "transition_month"] = True

        # First 12 months
        end_first_year = s + pd.DateOffset(months=11)
        first_mask = (df["ym"] >= s.to_period("M").to_timestamp()) & \
                     (df["ym"] <= end_first_year.to_period("M").to_timestamp())
        df.loc[first_mask, "first_year"] = True

        # Acting president periods
        for act_start, act_end in adm.get("acting_periods", []):
            act_s = pd.Timestamp(act_start)
            act_e = pd.Timestamp(act_end)
            act_mask = (df["ym"] >= act_s.to_period("M").to_timestamp()) & \
                       (df["ym"] <= act_e.to_period("M").to_timestamp())
            df.loc[act_mask, "acting_president"] = True

    # Election years
    na_years = {d.year for d in NATIONAL_ASSEMBLY_ELECTIONS}
    pres_years = {d.year for d in PRESIDENTIAL_ELECTIONS}
    df["na_election_year"]   = df["year"].isin(na_years)
    df["presidential_election_year"] = df["year"].isin(pres_years)

    # ODA events
    oda_map: dict[tuple, str] = {(y, m): lbl for y, m, lbl in KOREA_ODA_EVENTS}
    df["oda_policy_event"] = df.apply(
        lambda r: oda_map.get((r["year"], r["month"]), ""), axis=1
    )

    # International events
    intl_map: dict[tuple, str] = {(y, m): lbl for y, m, lbl in INTERNATIONAL_EVENTS}
    df["intl_event"] = df.apply(
        lambda r: intl_map.get((r["year"], r["month"]), ""), axis=1
    )

    return df[["year", "month", "president", "ideology",
               "transition_month", "first_year", "acting_president",
               "na_election_year", "presidential_election_year",
               "oda_policy_event", "intl_event"]]
