"""
ODA dataset preprocessor.

Input:  src/oda/korea_oda data.xlsx  (~83k rows, 2010-2023)

Outputs (all in src/processed/oda/):
  oda_clean.csv               — project-level, full normalized dataset with SDG + ISO3
  oda_sdg_annual.csv          — year × SDG aggregation  (input for Panel A/B)
  oda_country_sdg_annual.csv  — year × country_iso3 × SDG  (input for Panel C, primary)

SDG assignment methodology (three-tier priority):
  1. Direct SDG tag — parsed from the dataset's own 'SDGs' field (~27% of rows).
     Korea's KOICA/EDCF have reported SDG markers since 2018; values like '3',
     '16.5' are normalised to the goal integer.

  2. CRS sector code → SDG mapping (~65% of rows).
     Uses the 5-digit DAC CRS purpose code from 'sector_code', mapped via the
     OECD's official crosswalk:
       OECD (2015). "Preliminary Mapping Between Sustainable Development Goals
       and Targets With CRS Purpose Codes and Markers", Annex 3.
       DCD/DAC/STAT(2015)9. https://one.oecd.org/document/DCD/DAC/STAT(2015)9
     Methodology consistent with Pincet, Okabe & Pawelczyk (2019), "Linking Aid
     to the Sustainable Development Goals: A Machine Learning Approach", OECD
     Development Co-operation Working Paper No. 52.

  3. Cross-cutting policy markers — for rows with no code match, significant
     climate/gender/biodiversity markers are used as a fallback (~0% currently).
"""

from __future__ import annotations

import re
import sys
import io
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR  = Path(__file__).parent.parent
ODA_FILE  = BASE_DIR / "src" / "oda" / "korea_oda data.xlsx"
OUT_DIR   = BASE_DIR / "src" / "processed" / "oda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLEAN         = OUT_DIR / "oda_clean.csv"
OUT_SDG_ANNUAL    = OUT_DIR / "oda_sdg_annual.csv"
OUT_COUNTRY_SDG   = OUT_DIR / "oda_country_sdg_annual.csv"

sys.path.insert(0, str(Path(__file__).parent))
from reference.countries_ko import ko_to_iso3, ko_to_english

# ── Column rename map ─────────────────────────────────────────────────────────
COL_MAP = {
    "사업번호":            "project_id",
    "보고년도":            "year",
    "보고국가/기관":       "reporting_country",
    "자료제출기관":        "agency",
    "CRS번호":             "crs_id",
    "대륙명":              "region",
    "수원국":              "recipient_country",
    "사업실시기관명":      "implementing_org",
    "사업구분":            "project_type",
    "양자/다자간 구분":   "bilateral_multilateral",
    "자금구분":            "fund_type",
    "자금형태":            "aid_form",
    "원조유형":            "aid_type",
    "원조유형코드":        "aid_type_code",
    "사업명(한글)":        "project_name_ko",
    "사업명(영문)":        "project_name_en",
    "사업분야":            "sector",
    "사업분야코드":        "sector_code",
    "SDGs":                "sdg_tag_raw",
    "사업개시(예정)일":   "start_date",
    "사업완공(예정)일":   "end_date",
    "사업설명":            "description",
    "성평등":              "marker_gender",
    "환경 지원":           "marker_environment",
    "민주적/포용적 거버넌스": "marker_governance",
    "기후변화 완화":       "marker_climate_mitigation",
    "기후변화 적응":       "marker_climate_adaptation",
    "생물의 다양성":       "marker_biodiversity",
    "약정액\n[백만달러]":  "commitment_musd",
    "지출액\n[백만달러]":  "disbursement_musd",
    "순지출액\n[백만달러]":"net_disbursement_musd",
    "증여등가액\n[백만달러]": "grant_equivalent_musd",
}

# ── CRS purpose code → primary SDG mapping ───────────────────────────────────
# Source: OECD DCD/DAC/STAT(2015)9 Annex 3 — official CRS-SDG crosswalk.
# Lookup is 5-digit exact match first; falls back to 3-digit sector-group prefix.
# Where a sector spans multiple SDGs the most prominent goal is assigned
# (consistent with Pincet et al., 2019, OECD Working Paper No. 52).

CRS_EXACT_TO_SDG: dict[str, int] = {
    # SDG 1 — No Poverty
    "16010": 1, "16020": 1, "16050": 1,
    "72010": 1, "72040": 1, "72050": 1,
    "73010": 1,
    # SDG 2 — Zero Hunger
    "31161": 2, "31162": 2, "31163": 2, "31164": 2, "31165": 2, "31166": 2,
    "31182": 2, "31191": 2, "31192": 2, "31193": 2, "31194": 2, "31195": 2,
    "52010": 2,
    # SDG 3 — Good Health
    "12110": 3, "12181": 3, "12182": 3, "12191": 3, "12196": 3,
    "12220": 3, "12230": 3, "12240": 3, "12250": 3,
    "12261": 3, "12262": 3, "12263": 3, "12281": 3,
    "13010": 3, "13020": 3, "13030": 3, "13040": 3, "13081": 3,
    # SDG 4 — Quality Education
    "11110": 4, "11120": 4, "11130": 4, "11182": 4,
    "11220": 4, "11230": 4, "11231": 4, "11232": 4,
    "11240": 4, "11250": 4, "11260": 4,
    "11320": 4, "11330": 4, "11420": 4, "11430": 4,
    # SDG 5 — Gender Equality
    "15170": 5, "15180": 5, "42010": 5,
    # SDG 6 — Clean Water & Sanitation
    "14010": 6, "14015": 6, "14020": 6, "14021": 6, "14022": 6,
    "14030": 6, "14031": 6, "14032": 6,
    "14040": 6, "14050": 6, "14081": 6,
    # SDG 7 — Clean Energy
    "23010": 7, "23020": 7, "23030": 7, "23040": 7, "23050": 7,
    "23061": 7, "23062": 7, "23063": 7, "23064": 7, "23065": 7,
    "23066": 7, "23067": 7, "23068": 7, "23069": 7, "23070": 7,
    "23077": 7, "23081": 7, "23082": 7,
    # SDG 8 — Decent Work & Economic Growth
    "24010": 8, "24020": 8, "24030": 8, "24040": 8, "24050": 8, "24081": 8,
    "25010": 8, "25020": 8, "25030": 8, "25040": 8,
    "33110": 8, "33120": 8, "33130": 8, "33140": 8, "33150": 8, "33181": 8,
    # SDG 9 — Industry, Innovation & Infrastructure
    "21010": 9, "21020": 9, "21030": 9, "21040": 9, "21050": 9,
    "21061": 9, "21081": 9,
    "22010": 9, "22020": 9, "22030": 9, "22040": 9, "22081": 9,
    "32110": 9, "32120": 9, "32130": 9, "32140": 9,
    "32161": 9, "32162": 9, "32163": 9, "32164": 9, "32165": 9,
    "32166": 9, "32167": 9, "32168": 9, "32169": 9,
    "32170": 9, "32171": 9, "32172": 9, "32181": 9,
    # SDG 10 — Reduced Inequalities
    "15190": 10, "16061": 10, "16062": 10,
    # SDG 11 — Sustainable Cities
    "43030": 11, "43040": 11, "43050": 11,
    "15185": 11,
    # SDG 13 — Climate Action
    "41010": 13, "41020": 13, "41030": 13,
    "41040": 13, "41050": 13, "41081": 13, "41082": 13,
    "74010": 13,
    # SDG 14 — Life Below Water
    "31310": 14, "31320": 14, "31381": 14, "31382": 14, "31391": 14,
    # SDG 15 — Life on Land
    "31210": 15, "31220": 15, "31261": 15, "31281": 15, "31282": 15, "31291": 15,
    # SDG 16 — Peace, Justice & Strong Institutions
    "15110": 16, "15111": 16, "15112": 16, "15113": 16, "15114": 16,
    "15116": 16, "15117": 16, "15119": 16,
    "15120": 16, "15121": 16, "15122": 16, "15123": 16, "15124": 16,
    "15125": 16, "15126": 16, "15127": 16, "15128": 16, "15129": 16,
    "15130": 16, "15131": 16, "15132": 16, "15133": 16, "15134": 16,
    "15135": 16, "15136": 16, "15137": 16, "15139": 16,
    "15150": 16, "15151": 16, "15152": 16, "15153": 16, "15154": 16,
    "15155": 16, "15156": 16, "15160": 16, "15161": 16, "15162": 16,
    "15163": 16, "15164": 16,
    "15210": 16, "15220": 16, "15230": 16, "15240": 16, "15250": 16,
    "15261": 16,
    # SDG 17 — Partnerships for the Goals
    "60010": 17, "60020": 17, "60030": 17, "60040": 17,
    "60061": 17, "60062": 17, "60063": 17,
    "91010": 17,
}

CRS_PREFIX_TO_SDG: dict[str, int] = {
    "111": 4, "112": 4, "113": 4, "114": 4,
    "121": 3, "122": 3, "123": 3,
    "130": 3,
    "140": 6,
    "151": 16, "152": 16, "153": 16,
    "160": 1,
    "210": 9,
    "220": 9,
    "230": 7,
    "240": 8,
    "250": 8,
    "311": 2,
    "312": 15,
    "313": 14,
    "321": 9,
    "322": 9,
    "323": 11,
    "331": 8,
    "332": 8,
    "410": 13,
    "420": 5,
    "430": 11,
    "431": 2,
    "432": 11,
    "520": 2,
    "600": 17,
    "720": 1, "730": 1,
    "740": 13,
    "910": 17, "930": 17,
}


def parse_crs_code(sector_code_raw) -> Optional[str]:
    if pd.isna(sector_code_raw):
        return None
    first = str(sector_code_raw).split(",")[0].strip()
    digits = re.sub(r"\D", "", first)
    return digits[:5] if len(digits) >= 5 else (digits if digits else None)


def parse_sdg_tag(raw) -> Optional[int]:
    if pd.isna(raw):
        return None
    m = re.match(r"(\d+)", str(raw).strip())
    if m:
        val = int(m.group(1))
        return val if 1 <= val <= 17 else None
    return None


def crs_to_sdg(sector_code_raw) -> Optional[int]:
    code = parse_crs_code(sector_code_raw)
    if not code:
        return None
    if code in CRS_EXACT_TO_SDG:
        return CRS_EXACT_TO_SDG[code]
    return CRS_PREFIX_TO_SDG.get(code[:3])


def infer_sdg(row) -> Optional[int]:
    sdg = parse_sdg_tag(row.get("sdg_tag_raw"))
    if sdg:
        return sdg
    sdg = crs_to_sdg(row.get("sector_code"))
    if sdg:
        return sdg
    for col, sdg_num in [
        ("marker_climate_mitigation", 13),
        ("marker_climate_adaptation", 13),
        ("marker_biodiversity", 15),
        ("marker_gender", 5),
    ]:
        val = str(row.get(col, "") or "").strip()
        if val in ("주요", "부분"):
            return sdg_num
    return None


# ── Load & clean ──────────────────────────────────────────────────────────────

print("Loading ODA xlsx ...")
df_raw = pd.read_excel(ODA_FILE, sheet_name=0, dtype=str)
print(f"  {len(df_raw):,} rows, {len(df_raw.columns)} columns loaded.")

# Rename known columns
rename = {}
for ko, en in COL_MAP.items():
    matches = [c for c in df_raw.columns if c == ko]
    if matches:
        rename[matches[0]] = en

df = df_raw.rename(columns=rename)

# Coerce numeric columns
for col in ["commitment_musd", "disbursement_musd", "net_disbursement_musd",
            "grant_equivalent_musd"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

if "year" in df.columns:
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

# ── Country ISO3 mapping ──────────────────────────────────────────────────────

print("Adding ISO3 country codes ...")
if "recipient_country" in df.columns:
    df["country_iso3"]      = df["recipient_country"].apply(ko_to_iso3)
    df["recipient_country_en"] = df["recipient_country"].apply(ko_to_english)

n_mapped   = df["country_iso3"].notna().sum()
n_unmapped = df["country_iso3"].isna().sum()
print(f"  Mapped: {n_mapped:,}  |  Unmapped (regional/unspecified): {n_unmapped:,}")

if n_unmapped > 0:
    top_unmapped = (
        df[df["country_iso3"].isna()]["recipient_country"]
        .value_counts().head(10)
    )
    print("  Top unmapped recipient_country values:")
    for name, cnt in top_unmapped.items():
        print(f"    {name}: {cnt:,}")

# ── SDG assignment ────────────────────────────────────────────────────────────

print("Assigning SDG labels ...")
df["sdg"] = df.apply(infer_sdg, axis=1)
df["sdg_source"] = "none"
df.loc[df["sdg_tag_raw"].apply(parse_sdg_tag).notna(), "sdg_source"] = "direct_tag"
df.loc[
    (df["sdg_source"] == "none") & df["sector_code"].apply(crs_to_sdg).notna(),
    "sdg_source"
] = "sector_code"
df.loc[df["sdg"].notna() & (df["sdg_source"] == "none"), "sdg_source"] = "marker"

n_tagged   = (df["sdg_source"] == "direct_tag").sum()
n_sector   = (df["sdg_source"] == "sector_code").sum()
n_marker   = (df["sdg_source"] == "marker").sum()
n_untagged = df["sdg"].isna().sum()
print(f"  Direct SDG tag:   {n_tagged:,}  ({n_tagged/len(df)*100:.1f}%)")
print(f"  Sector code map:  {n_sector:,}  ({n_sector/len(df)*100:.1f}%)")
print(f"  Marker:           {n_marker:,}  ({n_marker/len(df)*100:.1f}%)")
print(f"  Unassigned:       {n_untagged:,}  ({n_untagged/len(df)*100:.1f}%)")

# ── Save oda_clean.csv ────────────────────────────────────────────────────────

keep_cols = [c for c in [
    "project_id", "year", "agency",
    "recipient_country", "recipient_country_en", "country_iso3", "region",
    "project_type", "bilateral_multilateral", "fund_type",
    "aid_form", "aid_type", "aid_type_code",
    "sector", "sector_code",
    "project_name_ko", "project_name_en",
    "sdg_tag_raw", "sdg", "sdg_source",
    "commitment_musd", "disbursement_musd", "net_disbursement_musd", "grant_equivalent_musd",
    "marker_gender", "marker_environment", "marker_climate_mitigation",
    "marker_climate_adaptation", "marker_biodiversity", "marker_governance",
    "start_date", "end_date",
] if c in df.columns]

df_clean = df[keep_cols].copy()
df_clean.to_csv(OUT_CLEAN, index=False, encoding="utf-8-sig")
print(f"\nWrote oda_clean.csv: {len(df_clean):,} rows, {len(keep_cols)} columns")

# ── oda_sdg_annual.csv — year × SDG ──────────────────────────────────────────

print("\nBuilding oda_sdg_annual.csv ...")
df_sdg = df_clean[df_clean["sdg"].notna()].copy()
df_sdg["sdg"]  = df_sdg["sdg"].astype(int)
df_sdg["year"] = df_sdg["year"].astype(int)

sdg_annual = (
    df_sdg.groupby(["year", "sdg"])
    .agg(
        disbursement_musd     = ("disbursement_musd",     "sum"),
        commitment_musd       = ("commitment_musd",       "sum"),
        net_disbursement_musd = ("net_disbursement_musd", "sum"),
        n_projects            = ("project_id",            "count"),
    )
    .reset_index()
)
sdg_annual.to_csv(OUT_SDG_ANNUAL, index=False, encoding="utf-8-sig")
print(f"Wrote oda_sdg_annual.csv: {len(sdg_annual):,} year×SDG rows  "
      f"({sdg_annual['year'].nunique()} years × up to 17 SDGs)")

# ── oda_country_sdg_annual.csv — year × country_iso3 × SDG  (Panel C input) ──

print("\nBuilding oda_country_sdg_annual.csv ...")
df_c = df_sdg[df_sdg["country_iso3"].notna()].copy()

country_sdg = (
    df_c.groupby(["year", "country_iso3", "sdg"])
    .agg(
        disbursement_musd     = ("disbursement_musd",     "sum"),
        commitment_musd       = ("commitment_musd",       "sum"),
        net_disbursement_musd = ("net_disbursement_musd", "sum"),
        n_projects            = ("project_id",            "count"),
        # Convenience: how many distinct fund types (grant vs loan mix indicator)
        n_grant_projects      = ("fund_type",
                                 lambda x: (x.str.contains("무상", na=False)).sum()),
        n_loan_projects       = ("fund_type",
                                 lambda x: (x.str.contains("유상", na=False)).sum()),
    )
    .reset_index()
)

# Share of total annual disbursement (useful for relative measure)
year_totals = sdg_annual.groupby("year")["disbursement_musd"].sum().rename("year_total_musd")
country_sdg = country_sdg.join(year_totals, on="year")
country_sdg["oda_share_pct"] = (
    country_sdg["disbursement_musd"] / country_sdg["year_total_musd"] * 100
).round(4)
country_sdg.drop(columns=["year_total_musd"], inplace=True)

country_sdg.to_csv(OUT_COUNTRY_SDG, index=False, encoding="utf-8-sig")
print(f"Wrote oda_country_sdg_annual.csv: {len(country_sdg):,} rows  "
      f"({country_sdg['country_iso3'].nunique()} countries, "
      f"{country_sdg['year'].nunique()} years)")

# ── Summary ───────────────────────────────────────────────────────────────────

total_disb = sdg_annual["disbursement_musd"].sum()
print(f"\n{'='*60}")
print(f"Total disbursement (SDG-tagged): ${total_disb:,.1f}M USD")
print(f"Year range: {sdg_annual['year'].min()}–{sdg_annual['year'].max()}")
print(f"\nOutputs written to: {OUT_DIR.relative_to(BASE_DIR)}/")
print(f"  oda_clean.csv               ({len(df_clean):,} rows)")
print(f"  oda_sdg_annual.csv          ({len(sdg_annual):,} rows)")
print(f"  oda_country_sdg_annual.csv  ({len(country_sdg):,} rows)")
print("Done.")
