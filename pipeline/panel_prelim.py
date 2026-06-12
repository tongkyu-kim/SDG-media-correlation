"""
Preliminary 5-year panel analysis: Korean media SDG coverage vs ODA allocation.

Method:
  - Media SDG coverage: keyword-based detection on article title + keywords
    using SDG_KEYWORDS_KO (same lexicon used by the ML classifier).
    This is a fast proxy; the full ML-classified version will replace it later.
  - ODA allocation: from preprocessed per-year JSON files (OECD CRS crosswalk).

Panel: 2010-2016 (7 years × 17 SDGs, balanced on SDG dimension).
  Note: 2016 is partial (11/~52 weekly files); flagged in output.

Output: printed summary tables + src/oda/panel_prelim.csv
"""

from __future__ import annotations

import json
import sys
import io
import warnings
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR  = Path(__file__).parent.parent
NEWS_DIR  = BASE_DIR / "src" / "news"
ODA_DIR   = BASE_DIR / "src" / "oda" / "processed"
OUT_CSV   = BASE_DIR / "src" / "oda" / "panel_prelim.csv"

YEARS = list(range(2010, 2017))   # 2016 partial but included
PARTIAL_YEARS = {2016}

SDG_NAMES = {
    1:"No Poverty", 2:"Zero Hunger", 3:"Good Health", 4:"Quality Education",
    5:"Gender Equality", 6:"Clean Water", 7:"Clean Energy", 8:"Decent Work",
    9:"Industry & Innovation", 10:"Reduced Inequality", 11:"Sustainable Cities",
    12:"Responsible Consumption", 13:"Climate Action", 14:"Life Below Water",
    15:"Life on Land", 16:"Peace & Justice", 17:"Partnerships",
}

# ── Load keyword lists ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from classify.keywords_ko import SDG_KEYWORDS_KO


def sdg_hits(text: str) -> list[int]:
    """Return list of SDGs with at least one keyword match in text."""
    if not text or pd.isna(text):
        return []
    t = str(text).lower()
    return [sdg for sdg, kws in SDG_KEYWORDS_KO.items() if any(kw in t for kw in kws)]


# ── Step 1: media SDG counts per year ────────────────────────────────────────

print("=" * 62)
print("STEP 1 — Media SDG coverage (keyword proxy)")
print("=" * 62)

media_records = []   # list of {year, sdg, n_articles, n_total}

for yr in YEARS:
    yr_dir = NEWS_DIR / str(yr)
    files = sorted(f for f in yr_dir.glob("*.csv") if "_classified" not in f.name)
    if not files:
        print(f"  {yr}: no files found, skipping.")
        continue

    sdg_counter: dict[int, int] = defaultdict(int)
    n_total = 0

    for f in files:
        try:
            df = pd.read_csv(f, dtype=str, encoding="utf-8-sig",
                             usecols=lambda c: c in
                             ["title", "keywords", "제목", "키워드"])
        except Exception:
            continue

        # Normalise column names
        df.columns = [c.strip() for c in df.columns]
        if "제목" in df.columns:
            df = df.rename(columns={"제목": "title", "키워드": "keywords"})

        for _, row in df.iterrows():
            text = f"{row.get('title', '')} {row.get('keywords', '')}"
            hits = sdg_hits(text)
            for sdg in set(hits):   # count article once per SDG
                sdg_counter[sdg] += 1
            n_total += 1

    note = " (partial)" if yr in PARTIAL_YEARS else ""
    print(f"  {yr}{note}: {n_total:,} articles, "
          f"{sum(sdg_counter.values()):,} SDG-keyword hits across "
          f"{len(sdg_counter)} SDGs")

    for sdg in range(1, 18):
        media_records.append({
            "year":      yr,
            "sdg":       sdg,
            "n_articles": sdg_counter.get(sdg, 0),
            "n_total":   n_total,
            "partial":   yr in PARTIAL_YEARS,
        })

df_media = pd.DataFrame(media_records)
df_media["media_share_pct"] = (
    df_media["n_articles"] / df_media["n_total"] * 100
).round(3)

# ── Step 2: ODA data per year ─────────────────────────────────────────────────

print("\n" + "=" * 62)
print("STEP 2 — ODA disbursement (OECD CRS crosswalk)")
print("=" * 62)

oda_records = []

for yr in YEARS:
    path = ODA_DIR / f"oda_{yr}.json"
    if not path.exists():
        print(f"  {yr}: ODA file missing.")
        continue
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    yr_total = sum(r["disbursement_musd"] for r in data)
    by_sdg = {r["sdg"]: r["disbursement_musd"] for r in data}
    print(f"  {yr}: ${yr_total:,.1f}M total, {len(data)} SDGs covered")

    for sdg in range(1, 18):
        oda_records.append({
            "year":             yr,
            "sdg":              sdg,
            "oda_disbursement": by_sdg.get(sdg, 0.0),
            "oda_total":        yr_total,
        })

df_oda = pd.DataFrame(oda_records)
df_oda["oda_share_pct"] = (
    df_oda["oda_disbursement"] / df_oda["oda_total"] * 100
).round(3)

# ── Step 3: Join panel ────────────────────────────────────────────────────────

df_panel = df_media.merge(df_oda, on=["year", "sdg"])
df_panel.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# ── Step 4: Correlations ──────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("STEP 3 — Panel correlations: media share vs ODA share")
print("=" * 62)

# A) Cross-sectional correlation per year (SDG as unit, N=17)
print("\n(A) Cross-sectional: within each year, corr(media_share, oda_share) across 17 SDGs")
print(f"  {'Year':<6}  {'Pearson r':>10}  {'p':>7}  {'Spearman ρ':>11}  {'p':>7}  {'N':>4}")
print("  " + "-" * 52)
for yr in YEARS:
    sub = df_panel[df_panel["year"] == yr]
    x = sub["media_share_pct"].values
    y = sub["oda_share_pct"].values
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    note = "*" if yr in PARTIAL_YEARS else " "
    print(f"  {yr}{note}     {pr:>+.3f}      {pp:>6.3f}    {sr:>+.3f}          {sp:>6.3f}   {len(sub):>3}")

# B) Time-series correlation per SDG (year as unit, N=7)
print("\n(B) Time-series: for each SDG, corr(media_share, oda_share) across 2010-2016")
print(f"  {'SDG':<4}  {'Name':<28}  {'Pearson r':>10}  {'p':>7}  {'Spearman ρ':>11}  {'p':>7}")
print("  " + "-" * 72)

ts_rows = []
for sdg in range(1, 18):
    sub = df_panel[df_panel["sdg"] == sdg].sort_values("year")
    x = sub["media_share_pct"].values
    y = sub["oda_share_pct"].values
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        continue
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    sig = "**" if pp < 0.05 else ("*" if pp < 0.10 else "  ")
    print(f"  {sdg:<4}  {SDG_NAMES[sdg]:<28}  {pr:>+.3f}{sig}    {pp:>6.3f}    {sr:>+.3f}          {sp:>6.3f}")
    ts_rows.append({"sdg": sdg, "name": SDG_NAMES[sdg], "pearson_r": pr,
                    "pearson_p": pp, "spearman_r": sr, "spearman_p": sp})

# C) Pooled panel correlation (all year×SDG pairs)
print("\n(C) Pooled panel: all year×SDG observations (N=119)")
x_all = df_panel["media_share_pct"].values
y_all = df_panel["oda_share_pct"].values
pr, pp = stats.pearsonr(x_all, y_all)
sr, sp = stats.spearmanr(x_all, y_all)
print(f"  Pearson  r = {pr:+.4f}  (p = {pp:.4f})")
print(f"  Spearman ρ = {sr:+.4f}  (p = {sp:.4f})")

# ── Step 5: Summary table ─────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("STEP 4 — Average media share vs ODA share by SDG (2010-2015)")
print("         (excluding partial 2016)")
print("=" * 62)

full_years = df_panel[~df_panel["partial"]]
avg = (
    full_years.groupby("sdg")[["media_share_pct", "oda_share_pct"]]
    .mean()
    .round(3)
    .reset_index()
)
avg["name"]  = avg["sdg"].map(SDG_NAMES)
avg["gap"]   = (avg["oda_share_pct"] - avg["media_share_pct"]).round(3)
avg["ratio"] = (avg["oda_share_pct"] / avg["media_share_pct"].replace(0, np.nan)).round(2)

avg_sorted = avg.sort_values("oda_share_pct", ascending=False)
print(f"\n  {'SDG':<4}  {'Name':<28}  {'Media%':>7}  {'ODA%':>7}  {'Gap(O-M)':>9}  {'ODA/Media':>10}")
print("  " + "-" * 72)
for _, row in avg_sorted.iterrows():
    ratio_str = f"{row['ratio']:.2f}x" if pd.notna(row['ratio']) else "  N/A"
    print(f"  {int(row['sdg']):<4}  {row['name']:<28}  "
          f"{row['media_share_pct']:>7.3f}  {row['oda_share_pct']:>7.3f}  "
          f"{row['gap']:>+9.3f}  {ratio_str:>10}")

print(f"\n  * 2016 is partial (~21% of annual files). Correlations include it;")
print(f"    averages exclude it to avoid systematic downward bias.")
print(f"\nPanel CSV saved to: {OUT_CSV.relative_to(BASE_DIR)}")
