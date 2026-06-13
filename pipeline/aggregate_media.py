"""
Media variable aggregation pipeline.

Reads BigKinds news CSVs (raw or ML-classified) and computes, per
(year, month, sdg, country_iso3):

  Frequency
    freq_articles         — raw article count
    freq_share_sdg        — share within SDG-month total
    freq_share_total      — share of all articles that month

  Concentration (outlet diversity)
    hhi_concentration     — Herfindahl-Hirschman Index (0-1)
    entropy_diversity     — Shannon entropy (bits)

  Persistence / Rolling coverage
    rolling_3m            — sum over trailing 3 months
    rolling_6m            — sum over trailing 6 months
    rolling_12m           — sum over trailing 12 months
    cumulative_ytd        — cumulative count since Jan of that year
    cumulative_all_time   — cumulative count since dataset start
    persistence_score     — current month / trailing-12m monthly average

  Sentiment (requires classified CSVs)
    avg_sentiment         — mean sentiment score (0-1, where 1=positive)
    var_sentiment         — variance of sentiment scores
    sd_sentiment          — standard deviation
    share_positive        — fraction of positive articles
    share_neutral         — fraction of neutral articles
    share_negative        — fraction of negative articles
    polarization_index    — share_positive × share_negative (high = contested)

Mode:
  If *_classified.csv files are present → use sdg_label, sentiment columns.
  Otherwise → fall back to keyword-based SDG detection (preliminary).
  Country detection always uses keyword matching (reference/countries_ko.py).

Usage:
  python aggregate_media.py --years 2010 2011 2012 2013 2014 2015 2016
  python aggregate_media.py --years 2010-2016           # range shorthand
  python aggregate_media.py                             # all available years
"""

from __future__ import annotations

import logging
import re
import sys
import io
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Optional

import click
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent.parent
NEWS_DIR      = BASE_DIR / "src" / "news"
PROCESSED_DIR = BASE_DIR / "src" / "processed" / "news"
OUT_DIR       = BASE_DIR / "src" / "processed" / "media"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from classify.keywords_ko import SDG_KEYWORDS_KO, keyword_scores
from reference.countries_ko import detect_countries, COUNTRY_MAP

SDG_NAMES = {
    1:"No Poverty", 2:"Zero Hunger", 3:"Good Health", 4:"Quality Education",
    5:"Gender Equality", 6:"Clean Water", 7:"Clean Energy", 8:"Decent Work",
    9:"Industry & Innovation", 10:"Reduced Inequality", 11:"Sustainable Cities",
    12:"Responsible Consumption", 13:"Climate Action", 14:"Life Below Water",
    15:"Life on Land", 16:"Peace & Justice", 17:"Partnerships",
}

SENTIMENT_SCORE = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}


# ── Article-level extraction ──────────────────────────────────────────────────

def _norm_date(raw_date) -> Optional[tuple[int, int]]:
    """Parse pub_date to (year, month). Handles YYYYMMDD int or str."""
    s = str(raw_date).strip().replace("-", "").replace("/", "")[:8]
    if len(s) == 8 and s.isdigit():
        return int(s[:4]), int(s[4:6])
    return None


def _sentiment_numeric(label: str) -> float:
    return SENTIMENT_SCORE.get(str(label).lower().strip(), 0.5)


def _hhi(shares: np.ndarray) -> float:
    """Herfindahl-Hirschman Index from share array (values 0-1, sum=1)."""
    return float(np.sum(shares ** 2))


def _entropy(shares: np.ndarray) -> float:
    """Shannon entropy in bits from share array."""
    shares = shares[shares > 0]
    return float(-np.sum(shares * np.log2(shares))) if len(shares) > 0 else 0.0


def extract_articles(csv_path: Path) -> pd.DataFrame:
    """
    Read one BigKinds CSV, return article-level DataFrame with columns:
      year, month, sdg (list), country_iso3 (list), provider,
      sentiment_label, sentiment_score, is_classified
    """
    classified_path = PROCESSED_DIR / csv_path.parent.name / (csv_path.stem + "_classified.csv")
    use_classified = classified_path.exists()

    try:
        df = pd.read_csv(
            classified_path if use_classified else csv_path,
            dtype=str,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except Exception as e:
        logger.warning("Failed to read %s: %s", csv_path.name, e)
        return pd.DataFrame()

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "제목": "title", "키워드": "keywords", "본문": "body",
        "언론사": "provider", "일자": "pub_date",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Parse date
    date_col = "pub_date" if "pub_date" in df.columns else None
    if date_col is None:
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        ym = _norm_date(row.get(date_col, ""))
        if ym is None:
            continue
        year, month = ym

        title    = str(row.get("title",    "") or "")
        keywords = str(row.get("keywords", "") or "")
        body     = str(row.get("body",     "") or "")[:500]  # limit body scan
        text_for_country = f"{title} {keywords} {body}"
        text_for_sdg     = f"{title} {keywords}"

        # SDG assignment
        if use_classified and "sdg_label" in row:
            sdg_label = int(float(row["sdg_label"])) if str(row["sdg_label"]).replace(".","").isdigit() else 0
            sdg_list  = [sdg_label] if sdg_label > 0 else []
        else:
            kw_hits = keyword_scores(text_for_sdg)
            sdg_list = list(kw_hits.keys()) if kw_hits else []

        # Sentiment
        if use_classified and "sdg_favorability" in row:
            sent_label = str(row.get("sdg_favorability", "neutral")).lower().strip()
            sent_score = float(row.get("sentiment_score", 0.5) or 0.5)
        else:
            sent_label = "neutral"
            sent_score = 0.5

        # Country mentions
        countries = detect_countries(text_for_country)

        # Provider / outlet
        provider = str(row.get("provider", "") or "").strip()

        records.append({
            "year":           year,
            "month":          month,
            "sdg_list":       sdg_list,
            "country_list":   countries,
            "provider":       provider,
            "sentiment_label":sent_label,
            "sentiment_score":sent_score,
            "is_classified":  use_classified,
        })

    return pd.DataFrame(records)


# ── Monthly aggregation ───────────────────────────────────────────────────────

def aggregate_to_sdg_country_month(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode by (sdg, country) and aggregate to (year, month, sdg, country_iso3).
    """
    if articles_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in articles_df.iterrows():
        sdgs      = row["sdg_list"]      if row["sdg_list"]      else [0]
        countries = row["country_list"]  if row["country_list"]  else ["---"]
        for sdg in sdgs:
            for country in countries:
                rows.append({
                    "year":            row["year"],
                    "month":           row["month"],
                    "sdg":             sdg,
                    "country_iso3":    country,
                    "provider":        row["provider"],
                    "sentiment_score": row["sentiment_score"],
                    "sentiment_label": row["sentiment_label"],
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    groups = df.groupby(["year", "month", "sdg", "country_iso3"])

    agg = groups.agg(
        freq_articles     = ("sentiment_score", "count"),
        avg_sentiment     = ("sentiment_score", "mean"),
        var_sentiment     = ("sentiment_score", "var"),
        sd_sentiment      = ("sentiment_score", "std"),
    ).reset_index()

    # Sentiment shares
    def sentiment_shares(g):
        n = len(g)
        return pd.Series({
            "share_positive": (g["sentiment_label"] == "positive").sum() / n,
            "share_neutral":  (g["sentiment_label"] == "neutral").sum()  / n,
            "share_negative": (g["sentiment_label"] == "negative").sum() / n,
        })

    shares = df.groupby(["year", "month", "sdg", "country_iso3"]).apply(sentiment_shares).reset_index()
    agg = agg.merge(shares, on=["year", "month", "sdg", "country_iso3"], how="left")
    agg["polarization_index"] = agg["share_positive"] * agg["share_negative"]

    # Media concentration (HHI + entropy) per sdg-month (provider share)
    def concentration(g):
        counts = g["provider"].value_counts()
        total  = counts.sum()
        shares_arr = (counts / total).values if total > 0 else np.array([1.0])
        return pd.Series({
            "hhi_concentration": _hhi(shares_arr),
            "entropy_diversity": _entropy(shares_arr),
        })

    conc = df.groupby(["year", "month", "sdg", "country_iso3"]).apply(concentration).reset_index()
    agg  = agg.merge(conc, on=["year", "month", "sdg", "country_iso3"], how="left")

    agg = agg.fillna({"var_sentiment": 0, "sd_sentiment": 0, "polarization_index": 0})
    return agg


def add_rolling_and_persistence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling coverage and persistence columns.
    Input must have columns: year, month, sdg, country_iso3, freq_articles.
    """
    if df.empty:
        return df

    df = df.copy()
    df["ym_int"] = df["year"] * 100 + df["month"]

    all_ym = sorted(df["ym_int"].unique())
    ym_to_idx = {ym: i for i, ym in enumerate(all_ym)}

    def prior_n_months(ym: int, n: int) -> list[int]:
        idx = ym_to_idx.get(ym, 0)
        start = max(0, idx - n)
        return all_ym[start:idx]

    pivot = df.pivot_table(
        index="ym_int",
        columns=["sdg", "country_iso3"],
        values="freq_articles",
        aggfunc="sum",
        fill_value=0,
    )

    rolling_3  = {}
    rolling_6  = {}
    rolling_12 = {}
    persist    = {}
    cum_ytd    = {}
    cum_all    = defaultdict(int)

    for ym in all_ym:
        yr = ym // 100
        p3  = prior_n_months(ym, 3)
        p6  = prior_n_months(ym, 6)
        p12 = prior_n_months(ym, 12)

        for col in pivot.columns:
            val    = pivot.loc[ym, col]  if ym  in pivot.index else 0
            r3     = pivot.loc[pivot.index.isin(p3),  col].sum()
            r6     = pivot.loc[pivot.index.isin(p6),  col].sum()
            r12    = pivot.loc[pivot.index.isin(p12), col].sum()
            avg12  = r12 / len(p12) if p12 else 0
            ps     = val / avg12 if avg12 > 0 else 1.0

            cum_all[col] += val
            ym_ytd = [x for x in all_ym if x // 100 == yr and x <= ym]
            cyt    = pivot.loc[pivot.index.isin(ym_ytd), col].sum()

            rolling_3[(ym,  *col)] = r3
            rolling_6[(ym,  *col)] = r6
            rolling_12[(ym, *col)] = r12
            persist[(ym,    *col)] = ps
            cum_ytd[(ym,    *col)] = cyt

    df["rolling_3m"]        = df.apply(lambda r: rolling_3.get( (r["ym_int"], r["sdg"], r["country_iso3"]), 0), axis=1)
    df["rolling_6m"]        = df.apply(lambda r: rolling_6.get( (r["ym_int"], r["sdg"], r["country_iso3"]), 0), axis=1)
    df["rolling_12m"]       = df.apply(lambda r: rolling_12.get((r["ym_int"], r["sdg"], r["country_iso3"]), 0), axis=1)
    df["persistence_score"] = df.apply(lambda r: persist.get(   (r["ym_int"], r["sdg"], r["country_iso3"]), 1.0), axis=1)
    df["cumulative_ytd"]    = df.apply(lambda r: cum_ytd.get(   (r["ym_int"], r["sdg"], r["country_iso3"]), 0), axis=1)
    df["cumulative_all_time"]= df.apply(lambda r: cum_all.get(  (r["sdg"],    r["country_iso3"]),            0), axis=1)

    # Total articles per month (for share calculation)
    monthly_total = df.groupby(["year", "month"])["freq_articles"].transform("sum")
    sdg_month_total = df.groupby(["year", "month", "sdg"])["freq_articles"].transform("sum")
    df["freq_share_total"] = (df["freq_articles"] / monthly_total.replace(0, np.nan)).round(5)
    df["freq_share_sdg"]   = (df["freq_articles"] / sdg_month_total.replace(0, np.nan)).round(5)

    return df.drop(columns=["ym_int"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_years(years_arg: tuple[str]) -> list[int]:
    result = []
    for arg in years_arg:
        if "-" in arg and len(arg) == 9:  # range like "2010-2016"
            start, end = arg.split("-")
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(arg))
    return sorted(set(result))


@click.command()
@click.option("--years", "-y", multiple=True, default=(),
              help="Years to process: '2010' or '2010-2016'. Default: all.")
@click.option("--out", default="", help="Output CSV path (default: src/media/media_sdg_country_month.csv)")
def main(years: tuple, out: str) -> None:
    target_years = _parse_years(years) if years else None

    all_dirs = sorted(d for d in NEWS_DIR.iterdir() if d.is_dir() and d.name.isdigit())
    if target_years:
        all_dirs = [d for d in all_dirs if int(d.name) in target_years]

    if not all_dirs:
        click.echo("No year directories found.")
        return

    all_articles: list[pd.DataFrame] = []

    for yr_dir in all_dirs:
        yr = int(yr_dir.name)
        files = sorted(f for f in yr_dir.glob("*.csv") if "_classified" not in f.name)
        logger.info("Year %d: %d files", yr, len(files))

        yr_articles: list[pd.DataFrame] = []
        for f in files:
            chunk = extract_articles(f)
            if not chunk.empty:
                yr_articles.append(chunk)

        if not yr_articles:
            continue
        yr_df = pd.concat(yr_articles, ignore_index=True)
        logger.info("  → %d articles", len(yr_df))
        all_articles.append(yr_df)

    if not all_articles:
        click.echo("No articles loaded.")
        return

    articles = pd.concat(all_articles, ignore_index=True)
    is_classified = articles["is_classified"].any()
    click.echo(f"\nTotal articles: {len(articles):,}  "
               f"(classified={is_classified})")

    click.echo("Aggregating to SDG × country × month ...")
    agg = aggregate_to_sdg_country_month(articles)

    click.echo("Adding rolling/persistence variables ...")
    agg = add_rolling_and_persistence(agg)

    out_path = Path(out) if out else OUT_DIR / "media_sdg_country_month.csv"
    agg.to_csv(out_path, index=False, encoding="utf-8-sig")
    click.echo(f"\nWrote {len(agg):,} rows → {out_path.relative_to(BASE_DIR)}")

    # Also write SDG×month aggregate (marginalise over country)
    sdg_month = (
        agg.groupby(["year", "month", "sdg"])
        .agg(
            freq_articles     = ("freq_articles",    "sum"),
            avg_sentiment     = ("avg_sentiment",    "mean"),
            hhi_concentration = ("hhi_concentration","mean"),
            entropy_diversity = ("entropy_diversity","mean"),
            rolling_3m        = ("rolling_3m",       "sum"),
            rolling_6m        = ("rolling_6m",       "sum"),
            rolling_12m       = ("rolling_12m",      "sum"),
            persistence_score = ("persistence_score","mean"),
        )
        .reset_index()
    )
    sdg_month_path = OUT_DIR / "media_sdg_month.csv"
    sdg_month.to_csv(sdg_month_path, index=False, encoding="utf-8-sig")
    click.echo(f"Wrote SDG×month aggregate: {sdg_month_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
