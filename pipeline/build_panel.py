"""
Final panel assembly.

Joins all data sources into three output tables:

  Option A: SDG × Month
    src/panel/panel_sdg_month.csv

  Option B: Country × Month
    src/panel/panel_country_month.csv

  Option C (primary): Country × SDG × Month
    src/panel/panel_country_sdg_month.csv

Variables:
  From media (aggregate_media.py output):
    freq_articles, freq_share_total, freq_share_sdg
    hhi_concentration, entropy_diversity
    rolling_3m/6m/12m, persistence_score
    cumulative_ytd, cumulative_all_time
    avg_sentiment, var_sentiment, sd_sentiment
    share_positive, share_neutral, share_negative
    polarization_index

  From ODA (preprocess_oda.py output):
    oda_disbursement_musd, oda_commitment_musd
    oda_net_disbursement_musd, oda_n_projects
    oda_share_pct  (% of annual total)
    — ODA is annual; distributed evenly across 12 months

  From political context (reference/political_context.py):
    president, ideology
    transition_month, first_year, acting_president
    na_election_year, presidential_election_year
    oda_policy_event, intl_event

  From crisis events (reference/crisis_events.py):
    crisis_health_{1,2,3}, crisis_conflict_{1,2,3}
    crisis_food_{1,2,3}, crisis_disaster_{1,2,3}
    crisis_sdg_{1..17}
    crisis_any, crisis_severity_max, crisis_count

Usage:
  python build_panel.py --years 2010-2016
  python build_panel.py                     # all available years
  python build_panel.py --option C          # Country×SDG×Month only (default)
  python build_panel.py --option A          # SDG×Month only
  python build_panel.py --option all        # all three outputs
"""

from __future__ import annotations

import json
import logging
import sys
import io
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
MEDIA_DIR  = BASE_DIR / "src" / "media"
ODA_DIR    = BASE_DIR / "src" / "oda" / "processed"
ODA_ANNUAL = BASE_DIR / "src" / "oda" / "oda_sdg_annual.csv"
PANEL_DIR  = BASE_DIR / "src" / "panel"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from reference.political_context import build_political_panel
from reference.crisis_events      import build_crisis_panel
from reference.countries_ko       import COUNTRY_MAP


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_media(years: list[int] | None = None) -> pd.DataFrame:
    media_path = MEDIA_DIR / "media_sdg_country_month.csv"
    if not media_path.exists():
        logger.warning("Media file not found: %s — run aggregate_media.py first", media_path)
        return pd.DataFrame()
    df = pd.read_csv(media_path, dtype={"country_iso3": str}, encoding="utf-8-sig")
    if years:
        df = df[df["year"].isin(years)]
    return df


def load_oda_monthly(years: list[int] | None = None) -> pd.DataFrame:
    """
    Load annual ODA data from oda_sdg_annual.csv and expand to monthly
    by distributing each annual value evenly across 12 months.

    Also loads country-level ODA from per-year JSON files.
    """
    if not ODA_ANNUAL.exists():
        logger.warning("ODA annual file not found — run preprocess_oda.py first")
        return pd.DataFrame()

    oda = pd.read_csv(ODA_ANNUAL, encoding="utf-8-sig")
    if years:
        oda = oda[oda["year"].isin(years)]

    # Annual total for share calculation
    annual_totals = oda.groupby("year")["disbursement_musd"].sum().rename("year_total_musd")
    oda = oda.join(annual_totals, on="year")
    oda["oda_share_pct"] = (oda["disbursement_musd"] / oda["year_total_musd"] * 100).round(4)

    # Expand to monthly (12 months per year)
    monthly_rows = []
    for _, row in oda.iterrows():
        for month in range(1, 13):
            monthly_rows.append({
                "year":                    int(row["year"]),
                "month":                   month,
                "sdg":                     int(row["sdg"]),
                "oda_disbursement_musd":   round(float(row["disbursement_musd"]) / 12, 4),
                "oda_commitment_musd":     round(float(row.get("commitment_musd", 0)) / 12, 4),
                "oda_n_projects":          round(float(row.get("n_projects", 0)) / 12, 2),
                "oda_annual_disbursement": float(row["disbursement_musd"]),
                "oda_share_pct":           float(row["oda_share_pct"]),
            })

    return pd.DataFrame(monthly_rows)


def load_oda_country_monthly(years: list[int] | None = None) -> pd.DataFrame:
    """
    Load country×SDG×year ODA from per-year JSON files, expand to monthly.
    """
    rows = []
    yr_dirs = sorted(ODA_DIR.glob("oda_*.json"))
    for path in yr_dirs:
        yr = int(path.stem.split("_")[1])
        if years and yr not in years:
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Each record has sdg + top_recipients; we need country-level data from clean CSV
        # Country-level ODA is in oda_clean.csv if available
    # Note: country-level ODA requires oda_clean.csv (not committed, lives locally)
    # Return empty if not available; the SDG-level ODA is the fallback
    clean_path = BASE_DIR / "src" / "oda" / "oda_clean.csv"
    if not clean_path.exists():
        logger.info("oda_clean.csv not found; country-level ODA unavailable")
        return pd.DataFrame()

    logger.info("Loading country-level ODA from oda_clean.csv ...")
    oda = pd.read_csv(clean_path, encoding="utf-8-sig",
                      usecols=["year", "recipient_country", "sdg",
                               "disbursement_musd", "commitment_musd"])
    if years:
        oda = oda[oda["year"].isin(years)]

    # Map Korean recipient country names to ISO3
    from reference.countries_ko import ko_to_iso3
    oda["country_iso3"] = oda["recipient_country"].apply(ko_to_iso3)
    oda = oda[oda["country_iso3"].notna() & (oda["sdg"].notna())]
    oda["sdg"] = oda["sdg"].astype(int)

    agg = (
        oda.groupby(["year", "country_iso3", "sdg"])
        .agg(
            oda_disbursement_musd = ("disbursement_musd", "sum"),
            oda_commitment_musd   = ("commitment_musd",   "sum"),
            oda_n_projects        = ("disbursement_musd", "count"),
        )
        .reset_index()
    )

    # Expand to monthly
    monthly_rows = []
    for _, row in agg.iterrows():
        for month in range(1, 13):
            monthly_rows.append({
                "year":                   int(row["year"]),
                "month":                  month,
                "country_iso3":           row["country_iso3"],
                "sdg":                    int(row["sdg"]),
                "oda_disbursement_musd":  round(float(row["oda_disbursement_musd"]) / 12, 5),
                "oda_commitment_musd":    round(float(row["oda_commitment_musd"]) / 12, 5),
                "oda_n_projects":         round(float(row["oda_n_projects"]) / 12, 3),
                "oda_annual_disbursement":float(row["oda_disbursement_musd"]),
            })
    return pd.DataFrame(monthly_rows)


# ── Panel builders ────────────────────────────────────────────────────────────

def build_option_a(media: pd.DataFrame, oda_monthly: pd.DataFrame,
                   political: pd.DataFrame, crisis: pd.DataFrame,
                   years: list[int]) -> pd.DataFrame:
    """SDG × Month panel."""
    # Aggregate media over countries
    media_sdg = (
        media.groupby(["year", "month", "sdg"])
        .agg(
            freq_articles     = ("freq_articles",      "sum"),
            avg_sentiment     = ("avg_sentiment",       "mean"),
            var_sentiment     = ("var_sentiment",       "mean"),
            hhi_concentration = ("hhi_concentration",   "mean"),
            entropy_diversity = ("entropy_diversity",   "mean"),
            rolling_3m        = ("rolling_3m",          "sum"),
            rolling_6m        = ("rolling_6m",          "sum"),
            rolling_12m       = ("rolling_12m",         "sum"),
            persistence_score = ("persistence_score",   "mean"),
            share_positive    = ("share_positive",      "mean"),
            share_neutral     = ("share_neutral",       "mean"),
            share_negative    = ("share_negative",      "mean"),
            polarization_index= ("polarization_index",  "mean"),
        )
        .reset_index()
    )

    panel = media_sdg.merge(oda_monthly, on=["year", "month", "sdg"], how="outer")
    panel = panel.merge(political, on=["year", "month"], how="left")
    panel = panel.merge(crisis,    on=["year", "month"], how="left")

    if years:
        panel = panel[panel["year"].isin(years)]
    return panel.sort_values(["year", "month", "sdg"])


def build_option_b(media: pd.DataFrame, oda_country: pd.DataFrame,
                   political: pd.DataFrame, crisis: pd.DataFrame,
                   years: list[int]) -> pd.DataFrame:
    """Country × Month panel."""
    media_country = (
        media.groupby(["year", "month", "country_iso3"])
        .agg(
            freq_articles     = ("freq_articles",       "sum"),
            avg_sentiment     = ("avg_sentiment",        "mean"),
            persistence_score = ("persistence_score",   "mean"),
        )
        .reset_index()
    )

    if not oda_country.empty:
        oda_c = (
            oda_country.groupby(["year", "month", "country_iso3"])
            .agg(
                oda_disbursement_musd = ("oda_disbursement_musd", "sum"),
                oda_commitment_musd   = ("oda_commitment_musd",   "sum"),
            )
            .reset_index()
        )
        panel = media_country.merge(oda_c, on=["year", "month", "country_iso3"], how="outer")
    else:
        panel = media_country

    panel = panel.merge(political, on=["year", "month"], how="left")
    panel = panel.merge(crisis,    on=["year", "month"], how="left")

    if years:
        panel = panel[panel["year"].isin(years)]
    return panel.sort_values(["year", "month", "country_iso3"])


def build_option_c(media: pd.DataFrame, oda_country: pd.DataFrame,
                   oda_sdg: pd.DataFrame, political: pd.DataFrame,
                   crisis: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Country × SDG × Month panel (primary)."""
    if oda_country.empty:
        # Fall back to SDG-level ODA without country dimension
        panel = media.merge(oda_sdg, on=["year", "month", "sdg"], how="outer")
    else:
        panel = media.merge(oda_country, on=["year", "month", "country_iso3", "sdg"], how="outer")

    # Add SDG-level crisis indicator (crisis_sdg_N)
    crisis_sdg_cols = [c for c in crisis.columns if c.startswith("crisis_sdg_")]
    panel = panel.merge(political, on=["year", "month"], how="left")
    panel = panel.merge(crisis,    on=["year", "month"], how="left")

    # Per-row SDG-specific crisis flag
    panel["crisis_affects_sdg"] = False
    for col in crisis_sdg_cols:
        sdg_num = int(col.split("_")[-1])
        mask = (panel["sdg"] == sdg_num) & (panel[col] == 1)
        panel.loc[mask, "crisis_affects_sdg"] = True

    if years:
        panel = panel[panel["year"].isin(years)]
    return panel.sort_values(["year", "month", "sdg", "country_iso3"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_years(years_arg: tuple[str]) -> list[int] | None:
    if not years_arg:
        return None
    result = []
    for arg in years_arg:
        if re.match(r"\d{4}-\d{4}", arg):
            start, end = arg.split("-")
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(arg))
    return sorted(set(result))

import re


@click.command()
@click.option("--years", "-y", multiple=True, default=(), help="e.g. 2010 or 2010-2016")
@click.option("--option", default="C", type=click.Choice(["A", "B", "C", "all"]),
              help="Panel option: A=SDG×Month, B=Country×Month, C=Country×SDG×Month")
def main(years: tuple, option: str) -> None:
    target_years = _parse_years(years)

    logger.info("Loading media aggregation ...")
    media = load_media(target_years)
    if media.empty:
        click.echo("ERROR: No media data found. Run aggregate_media.py first.")
        return

    logger.info("Loading ODA data ...")
    oda_sdg     = load_oda_monthly(target_years)
    oda_country = load_oda_country_monthly(target_years)

    logger.info("Building political context panel ...")
    start_yr = min(target_years) if target_years else 2007
    end_yr   = max(target_years) if target_years else 2025
    political = build_political_panel(start_yr, end_yr)

    logger.info("Building crisis events panel ...")
    crisis = build_crisis_panel(start_yr, end_yr)

    yrs = target_years or []

    if option in ("A", "all"):
        logger.info("Building Option A: SDG × Month ...")
        df_a = build_option_a(media, oda_sdg, political, crisis, yrs)
        out  = PANEL_DIR / "panel_sdg_month.csv"
        df_a.to_csv(out, index=False, encoding="utf-8-sig")
        click.echo(f"Option A: {len(df_a):,} rows → {out.relative_to(BASE_DIR)}")

    if option in ("B", "all"):
        logger.info("Building Option B: Country × Month ...")
        df_b = build_option_b(media, oda_country, political, crisis, yrs)
        out  = PANEL_DIR / "panel_country_month.csv"
        df_b.to_csv(out, index=False, encoding="utf-8-sig")
        click.echo(f"Option B: {len(df_b):,} rows → {out.relative_to(BASE_DIR)}")

    if option in ("C", "all"):
        logger.info("Building Option C: Country × SDG × Month ...")
        df_c = build_option_c(media, oda_country, oda_sdg, political, crisis, yrs)
        out  = PANEL_DIR / "panel_country_sdg_month.csv"
        df_c.to_csv(out, index=False, encoding="utf-8-sig")
        click.echo(f"Option C: {len(df_c):,} rows → {out.relative_to(BASE_DIR)}")

    click.echo("\nDone. Panel files written to src/panel/")


if __name__ == "__main__":
    main()
