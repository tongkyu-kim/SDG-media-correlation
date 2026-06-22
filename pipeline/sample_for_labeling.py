"""
Generate the annotation sheet for manual labeling.

Reads news CSVs, draws a year-stratified random sample, and writes
src/labels/sample_for_labeling.csv with one row per article.

Coders fill in the annotation columns (prefixed label_*).
A second coder does the same 200-row overlap subset so inter-rater
reliability (Cohen's kappa) can be computed.

Usage:
  python sample_for_labeling.py               # 1000 articles, all years
  python sample_for_labeling.py --n 1200      # larger sample
  python sample_for_labeling.py --overlap 200 # size of two-coder overlap block
  python sample_for_labeling.py --seed 42
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
import config

BASE_DIR   = Path(__file__).parent.parent
DOCS_DIR   = BASE_DIR / "docs"
LABELS_DIR = config.LABELS_DIR     # src/labels/ — pipeline reads labeled data from here
DOCS_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

_NEWS_DIR = config.NEWS_DATA_DIR   # flat news_YYYY_MM.csv files


_READ_COLS = ["article_id", "date", "outlet", "title", "keywords", "top_keywords", "body"]


def _read_file(f: Path, n_rows: int | None = None) -> pd.DataFrame | None:
    """Read one news CSV. Pass n_rows to limit rows read (faster for large files)."""
    try:
        header = pd.read_csv(f, nrows=0, encoding="utf-8-sig", on_bad_lines="skip")
        cols   = [c for c in _READ_COLS if c in header.columns]
        if not cols:
            return None
        df = pd.read_csv(
            f, dtype=str, encoding="utf-8-sig",
            on_bad_lines="skip",
            usecols=cols,
            nrows=n_rows,
        )
        df["_source"] = f.name
        date_col = "date" if "date" in df.columns else None
        df["_year"] = (
            df[date_col].astype(str).str[:4].fillna("unknown")
            if date_col else "unknown"
        )
        return df
    except Exception as e:
        logger.warning("Could not read %s: %s", f.name, e)
        return None


def stratified_sample_from_files(
    files: list[Path], n: int, seed: int, files_per_year: int = 3
) -> pd.DataFrame:
    """
    Fast stratified sample.

    Groups files by year (extracted from filename news_YYYY_MM.csv),
    picks `files_per_year` random files per year, reads only those files,
    and samples proportionally.  Reads at most files_per_year × n_years files
    (default: 3 × 11 = 33 files) instead of all 204.
    """
    import random
    rng = random.Random(seed)

    # Group files by year using the filename pattern news_YYYY_MM.csv
    year_files: dict[str, list[Path]] = {}
    for f in files:
        parts = f.stem.split("_")          # ["news", "2016", "03"]
        yr    = parts[1] if len(parts) >= 2 else "unknown"
        year_files.setdefault(yr, []).append(f)

    years = sorted(year_files.keys())
    logger.info("Found %d years: %s", len(years), years)

    # Per-year quota — equal allocation (adjust for actual content below)
    base_quota = max(1, n // len(years))

    parts: list[pd.DataFrame] = []
    for yr in years:
        yr_files = year_files[yr]
        # Pick a random subset of files for this year
        chosen = rng.sample(yr_files, min(files_per_year, len(yr_files)))
        logger.info("  %s: reading %d/%d files ...", yr, len(chosen), len(yr_files))

        yr_parts = [_read_file(f) for f in chosen]
        yr_df    = pd.concat([d for d in yr_parts if d is not None], ignore_index=True)
        if yr_df.empty:
            continue

        quota  = min(base_quota, len(yr_df))
        sample = yr_df.sample(n=quota, random_state=seed)
        parts.append(sample)

    if not parts:
        logger.error("No articles could be sampled.")
        sys.exit(1)

    result = pd.concat(parts, ignore_index=True)
    # Trim or top-up to exactly n
    if len(result) > n:
        result = result.sample(n=n, random_state=seed)
    logger.info("Sampled %d articles from %d years", len(result), len(years))
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


@click.command()
@click.option("--n",       default=1000, show_default=True, type=int)
@click.option("--overlap", default=200,  show_default=True, type=int,
              help="First N rows are labeled by BOTH coders for kappa")
@click.option("--seed",    default=2025, show_default=True, type=int)
@click.option("--year",    default="", help="Restrict to one year")
def main(n: int, overlap: int, seed: int, year: str) -> None:
    pattern = f"news_{year}_*.csv" if year else "news_*.csv"
    files = sorted(
        p for p in _NEWS_DIR.glob(pattern)
        if not any(s in p.stem for s in ["_classified", "_oda", "_devrel"])
    )
    if not files:
        logger.error("No files found in %s matching %s", _NEWS_DIR, pattern)
        sys.exit(1)
    sample  = stratified_sample_from_files(files, n, seed)

    # body_preview: first 250 chars for coder context
    if "body" in sample.columns:
        sample["body_preview"] = sample["body"].fillna("").str[:250]
    else:
        sample["body_preview"] = ""

    # ── Annotation columns (empty — coders fill these) ────────────────────────
    # development_relevant  0 / 1
    #   1 = article discusses a development-related situation in a developing
    #       country (poverty, health, education, conflict, climate, aid).
    #   0 = domestic Korean news, international news about developed countries,
    #       or clearly unrelated (sports, entertainment, stock market, etc.)
    #
    # sdg_labels  pipe-separated SDG numbers OR "0"
    #   Which SDGs does this article address? e.g. "3|13" means SDG3 + SDG13.
    #   Use "0" if development_relevant=1 but no clear SDG.
    #   Leave blank if development_relevant=0.
    #
    # sentiment_country  -2 / -1 / 0 / 1 / 2
    #   Tone toward the recipient country's conditions / situation:
    #   -2 = strongly negative (severe crisis, suffering, collapse)
    #   -1 = somewhat negative (problems, challenges, concerns)
    #    0 = neutral / balanced / factual
    #   +1 = somewhat positive (progress, improvement, hope)
    #   +2 = strongly positive (success, recovery, achievement)
    #   Leave blank if development_relevant=0.
    #
    # crisis_flag  0 / 1
    #   1 = article describes an active humanitarian / conflict / disaster /
    #       health emergency involving a developing country.
    #
    # crisis_type  comma-separated subset of:
    #   conflict, disaster, food, health, refugee, economic, governance
    #   Leave blank if crisis_flag=0.
    #
    # notes  free text — any ambiguities or edge cases

    for col in [
        "label_development_relevant",
        "label_sdg_labels",
        "label_sentiment_country",
        "label_crisis_flag",
        "label_crisis_type",
        "notes",
        "coder_id",
    ]:
        sample[col] = ""

    # Mark overlap rows (first `overlap` rows — both coders label these)
    sample["overlap_row"] = ""
    sample.loc[:overlap - 1, "overlap_row"] = "YES"

    out_cols = [
        "overlap_row",
        "article_id", "date", "outlet", "title", "keywords", "body_preview",
        "label_development_relevant",
        "label_sdg_labels",
        "label_sentiment_country",
        "label_crisis_flag",
        "label_crisis_type",
        "notes",
        "coder_id",
        "_source",
    ]
    out = sample[[c for c in out_cols if c in sample.columns]]

    # Write to docs/ (for easy access) and src/labels/ (for pipeline)
    docs_path   = DOCS_DIR   / "sample_for_labeling.csv"
    labels_path = LABELS_DIR / "sample_for_labeling.csv"
    out.to_csv(docs_path,   index=False, encoding="utf-8-sig")
    out.to_csv(labels_path, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────────────────────
    click.echo(f"\nAnnotation sheet: {len(out):,} articles")
    click.echo(f"  → {docs_path.relative_to(BASE_DIR)}   (upload this to Google Sheets)")
    click.echo(f"  → {labels_path.relative_to(BASE_DIR)}  (pipeline reads from here)")
    click.echo(f"Overlap block (both coders): rows 0–{overlap - 1}")
    click.echo("\nYear distribution:")
    for yr, cnt in sample["_year"].value_counts().sort_index().items():
        click.echo(f"  {yr}: {cnt:,}")

    _print_instructions(out_path, overlap)


def _print_instructions(path: Path, overlap: int) -> None:
    click.echo("""
╔══════════════════════════════════════════════════════════════════╗
║                    ANNOTATION INSTRUCTIONS                       ║
╚══════════════════════════════════════════════════════════════════╝

1. Upload sample_for_labeling.csv to Google Sheets.

2. BOTH coders label rows where overlap_row = YES first.
   Compare results and resolve disagreements before continuing.

3. Each coder fills their assigned rows in these columns:

   label_development_relevant  →  0 or 1
     1 = discusses a developing country's development situation
     0 = domestic Korean news, developed-country news, other

   label_sdg_labels  →  pipe-separated numbers, e.g. "3|13" or "0"
     Only fill when development_relevant = 1
     SDG 1=Poverty  2=Hunger  3=Health  4=Education  5=Gender
     6=Water  7=Energy  8=Work  9=Industry  10=Inequality
     11=Cities  12=Consumption  13=Climate  14=Oceans
     15=Land  16=Peace  17=Partnership

   label_sentiment_country  →  -2, -1, 0, 1, or 2
     Toward the recipient country's situation (not toward Korea/ODA)
     -2=strongly negative  -1=negative  0=neutral
     +1=positive  +2=strongly positive
     Only fill when development_relevant = 1

   label_crisis_flag  →  0 or 1
     1 = article reports an active crisis (conflict, disaster, etc.)

   label_crisis_type  →  comma-separated
     conflict, disaster, food, health, refugee, economic, governance
     Only fill when crisis_flag = 1

   notes  →  any ambiguity, reason for edge case decision

   coder_id  →  your initials (e.g. TK or CA)

4. Save as sample_labeled.csv (keep same columns and row_ids).

5. Run validation: python validate_ml.py --task irc
   (requires coder1.csv and coder2.csv for the overlap rows)

6. Run training: python train_devrel_classifier.py
""")


if __name__ == "__main__":
    main()
