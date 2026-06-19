"""
Step 3 support — generate random sample for manual ODA labeling.

Reads cleaned news CSVs (NEWS_CLEAN_DIR), draws a stratified random
sample by year, and writes it to src/labels/sample_for_labeling.csv.

Annotation convention: fill in the oda_relevant column
  1 = article is about ODA / international development
  0 = not ODA-related

Usage:
  python sample_for_labeling.py                   # 1000 samples, all years
  python sample_for_labeling.py --n 500           # smaller sample
  python sample_for_labeling.py --seed 42         # reproducible
  python sample_for_labeling.py --year 2016       # restrict to one year
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
LABELS_DIR = config.LABELS_DIR
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def _news_source() -> Path:
    if config.NEWS_CLEAN_DIR.exists():
        return config.NEWS_CLEAN_DIR
    logger.warning(
        "Cleaned news dir not found (%s) — falling back to %s",
        config.NEWS_CLEAN_DIR, config.NEWS_DATA_DIR,
    )
    return config.NEWS_DATA_DIR


def load_all_articles(year: str | None = None) -> pd.DataFrame:
    source  = _news_source()
    pattern = f"news_{year}_*.csv" if year else "news_*.csv"
    files   = sorted(
        p for p in source.glob(pattern)
        if not any(s in p.stem for s in ["_classified", "_oda"])
    )

    if not files:
        logger.error("No CSV files found in %s matching %s", source, pattern)
        sys.exit(1)

    logger.info("Found %d CSV files in %s", len(files), source.name)
    chunks = []
    for f in files:
        try:
            df = pd.read_csv(
                f, dtype=str, encoding="utf-8-sig", low_memory=False,
                usecols=lambda c: c in [
                    "article_id", "date", "outlet", "title",
                    "category", "keywords", "top_keywords", "body",
                ],
            )
            df["_source_file"] = f.name
            chunks.append(df)
        except Exception as e:
            logger.warning("Could not read %s: %s", f.name, e)

    if not chunks:
        logger.error("No articles loaded.")
        sys.exit(1)

    all_df = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %d articles total", len(all_df))
    return all_df


@click.command()
@click.option("--n", default=1000, show_default=True, type=int,
              help="Total sample size")
@click.option("--seed", default=2025, show_default=True, type=int,
              help="Random seed for reproducibility")
@click.option("--year", default="",
              help="Restrict to a single year (e.g. 2016)")
@click.option("--stratify/--no-stratify", default=True,
              help="Stratify by year proportionally (default: on)")
def main(n: int, seed: int, year: str, stratify: bool) -> None:
    df = load_all_articles(year or None)

    df["_year"] = df["date"].astype(str).str[:4] if "date" in df.columns else "unknown"

    if stratify and not year:
        year_counts = df["_year"].value_counts()
        parts = []
        for yr, count in year_counts.items():
            n_yr  = max(1, round(n * count / len(df)))
            n_yr  = min(n_yr, count)
            parts.append(df[df["_year"] == yr].sample(n=n_yr, random_state=seed))
        sample = pd.concat(parts, ignore_index=True)
        # Trim or top-up to exactly n
        if len(sample) > n:
            sample = sample.sample(n=n, random_state=seed)
        elif len(sample) < n:
            remaining = df.drop(sample.index)
            gap       = min(n - len(sample), len(remaining))
            sample    = pd.concat(
                [sample, remaining.sample(n=gap, random_state=seed)],
                ignore_index=True,
            )
    else:
        n      = min(n, len(df))
        sample = df.sample(n=n, random_state=seed)

    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample["oda_relevant"] = ""   # coder fills: 1 or 0

    out_cols = [c for c in [
        "article_id", "date", "outlet", "title", "category",
        "keywords", "body", "oda_relevant", "_source_file",
    ] if c in sample.columns]
    out = sample[out_cols].rename(columns={"_source_file": "source_file"})

    out_path = LABELS_DIR / "sample_for_labeling.csv"
    out.to_csv(out_path, index=True, index_label="row_id", encoding="utf-8-sig")

    click.echo(f"\nSample written: {len(out):,} articles → {out_path.relative_to(BASE_DIR)}")
    click.echo("\nYear distribution:")
    for yr, cnt in sample["_year"].value_counts().sort_index().items():
        click.echo(f"  {yr}: {cnt:,}")
    click.echo("\nInstructions:")
    click.echo("  Fill in the 'oda_relevant' column:  1 = ODA-related,  0 = not ODA-related")
    click.echo(f"  Save as: src/labels/sample_labeled.csv")
    click.echo(f"  Then run: python train_oda_classifier.py")


if __name__ == "__main__":
    main()
