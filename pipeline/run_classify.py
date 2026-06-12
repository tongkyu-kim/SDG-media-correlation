"""
Batch SDG classification + sentiment analysis on BigKinds CSV files.

Input:  src/news/YYYY/MM/YYYY-MM-DD.csv  (from BigKinds download)
Output: src/news/YYYY/MM/YYYY-MM-DD_classified.csv
        (same file + 5 added columns)

Added columns:
  sdg_label        int   Primary SDG (1-17), 0 = not SDG-relevant
  sdg_score        float Confidence of SDG assignment (0-1)
  sdg_intensity    int   0=not relevant, 1=indirect, 2=moderate, 3=core
  sdg_favorability str   positive | neutral | negative
  sentiment_score  float Confidence of sentiment prediction

Classification input: title + " " + keywords  (fast & avoids 20k-token bodies)

Usage:
  python run_classify.py                          # all unclassified CSVs
  python run_classify.py --year 2019              # one year
  python run_classify.py --file src/news/2019/01/2019-01-07.csv
  python run_classify.py --batch-size 64          # tune for your GPU/RAM
  python run_classify.py --sdg-only               # skip sentiment
  python run_classify.py --force                  # re-classify already-done files
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

NEWS_DIR = Path(__file__).parent.parent / "src" / "news"

# Output column names
COL_SDG_LABEL   = "sdg_label"
COL_SDG_SCORE   = "sdg_score"
COL_SDG_INT     = "sdg_intensity"
COL_FAVOR       = "sdg_favorability"
COL_SENT_SCORE  = "sentiment_score"
ADDED_COLS      = [COL_SDG_LABEL, COL_SDG_SCORE, COL_SDG_INT, COL_FAVOR, COL_SENT_SCORE]

# Column names that hold the article text in BigKinds CSVs
TEXT_COLS_PRIMARY   = ["title", "keywords"]          # fast path (recommended)
TEXT_COLS_FALLBACK  = ["제목", "키워드"]             # if rename didn't happen


def _build_input_text(row: pd.Series) -> str:
    """Combine title + keywords into a single classification input."""
    parts = []
    for col in TEXT_COLS_PRIMARY + TEXT_COLS_FALLBACK:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
        if len(parts) == 2:
            break
    return " ".join(parts)


def classify_file(
    csv_path: Path,
    sdg_clf,
    sent_clf,
    batch_size: int,
    sdg_only: bool,
) -> Path:
    """Classify one CSV and write *_classified.csv. Returns output path."""
    out_path = csv_path.with_name(csv_path.stem + "_classified.csv")

    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    if df.empty:
        logger.warning("Empty file, skipping: %s", csv_path.name)
        return out_path

    # Build classification inputs
    texts = [_build_input_text(row) for _, row in df.iterrows()]

    # SDG classification
    logger.info("  SDG classification: %d articles ...", len(texts))
    sdg_results = sdg_clf.classify_batch(texts, batch_size=batch_size)
    df[COL_SDG_LABEL] = [r.sdg_label    for r in sdg_results]
    df[COL_SDG_SCORE] = [r.sdg_score    for r in sdg_results]
    df[COL_SDG_INT]   = [r.sdg_intensity for r in sdg_results]

    # Sentiment — run only on SDG-relevant articles to save time
    if sdg_only:
        df[COL_FAVOR]      = ""
        df[COL_SENT_SCORE] = None
    else:
        relevant_mask = df[COL_SDG_LABEL] > 0
        df[COL_FAVOR]      = "neutral"
        df[COL_SENT_SCORE] = 0.0

        if relevant_mask.any():
            relevant_texts = [texts[i] for i in df.index[relevant_mask]]
            logger.info("  Sentiment analysis: %d relevant articles ...", len(relevant_texts))
            sent_results = sent_clf.analyze_batch(relevant_texts, batch_size=batch_size)
            df.loc[relevant_mask, COL_FAVOR]      = [r.label for r in sent_results]
            df.loc[relevant_mask, COL_SENT_SCORE] = [r.score for r in sent_results]

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    n_sdg = (df[COL_SDG_LABEL] > 0).sum()
    logger.info("  Wrote %s  (%d/%d SDG-relevant)", out_path.name, n_sdg, len(df))
    return out_path


def find_csv_files(year: str | None = None) -> list[Path]:
    pattern = f"{year}/**/*.csv" if year else "**/*.csv"
    return sorted(
        p for p in NEWS_DIR.glob(pattern)
        if not p.stem.endswith("_classified")
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--file", "single_file", default="", metavar="PATH",
              help="Classify a single CSV file")
@click.option("--year", default="", help="Restrict to one year (e.g. 2019)")
@click.option("--batch-size", default=32, show_default=True, type=int,
              help="Articles per inference batch (raise for GPU, lower for CPU)")
@click.option("--sdg-only", is_flag=True,
              help="Skip sentiment analysis (faster)")
@click.option("--force", is_flag=True,
              help="Re-classify files that already have a _classified.csv")
def main(single_file: str, year: str, batch_size: int, sdg_only: bool, force: bool) -> None:
    from classify.sdg_classifier import SDGClassifier
    from classify.sentiment_analyzer import SentimentAnalyzer

    # Collect files
    if single_file:
        files = [Path(single_file)]
    else:
        files = find_csv_files(year or None)

    if not force:
        files = [f for f in files if not f.with_name(f.stem + "_classified.csv").exists()]

    if not files:
        click.echo("Nothing to classify.")
        return

    click.echo(f"Classifying {len(files)} files (batch_size={batch_size}, sdg_only={sdg_only})")

    # Load models once
    sdg_clf  = SDGClassifier()
    sent_clf = None if sdg_only else SentimentAnalyzer()

    errors = []
    for csv_path in tqdm(files, unit="file", desc="Classifying"):
        try:
            classify_file(csv_path, sdg_clf, sent_clf, batch_size, sdg_only)
        except Exception as exc:
            logger.error("Failed %s: %s", csv_path.name, exc)
            errors.append((csv_path, exc))

    click.echo(f"\nDone. {len(files) - len(errors)} succeeded, {len(errors)} failed.")
    if errors:
        for p, e in errors:
            click.echo(f"  {p.name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
