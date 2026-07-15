"""
Draw a stratified annotation sample from a pre-built sampling frame
(build_sampling_frame.py output) — a proper simple-random-sample-within-
stratum over the FULL scored population, not a per-year file subsample.

Text fields (title/keywords/top_keywords/body) are re-read from the
original news_YYYY_MM.csv source files only for the sampled article_ids,
keeping memory bounded regardless of frame size.

Usage:
  py sample_from_frame.py --frame src/processed/sampling_frame_2024_2025.csv \\
      --n 700 --overlap 175 --seed 4242 --tag round4_2024_2025

  py sample_from_frame.py --frame src/processed/sampling_frame_2024_2025.csv \\
      --n 500 --pct-candidate 1.0 --pct-borderline 0 --pct-negative 0 \\
      --seed 9001 --tag round4b_candidate_boost --exclude src/labels/sample_round4_2024_2025.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config

BASE_DIR   = Path(__file__).parent.parent
DOCS_DIR   = BASE_DIR / "docs"
LABELS_DIR = config.LABELS_DIR
NEWS_DIR   = config.NEWS_CLEAN_DIR if config.NEWS_CLEAN_DIR.exists() else config.NEWS_DATA_DIR


def _stratified_draw(
    frame: pd.DataFrame, n: int, seed: int,
    pct_candidate: float, pct_borderline: float, pct_negative: float,
) -> pd.DataFrame:
    target = {
        "candidate":  round(n * pct_candidate),
        "borderline": round(n * pct_borderline),
        "negative":   round(n * pct_negative),
    }
    parts = []
    leftover = 0
    for stratum in ["candidate", "borderline", "negative"]:
        pool = frame[frame["stratum"] == stratum]
        want = target[stratum] + leftover if stratum == "negative" else target[stratum]
        take = min(want, len(pool))
        if stratum != "negative":
            leftover += target[stratum] - take
        if take > 0:
            parts.append(pool.sample(n=take, random_state=seed))
        if take < want:
            click.echo(f"  WARNING: stratum '{stratum}' pool only had {len(pool):,} rows, "
                       f"wanted {want:,} — sample will be short by {want - take:,}")
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0]


@click.command()
@click.option("--frame", "frame_path", required=True, type=click.Path(exists=True))
@click.option("--n",       default=700, show_default=True, type=int)
@click.option("--overlap", default=175, show_default=True, type=int,
              help="First N rows are labeled by BOTH coders for kappa")
@click.option("--seed",    required=True, type=int, help="Use a fresh seed per round — never reuse a prior round's seed")
@click.option("--pct-candidate",  default=0.5, show_default=True, type=float)
@click.option("--pct-borderline", default=0.3, show_default=True, type=float)
@click.option("--pct-negative",   default=0.2, show_default=True, type=float)
@click.option("--tag", required=True, help="Suffix for output filenames, e.g. 'round4_2024_2025'")
@click.option("--exclude", "exclude_paths", multiple=True, type=click.Path(exists=True),
              help="Prior labeled/sampled CSV(s) whose article_id values must be excluded (repeatable)")
def main(
    frame_path: str, n: int, overlap: int, seed: int,
    pct_candidate: float, pct_borderline: float, pct_negative: float,
    tag: str, exclude_paths: tuple[str, ...],
) -> None:
    frame = pd.read_csv(frame_path, dtype=str, encoding="utf-8-sig")
    click.echo(f"Frame: {len(frame):,} rows from {frame_path}")

    if exclude_paths:
        exclude_ids: set[str] = set()
        for p in exclude_paths:
            prior = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
            if "article_id" in prior.columns:
                exclude_ids.update(prior["article_id"].dropna().tolist())
        before = len(frame)
        frame = frame[~frame["article_id"].isin(exclude_ids)]
        click.echo(f"Excluded {before - len(frame):,} previously-sampled article_ids "
                   f"({len(exclude_ids):,} ids from {len(exclude_paths)} file(s))")

    counts = frame["stratum"].value_counts()
    click.echo(f"Available pool -> candidate={counts.get('candidate', 0):,} "
               f"borderline={counts.get('borderline', 0):,} negative={counts.get('negative', 0):,}\n")

    sample = _stratified_draw(frame, n, seed, pct_candidate, pct_borderline, pct_negative)
    if sample.empty:
        click.echo("No rows sampled — check frame / pct arguments.")
        sys.exit(1)

    # ── Re-read text fields from source files, only for sampled article_ids ──
    needed_ids = set(sample["article_id"])
    by_file: dict[str, list[str]] = {}
    for src in sample["source_file"].unique():
        by_file[src] = None  # placeholder, we filter by article_id membership below

    text_parts = []
    for src in sample["source_file"].unique():
        f = NEWS_DIR / src
        if not f.exists():
            click.echo(f"  WARNING: source file missing, skipping: {f}")
            continue
        df = pd.read_csv(
            f, dtype=str, encoding="utf-8-sig",
            usecols=["article_id", "date", "outlet", "title", "keywords", "top_keywords", "body"],
        )
        df = df[df["article_id"].isin(needed_ids)]
        text_parts.append(df)

    text_df = pd.concat(text_parts, ignore_index=True) if text_parts else pd.DataFrame()
    out = sample.merge(text_df, on=["article_id", "date"], how="left", suffixes=("", "_text"))

    missing_text = out["title"].isna().sum() if "title" in out.columns else len(out)
    if missing_text:
        click.echo(f"  WARNING: {missing_text:,} sampled rows failed to rejoin text (source file mismatch?)")

    out["body_preview"] = out["body"].fillna("").str[:250] if "body" in out.columns else ""

    # Shuffle final row order so strata aren't blocked together, then mark overlap
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    out["overlap_row"] = ""
    out.loc[:overlap - 1, "overlap_row"] = "YES"

    for col in [
        "label_development_relevant", "label_sdg_labels", "label_sentiment_country",
        "label_crisis_flag", "label_crisis_type", "notes", "coder_id",
    ]:
        out[col] = ""

    out_cols = [
        "overlap_row", "stratum", "article_id", "date", "outlet", "title",
        "keywords", "top_keywords", "body_preview", "kw_sdg_hits", "kw_sdg_label", "policy_actor",
        "label_development_relevant", "label_sdg_labels", "label_sentiment_country",
        "label_crisis_flag", "label_crisis_type", "notes", "coder_id", "source_file",
    ]
    out = out[[c for c in out_cols if c in out.columns]]

    docs_path   = DOCS_DIR   / f"sample_{tag}.csv"
    labels_path = LABELS_DIR / f"sample_{tag}.csv"
    out.to_csv(docs_path,   index=False, encoding="utf-8-sig")
    out.to_csv(labels_path, index=False, encoding="utf-8-sig")

    click.echo(f"\nAnnotation sheet: {len(out):,} articles")
    click.echo(f"  -> {docs_path}")
    click.echo(f"  -> {labels_path}")
    click.echo(f"Overlap block (both coders): rows 0-{overlap - 1}")
    click.echo("\nStratum distribution:")
    for st, cnt in out["stratum"].value_counts().items():
        click.echo(f"  {st}: {cnt:,}")
    click.echo("\nYear/month distribution:")
    for ym, cnt in out["date"].astype(str).str[:6].value_counts().sort_index().items():
        click.echo(f"  {ym}: {cnt:,}")


if __name__ == "__main__":
    main()
