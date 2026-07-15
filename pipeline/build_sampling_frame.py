"""
Build a row-level sampling frame (article_id, date, stratum, score signals)
for a set of news_YYYY_MM.csv files, WITHOUT keeping full article text in
memory/output. This is the population any future annotation round should
draw from — a proper simple-random-sample-within-stratum over the FULL
corpus, rather than the lighter files_per_year/max_rows_per_file heuristic
in sample_for_labeling.py (which only pre-scores a handful of files per year
and is fine for a quick draw, but not a rigorous sampling frame).

Stratum definition matches classify/candidate_filter.py's v2 rule exactly
(OR across signals, not v1's AND-conjunction):
  candidate   policy_actor==1 OR kw_sdg_hits>=bert_min_hits OR ODA-country
              mention OR dev-vocab hit OR (cooccur-sector term + any country)
  borderline  not candidate, but mentions some (non-ODA/donor) country
  negative    no signal at all

Output: one row per article with just the columns needed to sample + rejoin
text later (article_id, source_file, date, year, month, stratum, kw_sdg_hits,
kw_sdg_label, policy_actor). Full text is re-read from source_file on demand
by sample_from_frame.py, keyed on article_id, so this frame stays small.

Usage:
  py build_sampling_frame.py --pattern "news_2024_*.csv" --pattern "news_2025_*.csv"
  py build_sampling_frame.py --out src/processed/sampling_frame_2024_2025.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import config as _cfg
from classify.keyword_classifier import KeywordClassifier
from classify.candidate_filter import compute_signals, classify_stratum

NEWS_DIR = _cfg.NEWS_CLEAN_DIR if _cfg.NEWS_CLEAN_DIR.exists() else _cfg.NEWS_DATA_DIR
DEFAULT_OUT = Path(__file__).parent.parent / "src" / "processed" / "sampling_frame_2024_2025.csv"


def _stratify(df: pd.DataFrame, kw_clf: KeywordClassifier, bert_min_hits: int) -> pd.DataFrame:
    kw = kw_clf.classify_dataframe(df)
    signals = compute_signals(df, kw_clf)
    stratum = classify_stratum(kw, signals, bert_min_hits=bert_min_hits)

    return pd.DataFrame({
        "article_id":    df["article_id"],
        "kw_sdg_hits":   kw["kw_sdg_hits"],
        "kw_sdg_label":  kw["kw_sdg_label"],
        "policy_actor":  kw["policy_actor"],
        "stratum":       stratum,
    })


@click.command()
@click.option("--pattern", "patterns", multiple=True,
              default=["news_2024_*.csv", "news_2025_*.csv"], show_default=True,
              help="Glob pattern(s) within NEWS_DIR; repeatable")
@click.option("--bert-min-hits", default=1, show_default=True, type=int)
@click.option("--out", "out_path", default=str(DEFAULT_OUT), show_default=True)
def main(patterns: tuple[str, ...], bert_min_hits: int, out_path: str) -> None:
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(NEWS_DIR.glob(pat)))
    files = sorted(set(f for f in files if not any(s in f.stem for s in ["_classified", "_oda"])))

    if not files:
        click.echo(f"No files found in {NEWS_DIR} matching {patterns}")
        return

    click.echo(f"Building sampling frame from {len(files)} files ...\n")

    kw_clf = KeywordClassifier()
    frame_parts = []
    t0 = time.time()

    for f in tqdm(files, unit="file"):
        df = pd.read_csv(f, dtype=str, encoding="utf-8-sig",
                          usecols=["article_id", "date", "title", "keywords", "top_keywords", "body"],
                          low_memory=False)
        if df.empty:
            continue

        part = _stratify(df, kw_clf, bert_min_hits)
        part["source_file"] = f.name
        part["date"] = df["date"]
        part["year"] = df["date"].astype(str).str[:4]
        part["month"] = df["date"].astype(str).str[4:6]

        counts = part["stratum"].value_counts()
        tqdm.write(f"  {f.name}: {len(part):,} rows -> "
                   f"candidate={counts.get('candidate', 0):,} "
                   f"borderline={counts.get('borderline', 0):,} "
                   f"negative={counts.get('negative', 0):,}")

        frame_parts.append(part)

    frame = pd.concat(frame_parts, ignore_index=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8-sig")

    elapsed = time.time() - t0
    counts = frame["stratum"].value_counts()
    click.echo(f"\nWrote {len(frame):,} rows to {out}")
    click.echo(f"  candidate={counts.get('candidate', 0):,}  "
               f"borderline={counts.get('borderline', 0):,}  "
               f"negative={counts.get('negative', 0):,}")
    click.echo(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
