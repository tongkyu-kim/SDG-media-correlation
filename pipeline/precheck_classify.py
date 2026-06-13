"""
Pre-check classifier output quality before running the full 697-file pipeline.

Runs the full keyword + E5 pipeline on one file and reports:
  1. Candidate reduction from the country-mention filter
  2. SDG distribution (keyword vs E5)
  3. Spot-check: top-5 articles per SDG (E5-classified) for manual review
  4. Agreement rate between keyword and E5 labels
  5. Score distribution (histogram by decile)

Usage:
  py precheck_classify.py                   # uses first 2016 file
  py precheck_classify.py --file path/to.csv
"""

from __future__ import annotations
import sys, io, logging
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import click

from classify.keyword_classifier import KeywordClassifier
from classify.sdg_classifier import SDGClassifier
from reference.countries_ko import detect_countries, detect_oda_recipient_countries

logging.basicConfig(level=logging.WARNING)

SDG_NAMES = {
    1:"No Poverty", 2:"Zero Hunger", 3:"Good Health", 4:"Quality Education",
    5:"Gender Equality", 6:"Clean Water", 7:"Clean Energy", 8:"Decent Work",
    9:"Industry & Innovation", 10:"Reduced Inequality", 11:"Sustainable Cities",
    12:"Responsible Consumption", 13:"Climate Action", 14:"Life Below Water",
    15:"Life on Land", 16:"Peace & Justice", 17:"Partnerships",
}

NEWS_DIR = Path(__file__).parent.parent / "src" / "news"


@click.command()
@click.option("--file", "csv_path", default="", help="Path to CSV file (default: first 2016 file)")
@click.option("--bert-min-hits", default=2, type=int)
def main(csv_path: str, bert_min_hits: int) -> None:
    if csv_path:
        f = Path(csv_path)
    else:
        files = sorted((NEWS_DIR / "2016").glob("*.csv"))
        if not files:
            click.echo("No 2016 files found. Pass --file path/to.csv")
            return
        f = files[0]

    click.echo(f"\nPre-check file: {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

    # ── Load ───────────────────────────────────────────────────────────────────
    df = pd.read_csv(f, dtype=str, encoding="utf-8-sig", low_memory=False)
    click.echo(f"Rows: {len(df):,}")

    title_col = "제목" if "제목" in df.columns else "title"
    kw_col    = "키워드" if "키워드" in df.columns else "keywords"
    body_col  = "본문" if "본문" in df.columns else ("body" if "body" in df.columns else None)

    # ── Keyword classifier ─────────────────────────────────────────────────────
    click.echo("\n[1/4] Keyword classification ...")
    kw_clf = KeywordClassifier()
    kw = kw_clf.classify_dataframe(df)

    n_kw_sdg = (kw["kw_sdg_label"] > 0).sum()
    click.echo(f"  Keyword SDG-relevant: {n_kw_sdg:,} / {len(df):,} ({n_kw_sdg/len(df)*100:.1f}%)")

    # ── Country filter ─────────────────────────────────────────────────────────
    click.echo("\n[2/4] Country-mention filter (ODA recipients only) ...")
    text_long = kw_clf._text_long(df, kw_clf._text_short(df))
    has_any_country = text_long.apply(lambda t: bool(detect_countries(t)))
    has_oda_country = text_long.apply(lambda t: bool(detect_oda_recipient_countries(t)))

    bert_mask = (
        (kw["policy_actor"] == 1) |
        ((kw["kw_sdg_hits"] >= bert_min_hits) & has_oda_country)
    )
    n_bert = bert_mask.sum()
    n_all_country  = ((kw["kw_sdg_hits"] >= bert_min_hits) & has_any_country).sum()
    n_oda_country  = ((kw["kw_sdg_hits"] >= bert_min_hits) & has_oda_country).sum()
    n_kw_only = ((kw["kw_sdg_hits"] >= bert_min_hits) & ~has_oda_country & (kw["policy_actor"] == 0)).sum()

    click.echo(f"  Any country mention:        {n_all_country:,} (old filter)")
    click.echo(f"  ODA recipient country only: {n_oda_country:,} (new filter — excludes USA/Japan/etc)")
    click.echo(f"  BERT candidates: {n_bert:,} ({n_bert/len(df)*100:.1f}%)")
    click.echo(f"    of which policy_actor=1: {(kw['policy_actor']==1).sum():,}")
    click.echo(f"    of which keyword+ODA-country: {n_oda_country:,}")
    click.echo(f"  Keyword-hits but no ODA country (keyword-only): {n_kw_only:,}")

    # ── E5 classification on BERT candidates (with translation) ──────────────
    click.echo(f"\n[3/4] Translate + E5 SDG classification on {n_bert:,} candidates ...")
    sdg_clf = SDGClassifier()
    sdg_clf._load_translator()
    sdg_clf._load()

    cand_df = df[bert_mask].copy()
    if body_col:
        texts = (cand_df[title_col].fillna("") + " " +
                 cand_df[kw_col].fillna("") + " " +
                 cand_df[body_col].fillna("").str[:200]).tolist()
    else:
        texts = (cand_df[title_col].fillna("") + " " + cand_df[kw_col].fillna("")).tolist()

    results = sdg_clf.classify_batch(texts, batch_size=256)

    e5_labels  = [r.sdg_label  for r in results]
    e5_scores  = [r.sdg_score  for r in results]
    cand_df = cand_df.copy()
    cand_df["e5_sdg"]   = e5_labels
    cand_df["e5_score"] = e5_scores
    cand_df["kw_sdg"]   = kw.loc[bert_mask, "kw_sdg_label"].values

    # ── Analysis ───────────────────────────────────────────────────────────────
    click.echo("\n[4/4] Results\n" + "="*60)

    # SDG distribution
    e5_relevant = cand_df[cand_df["e5_sdg"] > 0]
    click.echo(f"\nE5-classified as SDG-relevant: {len(e5_relevant):,} / {n_bert:,} candidates "
               f"({len(e5_relevant)/max(n_bert,1)*100:.1f}%)")

    if len(e5_relevant) > 0:
        sdg_counts = e5_relevant["e5_sdg"].value_counts().sort_index()
        click.echo("\nSDG distribution (E5):")
        for sdg, cnt in sdg_counts.items():
            bar = "█" * int(cnt / max(sdg_counts) * 30)
            click.echo(f"  SDG{sdg:2d} {SDG_NAMES.get(sdg,''):<28} {cnt:5d}  {bar}")

    # Agreement between keyword and E5
    both_sdg = cand_df[(cand_df["e5_sdg"] > 0) & (cand_df["kw_sdg"] > 0)]
    if len(both_sdg) > 0:
        agree = (both_sdg["e5_sdg"] == both_sdg["kw_sdg"]).sum()
        click.echo(f"\nKeyword ↔ E5 agreement (where both assigned): "
                   f"{agree}/{len(both_sdg)} = {agree/len(both_sdg)*100:.1f}%")

    # Score distribution
    if len(e5_relevant) > 0:
        scores = e5_relevant["e5_score"].values
        click.echo(f"\nE5 score distribution (SDG-relevant only):")
        click.echo(f"  min={scores.min():.2f}  median={np.median(scores):.2f}  "
                   f"p75={np.percentile(scores,75):.2f}  max={scores.max():.2f}")

    # Spot-check: top 3 articles per SDG (highest E5 score)
    click.echo("\n" + "="*60)
    click.echo("SPOT-CHECK: Top 3 articles per SDG (verify development relevance)")
    click.echo("="*60)
    for sdg in sorted(sdg_counts.index if len(e5_relevant) > 0 else []):
        sub = (e5_relevant[e5_relevant["e5_sdg"] == sdg]
               .sort_values("e5_score", ascending=False)
               .head(3))
        click.echo(f"\nSDG{sdg} — {SDG_NAMES.get(sdg,'')}:")
        for _, row in sub.iterrows():
            title = str(row.get(title_col, "")).strip()[:80]
            score = row["e5_score"]
            country_flag = "🌍" if detect_oda_recipient_countries(str(row.get(title_col,"")) +
                                                                  str(row.get(kw_col,""))) else "🇰🇷"
            click.echo(f"  [{score:.2f}] {country_flag} {title}")

    click.echo("\nDone. Review the spot-check above — look for domestic Korean articles "
               "in the results. If any appear, raise --bert-min-hits or inspect the filter.")


if __name__ == "__main__":
    main()
