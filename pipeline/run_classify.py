"""
Hybrid SDG classification pipeline for BigKinds Korean news CSVs.

Strategy
--------
1. Keyword classifier (no GPU) runs on ALL articles — identifies SDG candidates
   and extracts extended variables (aid_stance, issue_intensity, issue_frame,
   problem_solution, crisis_type, policy_actor).

2. BERT classifier (GPU) runs only on SDG candidates + policy-actor articles
   (~5-15% of corpus) — provides higher-accuracy SDG labels, confidence scores,
   secondary SDG, and sentiment.

3. Results are merged: BERT output takes priority over keyword output for the
   variables it covers; keyword output fills remaining columns.

Output: <filename>_classified.csv  (original columns + 12 added columns)

Added columns
-------------
  sdg_label          int   Primary SDG 1-17 (0 = not SDG-relevant)
  sdg_secondary      int   Secondary SDG 1-17 (0 = none)
  sdg_score          float Confidence 0.0-1.0 (BERT) or keyword density proxy
  sdg_intensity      int   0-3 relevance level (BERT) or keyword proxy
  sdg_favorability   str   positive | neutral | negative  (backward-compat)
  sentiment_score    float Confidence of sentiment prediction (0.5 if keyword-only)
  issue_intensity    int   0-5 severity/urgency scale (keyword-based)
  aid_stance         str   supportive | neutral | opposed  (keyword-based)
  issue_frame        str   dominant frame (keyword-based)
  problem_solution   str   problem | solution | mixed | neutral (keyword-based)
  crisis_type        str   comma-separated crisis types (keyword-based)
  policy_actor       int   1 if Korean ODA actor mentioned (keyword-based)

Usage
-----
  python run_classify.py                       # all unclassified CSVs, GPU if available
  python run_classify.py --year 2019           # one year only
  python run_classify.py --file path/to.csv    # single file
  python run_classify.py --batch-size 128      # larger batch for more GPU memory
  python run_classify.py --sdg-only            # skip sentiment analysis
  python run_classify.py --force               # re-classify already-done files
  python run_classify.py --keyword-only        # skip BERT entirely (keyword fallback only)
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

sys.path.insert(0, str(Path(__file__).parent))
import config as _cfg
from classify.keyword_classifier import KeywordClassifier
from classify.sdg_classifier import _sim_to_intensity
from classify.candidate_filter import compute_signals, is_candidate
from reference.countries_ko import detect_countries

# Raw/clean news source; ODA-filtered output lands in PROCESSED_DIR
NEWS_DIR      = _cfg.NEWS_CLEAN_DIR if _cfg.NEWS_CLEAN_DIR.exists() else _cfg.NEWS_DATA_DIR
PROCESSED_DIR = Path(__file__).parent.parent / "src" / "processed" / "news"
ODA_DIR       = PROCESSED_DIR   # train_oda_classifier.py writes *_oda.csv here

# ── Output column names ────────────────────────────────────────────────────────

BERT_COLS    = ["sdg_label", "sdg_labels", "sdg_secondary", "sdg_score",
                "sdg_intensity", "sdg_favorability", "sentiment_score",
                "sentiment_continuous"]
KEYWORD_COLS = ["issue_intensity", "aid_stance", "issue_frame",
                "problem_solution", "crisis_type", "policy_actor"]
ALL_ADDED    = BERT_COLS + KEYWORD_COLS + ["country_iso3"]

# Text columns used as BERT input
_TEXT_COLS_EN = ["title", "keywords"]
_TEXT_COLS_KO = ["제목", "키워드"]


def _build_input_text(row: pd.Series) -> str:
    """title + keywords — compact, matches BERT training context."""
    parts = []
    for col in _TEXT_COLS_EN + _TEXT_COLS_KO:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
        if len(parts) == 2:
            break
    return " ".join(parts)


def _secondary_from_scores(sdg_result) -> int:
    """Extract second-highest SDG from BERT all_scores dict (works for both result types)."""
    if not sdg_result.all_scores:
        return 0
    primary = getattr(sdg_result, "sdg_top", getattr(sdg_result, "sdg_label", 0))
    sorted_sdgs = sorted(sdg_result.all_scores.items(), key=lambda x: x[1], reverse=True)
    for sdg, score in sorted_sdgs:
        if sdg != primary and score >= 0.20:
            return sdg
    return 0


# ── Per-file classification ────────────────────────────────────────────────────

def classify_file(
    csv_path: Path,
    kw_clf: KeywordClassifier,
    sdg_clf,        # SDGClassifier or None
    sent_clf,       # SentimentAnalyzer or None
    batch_size: int,
    sdg_only: bool,
    bert_min_hits: int = 2,
    oda_filtered: bool = False,
) -> Path:
    """Classify one CSV and write *_classified.csv. Returns output path."""
    out_path = PROCESSED_DIR / (csv_path.stem + "_classified.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig", low_memory=False)
    if df.empty:
        logger.warning("Empty file, skipping: %s", csv_path.name)
        return out_path

    # When using ODA-filtered files, restrict BERT to ODA-relevant articles only
    if oda_filtered and "oda_relevant" in df.columns:
        n_before = len(df)
        df = df[df["oda_relevant"].astype(str) == "1"].copy()
        logger.info("  ODA filter: %d/%d articles kept", len(df), n_before)
        if df.empty:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            return out_path

    # ── Step 1: keyword classify ALL articles (fast, vectorized) ──────────────
    kw = kw_clf.classify_dataframe(df)

    # Initialize output with keyword defaults
    df["sdg_label"]             = kw["kw_sdg_label"]
    df["sdg_labels"]            = kw["kw_sdg_label"].apply(  # multi-label placeholder
        lambda x: str(x) if str(x) != "0" else ""
    )
    df["sdg_secondary"]         = kw["kw_sdg_secondary"]
    df["sdg_score"]             = kw["kw_sdg_score"]
    df["sdg_intensity"]         = kw["kw_sdg_intensity"]
    df["sdg_favorability"]      = kw["kw_sdg_favorability"]
    df["sentiment_score"]       = 0.5
    df["sentiment_continuous"]  = 0.0
    df["issue_intensity"]       = kw["issue_intensity"]
    df["aid_stance"]            = kw["aid_stance"]
    df["issue_frame"]           = kw["issue_frame"]
    df["problem_solution"]      = kw["problem_solution"]
    df["crisis_type"]           = kw["crisis_type"]
    df["policy_actor"]          = kw["policy_actor"]

    # ── Country detection — all articles, always ───────────────────────────────
    # Stores pipe-joined ISO3 codes per article (e.g. "VNM|KHM|ETH" or "").
    # Used by aggregate_media.py to build the country × SDG × year panel.
    _text_long = kw_clf._text_long(df, kw_clf._text_short(df))
    df["country_iso3"] = _text_long.apply(
        lambda t: "|".join(detect_countries(t)) if t else ""
    )

    # ── Step 2: BERT on development-relevant candidates only ─────────────────
    if sdg_clf is not None:
        # v2 candidate rule (see classify/candidate_filter.py): OR across
        # policy_actor / keyword-hits / ODA-country / dev-vocab signals,
        # loosened from v1's AND-conjunction to raise recall.
        signals = compute_signals(df, kw_clf)
        bert_mask = is_candidate(kw, signals, bert_min_hits=bert_min_hits)
        n_candidates = bert_mask.sum()

        if n_candidates > 0:
            logger.info("  BERT: %d/%d candidate articles ...", n_candidates, len(df))

            cand_df    = df[bert_mask]
            cand_texts = [_build_input_text(row) for _, row in cand_df.iterrows()]
            cand_idx   = cand_df.index

            # SDG classification (multi-label)
            sdg_results = sdg_clf.classify_multilabel_batch(cand_texts, batch_size=batch_size)

            df.loc[cand_idx, "sdg_label"]    = [r.sdg_top   for r in sdg_results]
            df.loc[cand_idx, "sdg_labels"]   = [
                "|".join(str(s) for s in r.sdg_labels) for r in sdg_results
            ]
            df.loc[cand_idx, "sdg_score"]    = [r.sdg_top_score for r in sdg_results]
            df.loc[cand_idx, "sdg_intensity"]= [
                _sim_to_intensity(r.sdg_top_score) for r in sdg_results
            ]
            df.loc[cand_idx, "sdg_secondary"]= [_secondary_from_scores(r) for r in sdg_results]

            # Sentiment — only on BERT-confirmed SDG-relevant articles
            if not sdg_only and sent_clf is not None:
                sdg_relevant_mask = pd.Series(
                    [r.sdg_top > 0 for r in sdg_results], index=cand_idx
                )
                rel_indices = cand_idx[sdg_relevant_mask.values]
                rel_texts   = [cand_texts[i] for i, flag in enumerate(sdg_relevant_mask) if flag]

                if len(rel_texts) > 0:
                    logger.info("  Sentiment: %d SDG-relevant articles ...", len(rel_texts))
                    sent_results = sent_clf.analyze_batch(rel_texts, batch_size=batch_size)
                    df.loc[rel_indices, "sdg_favorability"]     = [r.label      for r in sent_results]
                    df.loc[rel_indices, "sentiment_score"]      = [r.score      for r in sent_results]
                    df.loc[rel_indices, "sentiment_continuous"] = [r.continuous for r in sent_results]

    # ── Write output ───────────────────────────────────────────────────────────
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_sdg = (df["sdg_label"].astype(str) != "0").sum()
    logger.info("  Wrote %s  (%d/%d SDG-relevant)", out_path.name, n_sdg, len(df))
    return out_path


# ── File discovery ─────────────────────────────────────────────────────────────

def find_csv_files(year: str | None = None, oda_filtered: bool = False) -> list[Path]:
    if oda_filtered:
        # Read *_oda.csv files produced by train_oda_classifier.py;
        # only include rows where oda_relevant==1 during classify_file.
        pattern = f"news_{year}_*_oda.csv" if year else "news_*_oda.csv"
        return sorted(
            p for p in ODA_DIR.glob(pattern)
            if not p.stem.endswith("_classified")
        )
    pattern = f"news_{year}_*.csv" if year else "news_*.csv"
    return sorted(
        p for p in NEWS_DIR.glob(pattern)
        if not any(s in p.stem for s in ["_classified", "_oda"])
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--file", "single_file", default="", metavar="PATH",
              help="Classify a single CSV file")
@click.option("--year", default="", help="Restrict to one year (e.g. 2019)")
@click.option("--batch-size", default=128, show_default=True, type=int,
              help="Articles per E5 inference batch (128 optimal for RTX 3080)")
@click.option("--sdg-only", is_flag=True,
              help="Skip sentiment analysis (faster)")
@click.option("--bert-min-hits", default=1, show_default=True, type=int,
              help="Min keyword hits required to send an article to BERT (policy-actor articles always included); "
                   "v2 rule treats this as one of several OR'd signals, not an AND-requirement with country mention")
@click.option("--keyword-only", is_flag=True,
              help="Skip BERT entirely — keyword classification only (no GPU needed)")
@click.option("--oda-filtered", is_flag=True,
              help="Read *_oda.csv files from train_oda_classifier.py and skip oda_relevant==0 rows")
@click.option("--force", is_flag=True,
              help="Re-classify files that already have a _classified.csv")
def main(
    single_file: str,
    year: str,
    batch_size: int,
    sdg_only: bool,
    bert_min_hits: int,
    keyword_only: bool,
    oda_filtered: bool,
    force: bool,
) -> None:
    # ── Collect files ──────────────────────────────────────────────────────────
    if single_file:
        files = [Path(single_file)]
    else:
        files = find_csv_files(year or None, oda_filtered=oda_filtered)

    if not force:
        files = [
            f for f in files
            if not (PROCESSED_DIR / (f.stem + "_classified.csv")).exists()
        ]

    if not files:
        click.echo("Nothing to classify (use --force to re-run existing).")
        return

    click.echo(f"Files to process: {len(files)}")

    # ── Load models ────────────────────────────────────────────────────────────
    kw_clf = KeywordClassifier()
    click.echo("Keyword classifier ready.")

    sdg_clf  = None
    sent_clf = None

    if not keyword_only:
        try:
            from classify.sdg_classifier import SDGClassifier
            from classify.sentiment_analyzer import SentimentAnalyzer
            sdg_clf = SDGClassifier()
            if not sdg_only:
                sent_clf = SentimentAnalyzer()
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            click.echo(f"BERT models loaded — device: {device}  batch_size: {batch_size}")
        except ImportError as e:
            click.echo(f"Warning: BERT unavailable ({e}). Running keyword-only.")

    # ── Process ────────────────────────────────────────────────────────────────
    errors = []
    for csv_path in tqdm(files, unit="file", desc="Classifying"):
        try:
            classify_file(
                csv_path, kw_clf, sdg_clf, sent_clf,
                batch_size=batch_size, sdg_only=sdg_only,
                bert_min_hits=bert_min_hits, oda_filtered=oda_filtered,
            )
        except Exception as exc:
            logger.error("Failed %s: %s", csv_path.name, exc, exc_info=True)
            errors.append((csv_path, exc))

    click.echo(f"\nDone.  {len(files) - len(errors)} succeeded,  {len(errors)} failed.")
    if errors:
        for p, e in errors:
            click.echo(f"  {p.name}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
