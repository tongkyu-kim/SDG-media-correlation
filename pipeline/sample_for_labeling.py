"""
Generate the annotation sheet for manual labeling.

Reads news CSVs, pre-scores every article with the keyword classifier +
country detector, and draws a year-stratified sample ENRICHED for
development/ODA candidates (not a plain random sample).

Rationale: a plain random sample of Korean daily news is ~95%+ domestic
content, since only ~3-6% of raw articles pass even the loose keyword+
country filter (see docs/CLASSIFICATION_METHODS.md). Coding a sample that
skewed produces too few positive/borderline examples for coders to
calibrate against each other, which tanks inter-coder kappa even when raw
percent agreement looks fine. Instead, articles are bucketed into three
strata before sampling:

  candidate   passes the same policy_actor / keyword+country criteria
              run_classify.py uses to send articles to BERT — the
              highest-value cases to hand-label.
  borderline  some SDG keyword hits OR a country mention, but not both —
              the genuinely ambiguous cases that build coder calibration.
  negative    no keyword hits and no country mention — plain domestic
              news, kept as a smaller share just to confirm true negatives.

Because this is an enriched (not simple random) sample, the *prevalence*
of positives in the labeled sheet does NOT represent the corpus-wide
prevalence — don't report "X% of articles are development-relevant" from
this sample. Precision/recall/F1 computed within the "candidate" stratum
are valid corpus-level estimates for THAT stratum (which is also exactly
the population the classifier is applied to). The `stratum` column is
kept in the output so metrics can be reported/reweighted per stratum.

Coders fill in the annotation columns (prefixed label_*).
A second coder does the same overlap subset so inter-rater
reliability (Cohen's kappa) can be computed.

Usage:
  python sample_for_labeling.py                  # 600 articles, all years
  python sample_for_labeling.py --n 800          # larger sample
  python sample_for_labeling.py --overlap 150    # size of two-coder overlap block
  python sample_for_labeling.py --pct-candidate 0.5 --pct-borderline 0.3 --pct-negative 0.2
  python sample_for_labeling.py --seed 42
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd

# Windows consoles often default to a legacy codepage (e.g. cp949 on Korean
# locale) that can't encode the arrows/dashes in the log/summary output below.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
import config
from classify.keyword_classifier import KeywordClassifier
from classify.candidate_filter import compute_signals, classify_stratum

BASE_DIR   = Path(__file__).parent.parent
DOCS_DIR   = BASE_DIR / "docs"
LABELS_DIR = config.LABELS_DIR     # src/labels/ — pipeline reads labeled data from here
DOCS_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

_NEWS_DIR = config.NEWS_CLEAN_DIR if config.NEWS_CLEAN_DIR.exists() else config.NEWS_DATA_DIR


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


def _stratify(df: pd.DataFrame, kw_clf: KeywordClassifier, bert_min_hits: int = 1) -> pd.Series:
    """
    Bucket articles using the same v2 candidate rule run_classify.py uses to
    route articles to BERT (classify/candidate_filter.py), so the "candidate"
    stratum matches the population the real classifier actually scores.
    """
    kw = kw_clf.classify_dataframe(df)
    signals = compute_signals(df, kw_clf)
    return classify_stratum(kw, signals, bert_min_hits=bert_min_hits)


def enriched_stratified_sample(
    files: list[Path],
    n: int,
    seed: int,
    files_per_year: int = 3,
    max_rows_per_file: int = 8000,
    pct_candidate: float = 0.5,
    pct_borderline: float = 0.3,
    pct_negative: float = 0.2,
) -> pd.DataFrame:
    """
    Reads `files_per_year` random files per year (capped at `max_rows_per_file`
    rows each), scores every row with the keyword classifier + country
    detector, buckets into candidate/borderline/negative strata (see
    _stratify), then samples each year's quota proportionally across strata —
    oversampling candidate/borderline relative to their true rarity so the
    annotation sheet isn't 95%+ obvious negatives.
    """
    import random
    rng = random.Random(seed)

    year_files: dict[str, list[Path]] = {}
    for f in files:
        parts = f.stem.split("_")          # ["news", "2016", "03"]
        yr    = parts[1] if len(parts) >= 2 else "unknown"
        year_files.setdefault(yr, []).append(f)

    years = sorted(year_files.keys())
    logger.info("Found %d years: %s", len(years), years)

    kw_clf = KeywordClassifier()
    base_quota = max(1, n // len(years))

    parts: list[pd.DataFrame] = []
    for yr in years:
        yr_files = year_files[yr]
        chosen = rng.sample(yr_files, min(files_per_year, len(yr_files)))
        logger.info("  %s: reading %d/%d files ...", yr, len(chosen), len(yr_files))

        yr_parts = [_read_file(f, n_rows=max_rows_per_file) for f in chosen]
        yr_df    = pd.concat([d for d in yr_parts if d is not None], ignore_index=True)
        if yr_df.empty:
            continue

        yr_df["_stratum"] = _stratify(yr_df, kw_clf)
        counts = yr_df["_stratum"].value_counts()
        logger.info(
            "    scored %d articles -> candidate=%d borderline=%d negative=%d",
            len(yr_df), counts.get("candidate", 0),
            counts.get("borderline", 0), counts.get("negative", 0),
        )

        quota  = min(base_quota, len(yr_df))
        target = {
            "candidate":  round(quota * pct_candidate),
            "borderline": round(quota * pct_borderline),
            "negative":   round(quota * pct_negative),
        }

        # Candidate/borderline are scarce and taken first; any shortfall is
        # made up from negative, which is always plentiful.
        yr_sample_parts: list[pd.DataFrame] = []
        leftover = 0
        for stratum in ["candidate", "borderline", "negative"]:
            pool = yr_df[yr_df["_stratum"] == stratum]
            want = target[stratum] + leftover if stratum == "negative" else target[stratum]
            take = min(want, len(pool))
            if stratum != "negative":
                leftover += target[stratum] - take
            if take > 0:
                yr_sample_parts.append(pool.sample(n=take, random_state=seed))

        if yr_sample_parts:
            parts.append(pd.concat(yr_sample_parts, ignore_index=True))

    if not parts:
        logger.error("No articles could be sampled.")
        sys.exit(1)

    result = pd.concat(parts, ignore_index=True)
    if len(result) > n:
        result = result.sample(n=n, random_state=seed)

    counts = result["_stratum"].value_counts()
    logger.info(
        "Sampled %d articles total -> candidate=%d borderline=%d negative=%d",
        len(result), counts.get("candidate", 0),
        counts.get("borderline", 0), counts.get("negative", 0),
    )
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


@click.command()
@click.option("--n",       default=600, show_default=True, type=int)
@click.option("--overlap", default=150, show_default=True, type=int,
              help="First N rows are labeled by BOTH coders for kappa")
@click.option("--seed",    default=2025, show_default=True, type=int)
@click.option("--year",    default="", help="Restrict to one year")
@click.option("--files-per-year",    default=3,    show_default=True, type=int)
@click.option("--max-rows-per-file", default=8000, show_default=True, type=int,
              help="Cap rows read per file when pre-scoring for strata (keeps runtime bounded)")
@click.option("--pct-candidate",  default=0.5, show_default=True, type=float,
              help="Share of each year's quota drawn from the 'candidate' stratum")
@click.option("--pct-borderline", default=0.3, show_default=True, type=float,
              help="Share drawn from the 'borderline' stratum")
@click.option("--pct-negative",   default=0.2, show_default=True, type=float,
              help="Share drawn from the 'negative' (clear non-development) stratum")
def main(
    n: int, overlap: int, seed: int, year: str,
    files_per_year: int, max_rows_per_file: int,
    pct_candidate: float, pct_borderline: float, pct_negative: float,
) -> None:
    pattern = f"news_{year}_*.csv" if year else "news_*.csv"
    files = sorted(
        p for p in _NEWS_DIR.glob(pattern)
        if not any(s in p.stem for s in ["_classified", "_oda", "_devrel"])
    )
    if not files:
        logger.error("No files found in %s matching %s", _NEWS_DIR, pattern)
        sys.exit(1)
    sample = enriched_stratified_sample(
        files, n, seed,
        files_per_year=files_per_year,
        max_rows_per_file=max_rows_per_file,
        pct_candidate=pct_candidate,
        pct_borderline=pct_borderline,
        pct_negative=pct_negative,
    )
    sample = sample.rename(columns={"_stratum": "stratum"})

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
        "stratum",
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
    click.echo("\nStratum distribution (candidate/borderline = enriched, not corpus-representative):")
    for st, cnt in sample["stratum"].value_counts().items():
        click.echo(f"  {st}: {cnt:,}")

    _print_instructions(docs_path, overlap)


def _print_instructions(path: Path, overlap: int) -> None:
    click.echo("""
╔══════════════════════════════════════════════════════════════════╗
║                    ANNOTATION INSTRUCTIONS                       ║
╚══════════════════════════════════════════════════════════════════╝

0. This sheet is ENRICHED, not a plain random sample: the 'stratum'
   column shows why each article was picked ('candidate' = looks
   development/ODA-relevant by keyword+country match, 'borderline' =
   some signal but ambiguous, 'negative' = no signal, kept as a control
   group). Don't report the share of 1s in this sheet as "X% of Korean
   news is development-relevant" — it isn't a representative sample of
   the corpus, it's deliberately weighted toward the interesting cases.

1. Upload sample_for_labeling.csv to Google Sheets.

2. BOTH coders label rows where overlap_row = YES first.
   Compare results and resolve disagreements before continuing —
   discuss the 'candidate' and 'borderline' rows especially, since
   that's where genuine judgment calls live.

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
