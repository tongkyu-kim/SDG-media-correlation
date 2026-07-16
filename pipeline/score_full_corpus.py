"""
Score the full 2007-2025 corpus with the validated precision scorer
(structured rule signals + TF-IDF text -- "Scorer C" in
evaluate_precision_scorer.py, PR-AUC 0.61 vs 0.30 for structured-only in
5-fold CV on the 560-row ground truth), and keep only candidate/borderline
rows -- the negative stratum had 0/119 positives in ground truth, not worth
scoring.

Trains the final model on ALL 560 hand-coded ground-truth rows (not a CV
split -- cross-validation already happened in evaluate_precision_scorer.py
to pick this approach), then streams through every news_YYYY_MM.csv source
file once. Unlike build_sampling_frame.py + sample_from_frame.py's two-pass
design (lightweight frame now, re-read text later for just the sampled
rows), this script keeps text for every kept row as it streams, since it's
already reading it -- avoids a second 32GB read later.

Writes incrementally (append per file) to bound memory -- do not
accumulate all rows in memory before writing.

Output: src/processed/scored_candidates_full_2007_2025.csv
  (article_id, source_file, date, year, stratum, kw_sdg_hits, policy_actor,
   precision_score, title, keywords, top_keywords, body_preview)

Usage:
  ./.venv/Scripts/python.exe pipeline/score_full_corpus.py
  ./.venv/Scripts/python.exe pipeline/score_full_corpus.py --pattern "news_2024_*.csv"  # test on a subset first
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import config as _cfg
from classify.keyword_classifier import KeywordClassifier
from classify.candidate_filter import compute_signals, classify_stratum

NEWS_DIR = _cfg.NEWS_CLEAN_DIR if _cfg.NEWS_CLEAN_DIR.exists() else _cfg.NEWS_DATA_DIR
LABELS_PATH = Path(__file__).parent.parent / "src" / "labels" / "sample_labeled.csv"
DEFAULT_OUT = Path(__file__).parent.parent / "src" / "processed" / "scored_candidates_full_2007_2025.csv"

STRUCT_FEATURE_COLS = [
    "kw_sdg_hits", "policy_actor", "kw_sdg_label_count",
    "has_dev_vocab", "has_cooccur_sector", "has_oda_country",
]


def _kw_label_count(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(lambda s: 0 if s in ("0", "nan") else len(s.split(",")))


def train_final_model(kw_clf: KeywordClassifier):
    """Train on all 560 labeled rows (no held-out split -- CV already done separately)."""
    df = pd.read_csv(LABELS_PATH, dtype={"article_id": str})
    df["body"] = df["body_preview"]

    kw = kw_clf.classify_dataframe(df)
    signals = compute_signals(df, kw_clf)

    X_struct = pd.DataFrame({
        "kw_sdg_hits": kw["kw_sdg_hits"].astype(float),
        "policy_actor": kw["policy_actor"].astype(float),
        "kw_sdg_label_count": _kw_label_count(kw["kw_sdg_label"]).astype(float),
        "has_dev_vocab": signals["has_dev_vocab"].astype(float),
        "has_cooccur_sector": signals["has_cooccur_sector"].astype(float),
        "has_oda_country": signals["has_oda_country"].astype(float),
    })

    text = df["title"].fillna("") + " " + df["keywords"].fillna("") + " " + df["body_preview"].fillna("")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    X_text = tfidf.fit_transform(text)

    X = hstack([csr_matrix(X_struct.values), X_text])
    y = df["label_development_relevant"].astype(int)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)

    n_pos = int(y.sum())
    click.echo(f"Trained final model on {len(df)} rows ({n_pos} positive) with {X.shape[1]} features\n")
    return clf, tfidf


def score_dataframe(df: pd.DataFrame, kw_clf: KeywordClassifier, clf: LogisticRegression, tfidf: TfidfVectorizer) -> pd.DataFrame:
    kw = kw_clf.classify_dataframe(df)
    signals = compute_signals(df, kw_clf)
    stratum = classify_stratum(kw, signals, bert_min_hits=1)

    keep = stratum != "negative"
    if not keep.any():
        return pd.DataFrame()

    df_k = df.loc[keep].reset_index(drop=True)
    kw_k = kw.loc[keep].reset_index(drop=True)
    sig_k = {k: v.loc[keep].reset_index(drop=True) for k, v in signals.items()}
    stratum_k = stratum.loc[keep].reset_index(drop=True)

    X_struct = pd.DataFrame({
        "kw_sdg_hits": kw_k["kw_sdg_hits"].astype(float),
        "policy_actor": kw_k["policy_actor"].astype(float),
        "kw_sdg_label_count": _kw_label_count(kw_k["kw_sdg_label"]).astype(float),
        "has_dev_vocab": sig_k["has_dev_vocab"].astype(float),
        "has_cooccur_sector": sig_k["has_cooccur_sector"].astype(float),
        "has_oda_country": sig_k["has_oda_country"].astype(float),
    })
    text = df_k["title"].fillna("") + " " + df_k["keywords"].fillna("") + " " + df_k.get("body", pd.Series("", index=df_k.index)).fillna("")
    X_text = tfidf.transform(text)
    X = hstack([csr_matrix(X_struct.values), X_text])
    scores = clf.predict_proba(X)[:, 1]

    return pd.DataFrame({
        "article_id": df_k["article_id"],
        "date": df_k["date"],
        "year": df_k["date"].astype(str).str[:4],
        "stratum": stratum_k,
        "kw_sdg_hits": kw_k["kw_sdg_hits"],
        "policy_actor": kw_k["policy_actor"],
        "precision_score": scores,
        "title": df_k["title"],
        "keywords": df_k["keywords"],
        "top_keywords": df_k.get("top_keywords", ""),
        "body_preview": df_k.get("body", pd.Series("", index=df_k.index)).str[:300],
    })


# --- Per-worker-process globals, set once via _init_worker so the model/vectorizer
# and compiled regex patterns aren't rebuilt for every file. ---
_worker_kw_clf: KeywordClassifier | None = None
_worker_clf: LogisticRegression | None = None
_worker_tfidf: TfidfVectorizer | None = None


def _init_worker(clf: LogisticRegression, tfidf: TfidfVectorizer) -> None:
    global _worker_kw_clf, _worker_clf, _worker_tfidf
    _worker_kw_clf = KeywordClassifier()
    _worker_clf = clf
    _worker_tfidf = tfidf


def _process_one_file(file_path_str: str, part_dir_str: str) -> tuple[str, int, int, bool]:
    f = Path(file_path_str)
    part_dir = Path(part_dir_str)
    done_marker = part_dir / f"{f.stem}.done"

    # Resume support: a .done marker (written after this file was fully
    # scored in a prior run, possibly on a different machine via Dropbox
    # sync) means skip re-reading/re-scoring -- just report its saved stats.
    if done_marker.exists():
        n_seen_str, n_kept_str = done_marker.read_text().strip().split(",")
        return f.name, int(n_seen_str), int(n_kept_str), True

    df = pd.read_csv(
        f, dtype=str, encoding="utf-8-sig",
        usecols=["article_id", "date", "title", "keywords", "top_keywords", "body"],
        low_memory=False,
    )
    if df.empty:
        done_marker.write_text("0,0")
        return f.name, 0, 0, False

    scored = score_dataframe(df, _worker_kw_clf, _worker_clf, _worker_tfidf)
    scored["source_file"] = f.name
    if not scored.empty:
        part_path = part_dir / f"{f.stem}.csv"
        scored.to_csv(part_path, index=False, encoding="utf-8-sig")
    done_marker.write_text(f"{len(df)},{len(scored)}")
    return f.name, len(df), len(scored), False


@click.command()
@click.option("--pattern", "patterns", multiple=True, default=["news_*.csv"], show_default=True,
              help="Glob pattern(s) within NEWS_DIR; repeatable")
@click.option("--out", "out_path", default=str(DEFAULT_OUT), show_default=True)
@click.option("--workers", default=None, type=int,
              help="Parallel worker processes. Default: cpu_count - 4 (leaves headroom for other work)")
def main(patterns: tuple[str, ...], out_path: str, workers: int | None) -> None:
    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(NEWS_DIR.glob(pat)))
    files = sorted(set(f for f in files if not any(s in f.stem for s in ["_classified", "_oda"])))

    if not files:
        click.echo(f"No files found in {NEWS_DIR} matching {patterns}")
        return

    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 4)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    part_dir = out.parent / f"{out.stem}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    # Backfill .done markers for part CSVs written by a pre-resume-support run
    # (n_seen is not recoverable for these -- approximated as n_kept, a
    # reporting-only figure, doesn't affect the final merged output).
    for f in files:
        done_marker = part_dir / f"{f.stem}.done"
        part_csv = part_dir / f"{f.stem}.csv"
        if not done_marker.exists() and part_csv.exists():
            n_kept = sum(1 for _ in open(part_csv, encoding="utf-8-sig")) - 1
            done_marker.write_text(f"{n_kept},{n_kept}")

    n_already_done = sum(1 for f in files if (part_dir / f"{f.stem}.done").exists())
    click.echo(f"Scoring {len(files)} files from {NEWS_DIR} with {workers} parallel workers "
               f"({n_already_done} already done from a prior run -- will be skipped) ...\n")

    kw_clf = KeywordClassifier()
    clf, tfidf = train_final_model(kw_clf)

    t0 = time.time()
    total_kept = 0
    total_seen = 0
    n_resumed = 0
    n_fresh = 0

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(clf, tfidf)) as pool:
        futures = {pool.submit(_process_one_file, str(f), str(part_dir)): f for f in files}
        with tqdm(total=len(files), unit="file") as pbar:
            for fut in as_completed(futures):
                name, n_seen, n_kept, was_resumed = fut.result()
                total_seen += n_seen
                total_kept += n_kept
                n_resumed += int(was_resumed)
                n_fresh += int(not was_resumed)
                elapsed = time.time() - t0
                pbar.update(1)
                tag = "resumed" if was_resumed else "scored"
                tqdm.write(f"  {name} [{tag}]: {n_seen:,} rows -> {n_kept:,} kept "
                           f"(running total: {total_kept:,} kept / {total_seen:,} seen, "
                           f"{n_resumed} resumed + {n_fresh} freshly scored, {elapsed:.0f}s elapsed)")

    click.echo("\nMerging part files ...")
    part_files = sorted(part_dir.glob("*.csv"))
    if out.exists():
        out.unlink()
    header_written = False
    for pf in tqdm(part_files, unit="part"):
        part = pd.read_csv(pf, dtype=str, encoding="utf-8-sig")
        part.to_csv(out, mode="a", index=False, header=not header_written, encoding="utf-8-sig")
        header_written = True

    # Cleanup is best-effort -- a transient Windows file lock (e.g. AV scan)
    # on the temp dir must not fail the job after the real output is written.
    for attempt in range(3):
        try:
            shutil.rmtree(part_dir)
            break
        except (PermissionError, OSError) as e:
            if attempt == 2:
                click.echo(f"Note: could not remove temp dir {part_dir} ({e}); safe to delete manually.")
            else:
                time.sleep(2)

    elapsed = time.time() - t0
    click.echo(f"\nDone. {total_kept:,} / {total_seen:,} rows kept ({total_kept/max(total_seen,1):.1%}) -> {out}")
    click.echo(f"Elapsed: {elapsed:.0f}s ({elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
