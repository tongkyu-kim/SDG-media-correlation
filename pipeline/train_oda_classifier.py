"""
Step 3 — Train and apply ODA relevance classifier.

Trains a TF-IDF + LogisticRegression classifier on ~1,000 manually labeled
articles (oda_relevant = 1 or 0) and applies it to all cleaned news CSVs,
writing *_oda.csv files with oda_relevant and oda_prob columns.

Only ODA-relevant articles should proceed to SDG classification (step 4).
The trained model is saved to models/oda_classifier.pkl.

Usage:
  python train_oda_classifier.py                 # train and apply to all CSVs
  python train_oda_classifier.py --eval-only     # cross-validation only, no save/apply
  python train_oda_classifier.py --apply-only    # use saved model, skip retraining
  python train_oda_classifier.py --year 2016     # apply to one year only
  python train_oda_classifier.py --force         # overwrite existing _oda.csv files
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
import config

BASE_DIR     = Path(__file__).parent.parent
LABELS_DIR   = config.LABELS_DIR
MODELS_DIR   = config.MODELS_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH  = MODELS_DIR / "oda_classifier.pkl"
OUT_DIR     = BASE_DIR / "src" / "processed" / "news"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Seed keywords for feature augmentation and keyword-only fallback
_ODA_SEEDS: list[str] = [
    "ODA", "공적개발원조", "원조", "개발협력", "KOICA", "코이카", "EDCF",
    "무상원조", "유상원조", "개발도상국", "수원국", "공여국",
    "국제개발", "해외원조", "개발원조",
    "OECD DAC", "세계은행", "ADB", "UNDP", "유엔개발",
    "빈곤퇴치", "식수위생", "농업개발", "역량강화", "기술협력",
    "지속가능발전", "SDG", "새천년개발목표", "MDG",
]


# ── Feature building ──────────────────────────────────────────────────────────

def build_text(row: pd.Series) -> str:
    title    = str(row.get("title",    "") or "")
    keywords = str(row.get("keywords", "") or "")
    body     = str(row.get("body",     "") or "")[:500]
    return f"{title} {keywords} {body}".strip()


def keyword_fallback(text: str) -> int:
    t = text.lower()
    return int(any(kw.lower() in t for kw in _ODA_SEEDS))


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(labels_path: Path, eval_only: bool = False):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.metrics import classification_report
        from sklearn.pipeline import Pipeline
    except ImportError:
        logger.error("scikit-learn required: pip install scikit-learn")
        sys.exit(1)

    if not labels_path.exists():
        logger.error(
            "Labels file not found: %s\n"
            "Run sample_for_labeling.py first, annotate, then save as sample_labeled.csv",
            labels_path,
        )
        sys.exit(1)

    df = pd.read_csv(labels_path, dtype=str, encoding="utf-8-sig", low_memory=False)

    if "oda_relevant" not in df.columns:
        logger.error("Labels file must have an 'oda_relevant' column (values: 1 or 0)")
        sys.exit(1)

    df = df[df["oda_relevant"].isin(["0", "1", "0.0", "1.0"])].copy()
    df["oda_relevant"] = df["oda_relevant"].astype(float).astype(int)

    if len(df) < 50:
        logger.error("Need at least 50 labeled articles (got %d)", len(df))
        sys.exit(1)

    texts  = df.apply(build_text, axis=1).tolist()
    labels = df["oda_relevant"].tolist()

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    logger.info("Labels: %d ODA-relevant, %d non-ODA, %d total", n_pos, n_neg, len(labels))

    # Character n-gram TF-IDF is robust for Korean morphology without a tokenizer
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=50_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )),
    ])

    logger.info("Running 5-fold stratified cross-validation ...")
    cv = cross_validate(
        pipeline, texts, labels,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=["accuracy", "f1", "precision", "recall", "roc_auc"],
    )

    click.echo("\n── Cross-validation (5-fold, stratified) ──────────────────────")
    for key, scores in cv.items():
        if key.startswith("test_"):
            name = key[5:]
            click.echo(f"  {name:<12}  {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    if eval_only:
        return None

    logger.info("Training final model on all %d labeled articles ...", len(texts))
    pipeline.fit(texts, labels)

    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(pipeline, fh)
    logger.info("Model saved → %s", MODEL_PATH.relative_to(BASE_DIR))

    return pipeline


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model():
    if not MODEL_PATH.exists():
        logger.error(
            "No trained model at %s.\nRun without --apply-only to train first.",
            MODEL_PATH,
        )
        sys.exit(1)
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


def apply_to_file(csv_path: Path, pipeline, threshold: float = 0.5) -> Path:
    out_path = OUT_DIR / (csv_path.stem + "_oda.csv")

    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig", low_memory=False)
    if df.empty:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    texts = df.apply(build_text, axis=1).tolist()

    try:
        proba          = pipeline.predict_proba(texts)[:, 1]
        df["oda_prob"] = proba.round(4)
        df["oda_relevant"] = (proba >= threshold).astype(int)
    except Exception as exc:
        logger.error("Prediction failed for %s: %s — using keyword fallback", csv_path.name, exc)
        df["oda_prob"]     = 0.0
        df["oda_relevant"] = [keyword_fallback(t) for t in texts]

    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_oda = int(df["oda_relevant"].sum())
    logger.info("  %s: %d/%d ODA-relevant (%.1f%%)",
                csv_path.name, n_oda, len(df), n_oda / len(df) * 100)
    return out_path


def find_news_files(year: str | None = None) -> list[Path]:
    source  = config.NEWS_CLEAN_DIR if config.NEWS_CLEAN_DIR.exists() else config.NEWS_DATA_DIR
    pattern = f"news_{year}_*.csv" if year else "news_*.csv"
    return sorted(
        p for p in source.glob(pattern)
        if not any(s in p.stem for s in ["_oda", "_classified"])
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--labels",
              default=str(config.LABELS_DIR / "sample_labeled.csv"),
              show_default=True,
              help="CSV with manual ODA labels (oda_relevant = 1 or 0)")
@click.option("--apply-only", is_flag=True,
              help="Skip training, apply the saved model")
@click.option("--eval-only", is_flag=True,
              help="Cross-validation metrics only — do not save or apply")
@click.option("--year", default="", help="Apply to one year (e.g. 2016)")
@click.option("--threshold", default=0.5, show_default=True, type=float,
              help="Probability threshold for ODA-positive classification")
@click.option("--force", is_flag=True,
              help="Overwrite already-processed _oda.csv files")
def main(labels: str, apply_only: bool, eval_only: bool,
         year: str, threshold: float, force: bool) -> None:

    if apply_only:
        pipeline = load_model()
    else:
        pipeline = train_model(Path(labels), eval_only=eval_only)
        if eval_only:
            return
        pipeline = load_model()   # reload from disk to verify save

    files = find_news_files(year or None)
    if not force:
        files = [f for f in files
                 if not (OUT_DIR / (f.stem + "_oda.csv")).exists()]

    if not files:
        click.echo("Nothing to classify (use --force to overwrite existing files).")
        return

    click.echo(f"\nApplying ODA classifier to {len(files)} files (threshold={threshold}) ...")
    errors: list[Path] = []
    for f in tqdm(files, unit="file"):
        try:
            apply_to_file(f, pipeline, threshold=threshold)
        except Exception as exc:
            logger.error("Failed %s: %s", f.name, exc)
            errors.append(f)

    click.echo(f"\nDone. {len(files) - len(errors)} succeeded, {len(errors)} failed.")
    click.echo(f"ODA-filtered files → {OUT_DIR.relative_to(BASE_DIR)}/")
    click.echo("\nNext: python run_classify.py --oda-filtered")


if __name__ == "__main__":
    main()
