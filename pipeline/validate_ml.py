"""
Step 10 — Validate ML classification outputs.

Four checks:
  oda       ODA classifier accuracy on held-out labeled data
  sdg       SDG classifier accuracy on a hand-coded validation sample
  irc       Inter-coder reliability for manual labeling (Cohen's kappa)
  replication  50% hand-coded vs 50% ML-coded comparison
                (pass criteria: R² > 0.90, slope b within [0.95, 1.05])

Usage:
  python validate_ml.py --task oda
  python validate_ml.py --task sdg  --sdg-labels src/labels/sdg_validation.csv
  python validate_ml.py --task irc  --coder1 ... --coder2 ...
  python validate_ml.py --task replication
  python validate_ml.py --task all
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
import config

BASE_DIR     = Path(__file__).parent.parent
LABELS_DIR   = config.LABELS_DIR
REPORTS_DIR  = BASE_DIR / "src" / "validation"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── ODA classifier validation ─────────────────────────────────────────────────

def validate_oda(labels_path: Path) -> dict:
    """20% held-out evaluation of the trained ODA classifier."""
    try:
        import pickle
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score,
            recall_score, roc_auc_score, classification_report, confusion_matrix,
        )
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.error("scikit-learn required: pip install scikit-learn")
        return {}

    model_path = config.MODELS_DIR / "oda_classifier.pkl"
    if not model_path.exists():
        logger.error("ODA model not found. Run train_oda_classifier.py first.")
        return {}

    if not labels_path.exists():
        logger.error("Labels file not found: %s", labels_path)
        return {}

    with open(model_path, "rb") as fh:
        pipeline = pickle.load(fh)

    from train_oda_classifier import build_text

    df = pd.read_csv(labels_path, dtype=str, encoding="utf-8-sig", low_memory=False)
    df = df[df["oda_relevant"].isin(["0", "1", "0.0", "1.0"])].copy()
    df["oda_relevant"] = df["oda_relevant"].astype(float).astype(int)

    texts  = df.apply(build_text, axis=1).tolist()
    labels = df["oda_relevant"].tolist()

    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    results = {
        "n_test":    len(y_test),
        "n_pos":     int(sum(y_test)),
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }

    click.echo("\n── ODA Classifier (20% held-out) ───────────────────────────────")
    for k, v in results.items():
        click.echo(f"  {k:<12}  {v}")

    click.echo(f"\n{classification_report(y_test, y_pred, target_names=['non-ODA', 'ODA'])}")
    cm = confusion_matrix(y_test, y_pred)
    click.echo(f"Confusion matrix  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

    return results


# ── SDG classifier validation ─────────────────────────────────────────────────

def validate_sdg(sdg_labels_path: Path) -> dict:
    """
    Accuracy of SDG classifier on a hand-coded validation set.

    Expected columns: article_id, sdg_hand (int 0-17), sdg_ml (int 0-17)
    """
    try:
        from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
    except ImportError:
        logger.error("scikit-learn required: pip install scikit-learn")
        return {}

    if not sdg_labels_path.exists():
        logger.error(
            "SDG validation file not found: %s\n"
            "Expected columns: article_id, sdg_hand, sdg_ml",
            sdg_labels_path,
        )
        return {}

    df = pd.read_csv(sdg_labels_path, dtype=str, encoding="utf-8-sig")
    df = df[df["sdg_hand"].notna() & df["sdg_ml"].notna()].copy()
    y_hand = df["sdg_hand"].astype(float).astype(int).tolist()
    y_ml   = df["sdg_ml"].astype(float).astype(int).tolist()

    results = {
        "n":        len(y_hand),
        "accuracy": round(accuracy_score(y_hand, y_ml), 4),
        "f1_macro": round(f1_score(y_hand, y_ml, average="macro", zero_division=0), 4),
        "kappa":    round(cohen_kappa_score(y_hand, y_ml), 4),
    }

    click.echo("\n── SDG Classifier Validation ───────────────────────────────────")
    for k, v in results.items():
        click.echo(f"  {k:<12}  {v}")

    click.echo("\n  Per-SDG accuracy (hand-coded subset):")
    for sdg in range(0, 18):
        idx    = [i for i, h in enumerate(y_hand) if h == sdg]
        if len(idx) < 3:
            continue
        h_sub  = [y_hand[i] for i in idx]
        m_sub  = [y_ml[i]   for i in idx]
        acc    = sum(h == m for h, m in zip(h_sub, m_sub)) / len(h_sub)
        label  = f"SDG {sdg}" if sdg > 0 else "Non-SDG"
        click.echo(f"    {label:<10}  n={len(h_sub):>4}  acc={acc:.3f}")

    return results


# ── Inter-coder reliability ───────────────────────────────────────────────────

def validate_irc(coder1_path: Path, coder2_path: Path) -> dict:
    """
    Cohen's kappa for manual ODA labeling agreement.

    Both files must have: article_id, oda_relevant (1 or 0)
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        logger.error("scikit-learn required: pip install scikit-learn")
        return {}

    for p in [coder1_path, coder2_path]:
        if not p.exists():
            logger.error("File not found: %s", p)
            return {}

    df1 = pd.read_csv(coder1_path, dtype=str, encoding="utf-8-sig")
    df2 = pd.read_csv(coder2_path, dtype=str, encoding="utf-8-sig")

    merged = df1.merge(
        df2[["article_id", "oda_relevant"]],
        on="article_id",
        suffixes=("_c1", "_c2"),
    )
    merged = merged[
        merged["oda_relevant_c1"].isin(["0", "1"]) &
        merged["oda_relevant_c2"].isin(["0", "1"])
    ].copy()

    if len(merged) < 10:
        logger.error("Need ≥10 jointly coded articles (got %d)", len(merged))
        return {}

    y1    = merged["oda_relevant_c1"].astype(int).tolist()
    y2    = merged["oda_relevant_c2"].astype(int).tolist()
    agree = sum(a == b for a, b in zip(y1, y2))
    kappa = cohen_kappa_score(y1, y2)

    interp = (
        "almost perfect (>0.80)" if kappa > 0.80 else
        "substantial (0.60–0.80)" if kappa > 0.60 else
        "moderate (0.40–0.60)"    if kappa > 0.40 else
        "fair (0.20–0.40)"        if kappa > 0.20 else
        "slight (<0.20)"
    )

    results = {
        "n_jointly_coded": len(merged),
        "pct_agreement":   round(agree / len(y1), 4),
        "cohen_kappa":     round(kappa, 4),
        "interpretation":  interp,
    }

    click.echo("\n── Inter-Coder Reliability ─────────────────────────────────────")
    for k, v in results.items():
        click.echo(f"  {k:<22}  {v}")

    if kappa < 0.60:
        click.echo("\n  WARNING: kappa < 0.60 — review coding guidelines before proceeding")
    else:
        click.echo("\n  OK: kappa ≥ 0.60")

    return results


# ── 50/50 replication check ───────────────────────────────────────────────────

def validate_replication(panel_path: Path, hand_share: float = 0.50) -> dict:
    """
    Split the panel randomly into hand-coded and ML-coded halves,
    regress ML ~ hand, check R² > 0.90 and slope b within [0.95, 1.05].
    """
    try:
        from scipy import stats
    except ImportError:
        logger.error("scipy required: pip install scipy")
        return {}

    if not panel_path.exists():
        logger.error("Panel file not found: %s", panel_path)
        logger.info("Run build_panel.py first.")
        return {}

    df = pd.read_csv(panel_path, encoding="utf-8-sig")

    target = "freq_articles"
    if target not in df.columns:
        logger.error("Panel must have '%s' column", target)
        return {}

    # Random 50/50 split on (year, sdg) cells
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split   = int(len(df) * hand_share)
    df_hand = df.iloc[:split]
    df_ml   = df.iloc[split:]

    hand_agg = df_hand.groupby(["year", "sdg"])[target].sum()
    ml_agg   = df_ml.groupby(  ["year", "sdg"])[target].sum()
    joint    = hand_agg.to_frame("hand").join(ml_agg.rename("ml"), how="inner").dropna()

    if len(joint) < 5:
        logger.error("Not enough matched cells for replication check (got %d)", len(joint))
        return {}

    slope, intercept, r, p, _ = stats.linregress(joint["hand"], joint["ml"])
    r2 = r ** 2

    pass_r2    = r2 > 0.90
    pass_slope = abs(slope - 1.0) < 0.05

    results = {
        "n_cells":    len(joint),
        "slope_b":    round(slope, 4),
        "intercept":  round(intercept, 4),
        "r_squared":  round(r2, 4),
        "p_value":    round(p, 6),
        "pass_r2":    pass_r2,
        "pass_slope": pass_slope,
        "overall":    "PASS" if (pass_r2 and pass_slope) else "FAIL",
    }

    click.echo("\n── 50/50 Hand vs ML Replication Check ─────────────────────────")
    for k, v in results.items():
        click.echo(f"  {k:<14}  {v}")

    status = "PASS" if (pass_r2 and pass_slope) else "FAIL"
    click.echo(
        f"\n  {status}: R²={r2:.4f} (need >0.90), b={slope:.4f} (need 0.95–1.05)"
    )

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--task", default="all",
              type=click.Choice(["oda", "sdg", "irc", "replication", "all"]),
              help="Which validation task to run")
@click.option("--labels",
              default=str(LABELS_DIR / "sample_labeled.csv"), show_default=True,
              help="Manual ODA labels CSV (for oda task)")
@click.option("--sdg-labels",
              default=str(LABELS_DIR / "sdg_validation.csv"), show_default=True,
              help="Hand-coded SDG labels CSV (columns: article_id, sdg_hand, sdg_ml)")
@click.option("--coder1",
              default=str(LABELS_DIR / "coder1_labels.csv"), show_default=True,
              help="Coder 1 labels CSV (article_id, oda_relevant)")
@click.option("--coder2",
              default=str(LABELS_DIR / "coder2_labels.csv"), show_default=True,
              help="Coder 2 labels CSV (article_id, oda_relevant)")
@click.option("--panel",
              default=str(BASE_DIR / "src" / "processed" / "panel" / "panel_sdg_month.csv"),
              show_default=True,
              help="Panel CSV for replication check")
@click.option("--out", default="",
              help="Output report CSV (default: src/validation/validation_report.csv)")
def main(task: str, labels: str, sdg_labels: str,
         coder1: str, coder2: str, panel: str, out: str) -> None:

    all_results: dict[str, dict] = {}

    if task in ("oda", "all"):
        all_results["oda"] = validate_oda(Path(labels))

    if task in ("sdg", "all"):
        all_results["sdg"] = validate_sdg(Path(sdg_labels))

    if task in ("irc", "all"):
        all_results["irc"] = validate_irc(Path(coder1), Path(coder2))

    if task in ("replication", "all"):
        all_results["replication"] = validate_replication(Path(panel))

    # Write summary report
    rows = [
        {"task": task_name, "metric": k, "value": str(v)}
        for task_name, res in all_results.items()
        if res
        for k, v in res.items()
    ]

    if rows:
        report   = pd.DataFrame(rows)
        out_path = Path(out) if out else REPORTS_DIR / "validation_report.csv"
        report.to_csv(out_path, index=False, encoding="utf-8-sig")
        click.echo(f"\nReport → {out_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
