"""
Cross-validate three candidate scorers against the reconciled 560-row
hand-coded ground truth (src/labels/sample_labeled.csv,
label_development_relevant is the target), to decide how much of a
full-corpus rescore is actually worth the time cost.

NOTE: article_id in sample_labeled.csv does not match the frame's article_id
format (looks like float-precision truncation somewhere upstream, e.g. an
Excel round-trip) -- a direct join finds zero matches. Instead, every
feature here is recomputed directly from each labeled row's own text
(title/keywords/body_preview), using the actual pipeline classifier code,
which is more correct anyway (guaranteed consistent with the current rule).

Scorer A ("frame-only"): kw_sdg_hits + policy_actor + kw_sdg_label only --
  the ONLY columns actually persisted in sampling_frame_full_2007_2025.csv
  for all 14.6M rows. Free to apply full-corpus: zero additional text I/O.
Scorer B ("full-signals"): adds has_dev_vocab / has_cooccur_sector /
  has_oda_country -- the rest of candidate_filter.py's rule inputs. These
  are NOT persisted in the frame, so applying to the full corpus means one
  full read of the 32GB text corpus (same cost as the original
  candidate-count scan, ~hours) to recompute them, but still no TF-IDF/ML.
Scorer C ("full-signals+TF-IDF"): adds a TF-IDF(title+keywords+body_preview)
  representation on top of B -- heaviest option.

Usage:
  ./.venv/Scripts/python.exe pipeline/evaluate_precision_scorer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from classify.keyword_classifier import KeywordClassifier
from classify.candidate_filter import compute_signals

LABELS_PATH = "src/labels/sample_labeled.csv"

df = pd.read_csv(LABELS_PATH, dtype={"article_id": str})
df["body"] = df["body_preview"]  # _text_long() looks for a "body" column

kw_clf = KeywordClassifier()
kw = kw_clf.classify_dataframe(df)
signals = compute_signals(df, kw_clf)

df["kw_sdg_hits"] = kw["kw_sdg_hits"]
df["policy_actor"] = kw["policy_actor"]
df["kw_sdg_label"] = kw["kw_sdg_label"]
df["has_dev_vocab"] = signals["has_dev_vocab"].astype(int)
df["has_cooccur_sector"] = signals["has_cooccur_sector"].astype(int)
df["has_oda_country"] = signals["has_oda_country"].astype(int)

y = df["label_development_relevant"].astype(int)
n_pos = int(y.sum())
print(f"Ground truth: {len(df)} rows, {n_pos} positive ({n_pos/len(df):.1%})\n")

# Sanity check: recomputed stratum-equivalent candidate flag should roughly
# match the file's own recorded `stratum` column, confirming the rule
# recomputation is behaving as expected.
recomputed_candidate = (
    (kw["policy_actor"] == 1)
    | ((kw["kw_sdg_hits"] >= 1) & signals["has_oda_country"])
    | signals["has_dev_vocab"]
    | signals["has_cooccur_sector"]
)
agreement = (recomputed_candidate == (df["stratum"] == "candidate")).mean()
print(f"Recomputed-candidate vs recorded-stratum agreement: {agreement:.1%}\n")

def precision_at_k(scores, y, k):
    order = np.argsort(-scores)[:k]
    return y.values[order].mean()


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ks = (50, 100, 200, 400)  # 560 total rows -- keep k within range

# kw_sdg_label is a comma-joined string of SDG numbers (e.g. "8,16") or "0" --
# turn into a count, no text needed.
kw_label_count = df["kw_sdg_label"].astype(str).apply(
    lambda s: 0 if s in ("0", "nan") else len(s.split(","))
)

# --- Scorer A: frame-only columns (zero extra text I/O on the full 14.6M-row corpus) ---
X_a = pd.DataFrame({
    "kw_sdg_hits": df["kw_sdg_hits"].astype(float),
    "policy_actor": df["policy_actor"].astype(float),
    "kw_sdg_label_count": kw_label_count.astype(float),
})
clf_a = LogisticRegression(max_iter=1000, class_weight="balanced")
scores_a = cross_val_predict(clf_a, X_a, y, cv=cv, method="predict_proba")[:, 1]

# --- Scorer B: + full rule signals (needs one full-corpus text pass, no ML/TF-IDF) ---
X_b = X_a.copy()
X_b["has_dev_vocab"] = df["has_dev_vocab"].astype(float)
X_b["has_cooccur_sector"] = df["has_cooccur_sector"].astype(float)
X_b["has_oda_country"] = df["has_oda_country"].astype(float)
clf_b = LogisticRegression(max_iter=1000, class_weight="balanced")
scores_b = cross_val_predict(clf_b, X_b, y, cv=cv, method="predict_proba")[:, 1]

# --- Scorer C: + TF-IDF text on top of B (heaviest) ---
text = (
    df["title"].fillna("") + " " + df["keywords"].fillna("") + " " + df["body_preview"].fillna("")
)
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
X_text = tfidf.fit_transform(text)
from scipy.sparse import hstack, csr_matrix

X_c = hstack([csr_matrix(X_b.values), X_text])
clf_c = LogisticRegression(max_iter=1000, class_weight="balanced")
scores_c = cross_val_predict(clf_c, X_c, y, cv=cv, method="predict_proba")[:, 1]

for name, scores in [
    ("A: frame-only (kw_sdg_hits + policy_actor + kw_sdg_label) -- free on full corpus", scores_a),
    ("B: + dev_vocab/cooccur_sector/oda_country -- needs 1 full-corpus text pass, no ML", scores_b),
    ("C: B + TF-IDF text -- heaviest, needs text pass + vectorization", scores_c),
]:
    ap = average_precision_score(y, scores)
    print(f"=== Scorer {name} ===")
    print(f"  Average precision (PR-AUC): {ap:.3f}")
    for k in ks:
        print(f"  Precision@{k}: {precision_at_k(scores, y, k):.3f}")
    print()

print("\n=== For reference: raw rule-based stratum precision (from SESSION_CONTEXT.md) ===")
print("  candidate stratum: 69/277 = 24.9%")
print("  borderline stratum: 4/164 = 2.4%")
print("  negative stratum: 0/119 = 0.0%")
