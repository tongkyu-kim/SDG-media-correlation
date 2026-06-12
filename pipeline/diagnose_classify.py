"""
Diagnostic run: classify a small sample across 2007-2009 and print
quality statistics to inform fine-tuning decisions.

Outputs:
  - SDG distribution (which SDGs fire, how often)
  - Score histogram
  - Sentiment distribution
  - 3 example articles per SDG (highest-scoring)
  - Translation quality spot-check
"""

import sys, warnings, json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from classify.sdg_classifier import SDGClassifier
from classify.sentiment_analyzer import SentimentAnalyzer

NEWS_DIR = Path(__file__).parent.parent / "src" / "news"
SAMPLE_PER_YEAR = 2   # files per year

# ── Collect sample files ──────────────────────────────────────────────────────
sample_files = []
for yr in [2007, 2008, 2009]:
    files = sorted((NEWS_DIR / str(yr)).glob("*.csv"))
    files = [f for f in files if "_classified" not in f.name]
    # Pick evenly-spaced files across the year
    step = max(1, len(files) // SAMPLE_PER_YEAR)
    sample_files += files[::step][:SAMPLE_PER_YEAR]

print(f"Sample files ({len(sample_files)}):")
for f in sample_files:
    print(f"  {f.parent.name}/{f.name}")

# ── Load sample rows ──────────────────────────────────────────────────────────
ROWS_PER_FILE = 80
frames = []
for f in sample_files:
    df = pd.read_csv(f, nrows=ROWS_PER_FILE, dtype=str, encoding="utf-8-sig")
    df["_source_file"] = f.name
    frames.append(df)

df_all = pd.concat(frames, ignore_index=True)
print(f"\nTotal articles in sample: {len(df_all)}")

# Build classification input texts
def build_text(row):
    title = str(row.get("title") or "")
    kw    = str(row.get("keywords") or "")
    cat   = str(row.get("category1") or "")
    # Include category1 for extra context (e.g. '사회', '경제')
    return f"{title} {cat} {kw}"[:700]

texts = [build_text(r) for _, r in df_all.iterrows()]

# ── Run classification ────────────────────────────────────────────────────────
print("\nRunning SDG classification ...")
sdg_clf  = SDGClassifier()
sdg_results = sdg_clf.classify_batch(texts, batch_size=16)

print("Running sentiment analysis ...")
sent_clf = SentimentAnalyzer()
sent_results = sent_clf.analyze_batch(texts, batch_size=32)

# ── Attach results ────────────────────────────────────────────────────────────
df_all["sdg_label"]        = [r.sdg_label     for r in sdg_results]
df_all["sdg_score"]        = [r.sdg_score     for r in sdg_results]
df_all["sdg_intensity"]    = [r.sdg_intensity for r in sdg_results]
df_all["sdg_favorability"] = [r.label         for r in sent_results]
df_all["sentiment_score"]  = [r.score         for r in sent_results]

# Also capture translation for spot-check
print("Spot-checking translations (first 10 articles) ...")
spot_texts = texts[:10]
en_translations = sdg_clf._translate(spot_texts)

# ── Statistics ────────────────────────────────────────────────────────────────
relevant = df_all[df_all["sdg_label"] > 0]
total    = len(df_all)
n_rel    = len(relevant)

print(f"\n{'='*60}")
print(f"CLASSIFICATION SUMMARY  ({total} articles)")
print(f"{'='*60}")
print(f"  SDG-relevant (label > 0) : {n_rel} / {total}  ({100*n_rel/total:.1f}%)")
print(f"  Not relevant (label = 0) : {total - n_rel}")

print(f"\n--- SDG distribution (relevant only) ---")
sdg_counts = Counter(df_all[df_all["sdg_label"] > 0]["sdg_label"])
SDG_NAMES = {
    1:"No Poverty",2:"Zero Hunger",3:"Good Health",4:"Quality Education",
    5:"Gender Equality",6:"Clean Water",7:"Clean Energy",8:"Decent Work",
    9:"Industry & Innovation",10:"Reduced Inequality",11:"Sustainable Cities",
    12:"Responsible Consumption",13:"Climate Action",14:"Life Below Water",
    15:"Life on Land",16:"Peace & Justice",17:"Partnerships"
}
for sdg, cnt in sorted(sdg_counts.items()):
    bar = "█" * int(cnt / max(sdg_counts.values()) * 20)
    print(f"  SDG {sdg:2d} {SDG_NAMES.get(sdg,''):<25} {cnt:4d}  {bar}")

print(f"\n--- Score distribution (all articles) ---")
bins = [(0,.1),(0.1,.2),(0.2,.25),(0.25,.35),(0.35,.5),(0.5,.65),(0.65,1.01)]
for lo, hi in bins:
    n = ((df_all["sdg_score"] >= lo) & (df_all["sdg_score"] < hi)).sum()
    bar = "█" * int(n / total * 40)
    print(f"  [{lo:.2f}-{hi:.2f})  {n:4d}  {bar}")

print(f"\n--- Intensity distribution ---")
for i in range(4):
    n = (df_all["sdg_intensity"] == i).sum()
    print(f"  Intensity {i}: {n:4d}  ({100*n/total:.1f}%)")

print(f"\n--- Sentiment distribution ---")
for lbl, cnt in Counter(df_all["sdg_favorability"]).most_common():
    print(f"  {lbl:<10} {cnt:4d}  ({100*cnt/total:.1f}%)")

print(f"\n--- Translation spot-check (KO → EN) ---")
for i, (ko, en) in enumerate(zip(spot_texts, en_translations)):
    print(f"  [{i+1}] KO: {ko[:80]}")
    print(f"       EN: {en[:80]}")
    print()

print(f"\n--- Top examples per SDG ---")
for sdg in sorted(sdg_counts.keys()):
    group = df_all[df_all["sdg_label"] == sdg].nlargest(3, "sdg_score")
    print(f"\n  SDG {sdg} – {SDG_NAMES.get(sdg,'')} (top 3):")
    for _, row in group.iterrows():
        print(f"    score={row['sdg_score']}  intensity={row['sdg_intensity']}")
        print(f"    title: {str(row.get('title',''))[:80]}")
        print(f"    keywords: {str(row.get('keywords',''))[:60]}")
        print()
