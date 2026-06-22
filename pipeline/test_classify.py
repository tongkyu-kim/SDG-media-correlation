"""Quick smoke test: classify 5 articles and print results."""
import sys, warnings, logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import config as _cfg
import pandas as pd
from classify.sdg_classifier import SDGClassifier
from classify.sentiment_analyzer import SentimentAnalyzer

# Pick the first available CSV from the clean directory (or raw if not present)
_search_dir = _cfg.NEWS_CLEAN_DIR if _cfg.NEWS_CLEAN_DIR.exists() else _cfg.NEWS_DATA_DIR
_candidates = sorted(_search_dir.glob("news_*.csv"))
if not _candidates:
    sys.exit(f"No news_*.csv files found in {_search_dir}")

CSV = _candidates[0]
print(f"Test file: {CSV.name}\n")

df = pd.read_csv(CSV, nrows=5, dtype=str, encoding="utf-8-sig")

# Handle both Korean and English column names
_title_col = "제목" if "제목" in df.columns else "title"
_kw_col    = "키워드" if "키워드" in df.columns else "keywords"

texts = []
for _, row in df.iterrows():
    t = str(row.get(_title_col, "") or "")
    k = str(row.get(_kw_col, "") or "")
    texts.append(f"{t} {k}"[:800])

print(f"Testing on {len(texts)} articles ...\n")

print("--- SDG Classifier ---")
sdg = SDGClassifier()
for i, (text, res) in enumerate(zip(texts, sdg.classify_batch(texts, batch_size=5))):
    print(f"[{i+1}] SDG={res.sdg_label}  score={res.sdg_score}  intensity={res.sdg_intensity}")
    print(f"     title: {text[:70]}...")

print("\n--- Sentiment Analyzer ---")
sent = SentimentAnalyzer()
for i, (text, res) in enumerate(zip(texts, sent.analyze_batch(texts, batch_size=5))):
    print(f"[{i+1}] {res.label}  score={res.score}")
