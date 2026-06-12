"""Quick smoke test: classify 5 articles and print results."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
from classify.sdg_classifier import SDGClassifier
from classify.sentiment_analyzer import SentimentAnalyzer

CSV = r"C:\Users\tkkim\OneDrive - ASEAN-Korea Centre\문서\GitHub\SDG-media-correlation\src\news\2007\NewsResult_20070101-20070107.csv"

df = pd.read_csv(CSV, nrows=5, dtype=str, encoding="utf-8-sig")

texts = []
for _, row in df.iterrows():
    t = str(row.get("title", "") or "")
    k = str(row.get("keywords", "") or "")
    texts.append(f"{t} {k}"[:800])

print(f"Testing on {len(texts)} articles ...\n")

print("--- SDG Classifier ---")
sdg = SDGClassifier()
for i, (text, res) in enumerate(zip(texts, sdg.classify_batch(texts, batch_size=5))):
    print(f"[{i+1}] SDG={res.sdg_label}  score={res.sdg_score}  intensity={res.sdg_intensity}")
    print(f"     title: {text[:60]}...")

print("\n--- Sentiment Analyzer ---")
sent = SentimentAnalyzer()
for i, (text, res) in enumerate(zip(texts, sent.analyze_batch(texts, batch_size=5))):
    print(f"[{i+1}] {res.label}  score={res.score}")
