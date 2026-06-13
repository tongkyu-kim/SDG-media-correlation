"""Batch size vs throughput benchmark for E5 classifier."""
import sys, time, io, torch, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import pandas as pd
from classify.keyword_classifier import KeywordClassifier
from classify.sdg_classifier import SDGClassifier

files = sorted(glob.glob("../src/news/2016/*.csv"))
df = pd.read_csv(files[0], dtype=str, encoding="utf-8-sig", low_memory=False)

kw = KeywordClassifier()
kw_out = kw.classify_dataframe(df)
mask = (kw_out["kw_sdg_hits"] >= 2) | (kw_out["policy_actor"] == 1)
cands = df[mask].head(1000)

title_col = "제목" if "제목" in df.columns else "title"
kw_col    = "키워드" if "키워드" in df.columns else "keywords"
texts = (cands[title_col].fillna("") + " " + cands[kw_col].fillna("")).tolist()
print(f"Test articles: {len(texts)}")

clf = SDGClassifier()
clf._load()

for bs in [128, 256, 512]:
    t0 = time.time()
    clf.classify_batch(texts, batch_size=bs)
    rate = len(texts) / (time.time() - t0)
    est_hrs = 5_900_000 / rate / 3600
    print(f"batch={bs:4d}: {rate:5.0f} articles/sec  (~{est_hrs:.1f}h for 5.9M)")
