"""Quick speed benchmark for the revised SDG classifier."""
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
cands = df[mask].head(500)

title_col = "제목" if "제목" in df.columns else "title"
kw_col    = "키워드" if "키워드" in df.columns else "keywords"
texts = (cands[title_col].fillna("") + " " + cands[kw_col].fillna("")).tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
print(f"Device: {device}  ({gpu_name})")
print(f"Test articles: {len(texts)}")

clf = SDGClassifier()
clf._load()

print("Running inference ...")
t0 = time.time()
results = clf.classify_batch(texts, batch_size=256)
elapsed = time.time() - t0
rate = len(texts) / elapsed
print(f"Speed: {rate:.0f} articles/sec  ({elapsed:.1f}s for {len(texts)})")
print(f"Est. total for 5.9M candidates: {5_900_000/rate/3600:.1f} hours")
print()
for r, t in zip(results[:5], texts[:5]):
    print(f"  SDG{r.sdg_label} ({r.sdg_score:.2f}) | {t[:70]}")
