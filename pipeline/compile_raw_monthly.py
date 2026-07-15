"""
Compile weekly BigKinds xlsx exports in src/raw/<year>/NewsResult_*.xlsx into
monthly CSVs matching the existing src/news/ and src/news_processed/ schema
(news_YYYY_MM.csv, columns: article_id,date,outlet,title,category,persons,
locations,organizations,keywords,top_keywords,body).

Deletes the original xlsx files once all months have been written successfully.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

RAW_DIR = Path(__file__).parent.parent / "src" / "raw"
OUT_DIRS = [
    Path(__file__).parent.parent / "src" / "news",
    Path(__file__).parent.parent / "src" / "news_processed",
]

COL_MAP = {
    "뉴스 식별자": "article_id",
    "일자": "date",
    "언론사": "outlet",
    "제목": "title",
    "통합 분류1": "category",
    "인물": "persons",
    "위치": "locations",
    "기관": "organizations",
    "키워드": "keywords",
    "특성추출(가중치순 상위 50개)": "top_keywords",
    "본문": "body",
}
FINAL_COLS = [
    "article_id", "date", "outlet", "title", "category",
    "persons", "locations", "organizations", "keywords", "top_keywords", "body",
]

xlsx_files = sorted(RAW_DIR.glob("*/NewsResult_*.xlsx"))
if not xlsx_files:
    print("No raw xlsx files found in", RAW_DIR)
    sys.exit(0)

print(f"Found {len(xlsx_files)} raw xlsx files - compiling by month ...\n")

frames_by_month: dict[str, list[pd.DataFrame]] = {}
errors = []

for path in tqdm(xlsx_files, unit="file"):
    try:
        df = pd.read_excel(path, dtype=str, engine="openpyxl")
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns=COL_MAP)

        missing = [c for c in FINAL_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"missing expected columns after rename: {missing}")

        df = df[FINAL_COLS]
        df["_ym"] = df["date"].str[:6]

        for ym, sub in df.groupby("_ym"):
            frames_by_month.setdefault(ym, []).append(sub.drop(columns="_ym"))

    except Exception as exc:
        errors.append((path, exc))
        tqdm.write(f"  ERROR {path.name}: {exc}")

if errors:
    print(f"\n{len(errors)} file(s) failed to parse - aborting before any writes/deletes.")
    for p, e in errors:
        print(f"  {p}: {e}")
    sys.exit(1)

print(f"\nWriting {len(frames_by_month)} monthly files ...")

for ym in sorted(frames_by_month):
    combined = pd.concat(frames_by_month[ym], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset="article_id", keep="first")
    combined = combined.sort_values("article_id").reset_index(drop=True)

    yyyy, mm = ym[:4], ym[4:6]
    fname = f"news_{yyyy}_{mm}.csv"

    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_dir / fname, index=False, encoding="utf-8-sig")

    dup_note = f" ({before - len(combined)} dupes dropped)" if before != len(combined) else ""
    print(f"  {fname}: {len(combined)} articles{dup_note}")

print("\nDeleting original raw xlsx files ...")
for path in xlsx_files:
    path.unlink()

print("Done.")
