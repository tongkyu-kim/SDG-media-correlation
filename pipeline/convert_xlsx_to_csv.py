"""
Convert all BigKinds xlsx files in src/news/ to CSV and delete the originals.

Column names are mapped from Korean to English.
Output: same directory, same filename, .csv extension, UTF-8-with-BOM encoding
        (UTF-8-sig so Excel opens correctly without needing import wizard).
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ── Column rename map ─────────────────────────────────────────────────────────
COL_MAP = {
    "뉴스 식별자":              "news_id",
    "일자":                     "pub_date",
    "언론사":                   "provider",
    "기고자":                   "reporter",
    "제목":                     "title",
    "통합 분류1":               "category1",
    "통합 분류2":               "category2",
    "통합 분류3":               "category3",
    "사건/사고 분류1":          "incident1",
    "사건/사고 분류2":          "incident2",
    "사건/사고 분류3":          "incident3",
    "인물":                     "persons",
    "위치":                     "places",
    "기관":                     "organizations",
    "키워드":                   "keywords",
    "특성추출(가중치순 상위 50개)": "top_keywords",
    "본문":                     "body",
    "URL":                      "url",
    "분석제외 여부":             "exclude_flag",
}

# ── Find files ────────────────────────────────────────────────────────────────
NEWS_DIR = Path(__file__).parent.parent / "src" / "news"
xlsx_files = sorted(NEWS_DIR.rglob("*.xlsx"))

if not xlsx_files:
    print("No xlsx files found in", NEWS_DIR)
    sys.exit(0)

print(f"Found {len(xlsx_files)} xlsx files - converting to CSV ...\n")

total_before = total_after = 0
errors = []

for xlsx_path in tqdm(xlsx_files, unit="file"):
    csv_path = xlsx_path.with_suffix(".csv")
    size_before = xlsx_path.stat().st_size

    try:
        df = pd.read_excel(xlsx_path, dtype=str, engine="openpyxl")

        # Rename columns; leave any unexpected columns unchanged
        df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})

        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        size_after = csv_path.stat().st_size
        total_before += size_before
        total_after  += size_after

        xlsx_path.unlink()

    except Exception as exc:
        errors.append((xlsx_path, exc))
        tqdm.write(f"  ERROR {xlsx_path.name}: {exc}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nDone.")
print(f"  xlsx total : {total_before / 1_048_576:.1f} MB")
print(f"  csv  total : {total_after  / 1_048_576:.1f} MB")
pct = (1 - total_after / total_before) * 100 if total_before else 0
print(f"  Reduction  : {pct:+.1f}%")

if errors:
    print(f"\n{len(errors)} file(s) failed:")
    for p, e in errors:
        print(f"  {p.name}: {e}")
    sys.exit(1)
