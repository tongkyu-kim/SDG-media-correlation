# Python script to split oda_korea_dataset.csv into year-based JSON files (2010-2025)
# Usage: python3 split_oda_csv.py

import csv
import json
import os

INPUT = os.path.join(os.path.dirname(__file__), 'oda_korea_dataset.csv')
OUTDIR = os.path.join(os.path.dirname(__file__), 'processed')
YEARS = [str(y) for y in range(2010, 2026)]
FIELDS = [
    'year', 'recipient_name', 'agency_name', 'aid_t', 'usd_commitment', 'interest1', 'purpose_code', 'sdg_focus'
]

os.makedirs(OUTDIR, exist_ok=True)
year_data = {y: [] for y in YEARS}

with open(INPUT, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        y = row.get('year', '')
        if y in YEARS:
            filtered = {f: row.get(f, '') for f in FIELDS}
            year_data[y].append(filtered)

for y in YEARS:
    out_path = os.path.join(OUTDIR, f'oda_{y}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(year_data[y], f, ensure_ascii=False, indent=2)
    print(f'Wrote {len(year_data[y])} records to {out_path}')
print('Done.')
