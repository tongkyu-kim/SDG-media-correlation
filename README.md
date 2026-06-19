# Korean Media Coverage and SDG Aid Allocation: A Correlation Study

This repository supports research into whether Korean news media attention to Sustainable Development Goal (SDG) topics predicts or correlates with subsequent shifts in Korea's Official Development Assistance (ODA) funding patterns. The hypothesis is that media salience — measured as article volume by SDG theme — may function as an agenda-setting signal that influences aid allocation decisions across short, medium, and long time horizons.

## Research Questions

1. Does increased Korean media coverage of a given SDG precede a measurable increase in ODA disbursements aligned to that SDG?
2. How do lag structures (1–6 months, 6–24 months, 2–5 years) affect the strength of the media–funding correlation?
3. Are certain SDG clusters more responsive to media salience than others?

## Data

**Korean ODA** — KOICA/MOFA project-level disbursement records (`src/oda/`), sourced from Korea's ODA statistical system. Each row is a project with recipient country, sector, SDG classification, and funding amount.

**Korean news media** — Article batches manually downloaded from [BigKinds](https://www.bigkinds.or.kr), Korea's national news database (`src/news/`). Files cover 2013–2023 and are stored as monthly CSVs (`news_YYYY_MM.csv`). Cleaned/deduplicated versions are in `src/[processed-2]/`.

## Repository Structure

```
SDG-media-correlation/
│
├── src/
│   ├── oda/                            # Raw ODA data (xlsx/csv)
│   ├── news/                           # Converted BigKinds CSVs (news_YYYY_MM.csv)
│   ├── [processed-2]/                  # Deduplicated, author-stripped CSVs
│   ├── labels/                         # Manual annotation files
│   │   ├── sample_for_labeling.csv     # 1,000-article ODA annotation sample
│   │   └── sample_labeled.csv          # Completed annotations (coder fills this)
│   └── processed/
│       ├── news/                       # ML-classified articles (*_oda.csv, *_classified.csv)
│       ├── oda/                        # Preprocessed ODA tables
│       ├── media/                      # Aggregated media attention measures
│       ├── panel/                      # Final merged panel datasets
│       └── validation/                 # ML validation reports
│
├── models/                             # Trained classifier weights
│   └── oda_classifier.pkl
│
├── pipeline/                           # Python analysis pipeline
│   ├── config.py                       # Paths and settings
│   ├── bigkinds/
│   │   └── media_codes.py              # Outlet → category mapping
│   ├── processor.py                    # BigKinds xlsx normalizer
│   ├── convert_xlsx_to_csv.py          # Convert raw xlsx downloads to CSV
│   ├── sample_for_labeling.py          # Step 3a: generate ODA annotation sample
│   ├── train_oda_classifier.py         # Step 3b: train & apply ODA relevance classifier
│   ├── run_classify.py                 # Step 4+5: SDG + sentiment classification (BERT)
│   ├── preprocess_oda.py               # Step 8: ODA cleaning and SDG mapping
│   ├── aggregate_media.py              # Step 7: article counts by SDG/country/month
│   ├── build_panel.py                  # Step 9: merge media + ODA into panel dataset
│   ├── validate_ml.py                  # Step 10: accuracy, Cohen's κ, replication check
│   └── panel_prelim.py                 # Preliminary panel correlations
│
└── index.html                          # ODA exploratory dashboard (browser prototype)
```

## Pipeline

Data flows through the following stages. Steps 1–2 are complete (data already downloaded and converted).

### Step 2 — Convert xlsx to CSV (if needed)

Raw BigKinds Excel downloads go from `src/news/<year>/` into `src/news/news_YYYY_MM.csv`:

```bash
python convert_xlsx_to_csv.py
```

Deduplicated and author-stripped CSVs are in `src/[processed-2]/` and are the input for all subsequent steps.

### Step 3 — ODA relevance classification

Generate a 1,000-article sample for manual annotation:

```bash
python sample_for_labeling.py
```

Open `src/labels/sample_for_labeling.csv`, fill in `oda_relevant` (1 = ODA-related, 0 = not), save as `sample_labeled.csv`. Then train and apply the classifier:

```bash
python train_oda_classifier.py          # train on labeled data, apply to all CSVs
python train_oda_classifier.py --eval-only   # cross-validation metrics only
```

### Steps 4 + 5 — SDG classification and sentiment analysis

Runs BERT-based SDG classification and sentiment on ODA-relevant articles only:

```bash
python run_classify.py --oda-filtered   # recommended: skip non-ODA articles
python run_classify.py --keyword-only   # fast keyword fallback, no GPU needed
```

### Step 7 — Aggregate media attention measures

Collapses article-level data to monthly SDG × country measures (rolling averages, HHI, sentiment shares, etc.):

```bash
python aggregate_media.py
python aggregate_media.py --years 2013-2023
```

### Step 8 — Preprocess ODA data

Cleans the ODA xlsx, assigns SDG labels via CRS sector crosswalk, outputs three tables:

```bash
python preprocess_oda.py
```

### Step 9 — Build panel dataset

Merges media and ODA into country × SDG × month panel:

```bash
python build_panel.py                   # Country × SDG × Month (default)
python build_panel.py --option all      # all three panel variants
```

### Step 10 — Validate ML outputs

```bash
python validate_ml.py --task all
python validate_ml.py --task irc --coder1 src/labels/coder1.csv --coder2 src/labels/coder2.csv
```

## Setup

Requires Python 3.10+:

```bash
cd pipeline
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Optional `.env` file for path overrides:

| Variable | Description |
|---|---|
| `NEWS_DATA_DIR` | Path to raw news CSVs (default: `src/news/`) |
| `NEWS_CLEAN_DIR` | Path to cleaned news CSVs (default: `src/[processed-2]/`) |
| `LABELS_DIR` | Path to annotation files (default: `src/labels/`) |
| `MODELS_DIR` | Path for saved model files (default: `models/`) |

## Exploratory Dashboard

`index.html` is a browser-based prototype for exploring the ODA dataset. Open it directly in a modern browser — no server required.

## Status

News data collected (2013–2023). Manual ODA annotation in progress (1,000-article sample). SDG classification and panel construction ready to run once annotations are complete.
