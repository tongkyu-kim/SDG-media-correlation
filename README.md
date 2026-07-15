# Korean Media Coverage and SDG Aid Allocation: A Correlation Study

This repository supports research into whether Korean news media attention to Sustainable Development Goal (SDG) topics predicts or correlates with subsequent shifts in Korea's Official Development Assistance (ODA) funding patterns. The hypothesis is that media salience — measured as article volume by SDG theme — may function as an agenda-setting signal that influences aid allocation decisions across short, medium, and long time horizons.

## Research Questions

1. Does increased Korean media coverage of a given SDG precede a measurable increase in ODA disbursements aligned to that SDG?
2. How do lag structures (1–6 months, 6–24 months, 2–5 years) affect the strength of the media–funding correlation?
3. Are certain SDG clusters more responsive to media salience than others?

## Data

**Korean ODA** — KOICA/MOFA project-level disbursement records (`src/oda/`), sourced from Korea's ODA statistical system. Each row is a project with recipient country, sector, SDG classification, and funding amount.

**Korean news media** — Article batches downloaded from [BigKinds](https://www.bigkinds.or.kr), Korea's national news database (`src/news/`). Files cover 2007–2023 and are stored as monthly CSVs (`news_YYYY_MM.csv`). Cleaned/deduplicated versions (`news_processed/`, ~28GB, 204 files) are too large for git — they're kept in an S3 bucket and synced locally on demand:

```bash
aws s3 sync s3://sdg-media-pipeline/news_processed/ "src/news_processed"
```

The pipeline reads from `src/news_processed/` by default; set `NEWS_CLEAN_DIR` in `pipeline/.env` to override — see Setup below.

## Repository Structure

```
SDG-media-correlation/
│
├── src/
│   ├── oda/                            # Raw + cleaned ODA data (xlsx/csv)
│   ├── news/                           # Raw converted BigKinds CSVs
│   ├── news_processed/                 # Deduplicated, cleaned CSVs — synced from S3, not in git
│   ├── labels/                         # Manual annotation files
│   │   ├── sample_for_labeling.csv     # Enriched-sample ODA/SDG/dev-relevance annotation sheet
│   │   └── sample_labeled.csv          # Completed annotations (coder fills this)
│   └── processed/
│       ├── news/                       # ML-classified articles (*_oda.csv, *_devrel.csv, *_classified.csv)
│       ├── oda/                        # Preprocessed ODA tables
│       ├── media/                      # Aggregated media attention measures
│       ├── panel/                      # Final merged panel datasets
│       └── validation/                 # ML validation reports
│
├── models/                             # Trained classifier weights
│   ├── oda_classifier.pkl
│   └── devrel_classifier/              # Fine-tuned KLUE-RoBERTa (HuggingFace model dir)
│
├── pipeline/                           # Python analysis pipeline
│   ├── config.py                       # Paths and settings
│   ├── bigkinds/
│   │   └── media_codes.py              # Outlet → category mapping
│   ├── classify/                       # Classification stages
│   │   ├── keyword_classifier.py       # Rule-based SDG/aid-stance/crisis keyword classifier
│   │   ├── sdg_classifier.py           # Zero-shot: translate (ko→en) + E5 embedding similarity to SDG anchors
│   │   └── sentiment_analyzer.py       # Pretrained KLUE-RoBERTa news sentiment
│   ├── reference/                      # Korean country names, crisis events, political context lookups
│   ├── processor.py                    # BigKinds xlsx normalizer
│   ├── convert_xlsx_to_csv.py          # Convert raw xlsx downloads to CSV
│   ├── sample_for_labeling.py          # Step 3a: generate enriched, stratified annotation sample
│   ├── train_oda_classifier.py         # Step 3b: train & apply ODA relevance classifier (supervised)
│   ├── train_devrel_classifier.py      # Step 3c: fine-tune development-relevance classifier (supervised)
│   ├── run_classify.py                 # Step 4+5: SDG + sentiment classification (BERT)
│   ├── preprocess_oda.py               # Step 8: ODA cleaning and SDG mapping
│   ├── aggregate_media.py              # Step 7: article counts by SDG/country/month
│   ├── build_panel.py                  # Step 9: merge media + ODA into panel dataset
│   ├── validate_ml.py                  # Step 10: accuracy, Cohen's κ, replication check
│   ├── benchmark_sdg.py                # GPU/CPU throughput benchmark for the SDG classifier
│   └── panel_prelim.py                 # Preliminary panel correlations
```

## Pipeline

Data flows through the following stages. Steps 1–2 are complete (data already downloaded and converted).

### Step 2 — Convert xlsx to CSV (if needed)

Raw BigKinds Excel downloads go from `src/news/<year>/` into `src/news/news_YYYY_MM.csv`:

```bash
python convert_xlsx_to_csv.py
```

Deduplicated and author-stripped CSVs (`news_processed/`, synced from S3 — see Data above) are the input for all subsequent steps.

### Step 3 — ODA / development-relevance classification

Generate an annotation sample. This is an **enriched, stratified** sample, not a plain random draw — every article is pre-scored with the keyword classifier + country detector and bucketed into `candidate` / `borderline` / `negative` strata (default 50/30/20% quota per year) so coders see a real mix of positives, ambiguous cases, and controls instead of an overwhelmingly negative sample:

```bash
python sample_for_labeling.py                  # default n=600, overlap=150
python sample_for_labeling.py --pct-candidate 0.5 --pct-borderline 0.3 --pct-negative 0.2
```

Open `docs/sample_for_labeling.csv` (upload to Google Sheets), fill in `label_development_relevant`, `label_sdg_labels`, `label_sentiment_country`, `label_crisis_flag`, `label_crisis_type` for your assigned rows — both coders do the `overlap_row = YES` rows first and reconcile disagreements. Save as `src/labels/sample_labeled.csv`. Then train and apply the classifiers:

```bash
python train_oda_classifier.py               # TF-IDF + LogisticRegression, CPU, needs >=50 labels
python train_oda_classifier.py --eval-only    # cross-validation metrics only

python train_devrel_classifier.py             # fine-tunes klue/roberta-base, needs >=20 labels, GPU recommended
python train_devrel_classifier.py --apply-all # apply to full corpus after training
```

Check labeling quality before trusting either classifier's training data:

```bash
python validate_ml.py --task irc --coder1 src/labels/coder1.csv --coder2 src/labels/coder2.csv
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
python aggregate_media.py --years 2007-2023
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
.venv\Scripts\activate       # Windows — if PowerShell blocks the script (execution policy),
                              # just call .venv\Scripts\python.exe directly instead
pip install -r requirements.txt
```

Cleaned news data (`news_processed/`) is too large for git (~28GB) and lives in
an S3 bucket instead. Configure AWS CLI (`aws configure`) with credentials that
have S3 read access, then:

```bash
aws s3 sync s3://sdg-media-pipeline/news_processed/ "src/news_processed"
```

`.env` file for path overrides (create `pipeline/.env`):

| Variable | Description |
|---|---|
| `NEWS_DATA_DIR` | Path to raw news CSVs (default: `src/news/`) |
| `NEWS_CLEAN_DIR` | Path to cleaned news CSVs (default: `src/news_processed/`) — set this to wherever you synced `news_processed/` |
| `LABELS_DIR` | Path to annotation files (default: `src/labels/`) |
| `MODELS_DIR` | Path for saved model files (default: `models/`) |

BERT-based steps (`run_classify.py`, `train_devrel_classifier.py`) run on CPU
but are much faster with a CUDA GPU. `torch.cuda.is_available()` auto-detects
this — no separate config needed, just make sure the CUDA-enabled build of
PyTorch is installed if you have a GPU.

## Status

- **Data collection & preprocessing (Phases 1–2): done.** News 2007–2023 and ODA records collected, cleaned, deduplicated.
- **ODA data construction (Phase 5): done.** Three-tier SDG mapping applied (96.4% coverage) — see `docs/CLASSIFICATION_METHODS.md` Section 2.
- **ML variable construction (Phase 3): in progress.** Keyword classifier and zero-shot SDG classifier (translate + E5 embedding similarity) are built; sentiment analysis uses a pretrained model. The two supervised classifiers (ODA relevance, development relevance) are built but untrained — waiting on manual annotation. Current annotation sample (595 articles, enriched/stratified — see Section 4.4 of `CLASSIFICATION_METHODS.md`) is generated and ready for coding.
- **Media aggregation, panel construction (Phases 4, 6): scripts ready, not yet run** — depend on classified data from Phase 3.
- **Regression analysis, identification strategy (Phases 7–8): not yet built.**
- **Validation (Phase 9): tooling ready** (`validate_ml.py`), pending labeled data and trained models to validate.
