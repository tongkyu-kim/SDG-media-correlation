# Korean Media Coverage and SDG Aid Allocation: A Correlation Study

This repository supports research into whether Korean news media attention to Sustainable Development Goal (SDG) topics predicts or correlates with subsequent shifts in Korea's Official Development Assistance (ODA) funding patterns. The hypothesis is that media salience — measured as article volume by SDG theme — may function as an agenda-setting signal that influences aid allocation decisions across short, medium, and long time horizons.

## Research Questions

1. Does increased Korean media coverage of a given SDG precede a measurable increase in ODA disbursements aligned to that SDG?
2. How do lag structures (1–6 months, 6–24 months, 2–5 years) affect the strength of the media–funding correlation?
3. Are certain SDG clusters more responsive to media salience than others?

## Data

**Korean ODA** — KOICA/MOFA project-level disbursement records (`src/oda/`), sourced from Korea's ODA statistical system. Each row is a project with recipient country, sector, SDG classification, and funding amount.

**Korean news media** — Weekly article batches from [BigKinds](https://www.bigkinds.or.kr), Korea's national news database (`src/news/<year>/`). Files cover 2010 onward and are organized by weekly intervals. Each batch contains article metadata including publication date, outlet, keywords, and category.

## Repository Structure

```
SDG-media-correlation/
│
├── src/
│   ├── oda/                        # Raw ODA data (csv)
│   └── news/                       # BigKinds weekly news exports
│       ├── 2010/
│       ├── 2011/
│       ├── 2012/
│       ├── 2013/
│       ├── ...
│       └── 2019
│
├── pipeline/                       # Python data pipeline
│   ├── config.py                   # Paths, credentials, settings
│   ├── bigkinds/                   # BigKinds API/scraper clients
│   │   ├── api_client.py
│   │   ├── api_client_official.py
│   │   ├── web_client.py
│   │   └── media_codes.py
│   ├── collect_daily.py            # Daily news collection
│   ├── backfill.py                 # Historical backfill
│   ├── scheduler.py                # Collection scheduler
│   ├── preprocess_oda.py           # ODA cleaning and SDG mapping
│   ├── aggregate_media.py          # Article counts by SDG/month
│   ├── run_classify.py             # SDG classification of articles
│   ├── build_panel.py              # Merge media + ODA into panel dataset
│   ├── panel_prelim.py             # Preliminary panel regressions
│   └── processor.py                # Shared processing utilities
│
├── docs/
│   └── METHODOLOGY.md              # SDG classification scheme, lag structure rationale
│
├── examples/
│   └── README.md                   # Sample data format notes
│
└── index.html                      # ODA exploratory dashboard (prototype)
```

## Pipeline

Data flows through four stages:

1. **Collection** — `collect_daily.py` / `backfill.py` pull article metadata from BigKinds for a given date range and write weekly Excel files into `src/news/<year>/`.

2. **Classification** — `run_classify.py` maps each article to one or more SDGs using keyword matching against the UN SDG keyword taxonomy. Results feed into the aggregation step.

3. **Aggregation** — `aggregate_media.py` collapses article counts to monthly SDG-level totals. `preprocess_oda.py` does the same for ODA disbursements.

4. **Panel construction** — `build_panel.py` merges the two monthly time series into a balanced panel dataset suitable for fixed-effects or distributed-lag regression.

## Setup

Requires Python 3.10+. Install dependencies and configure credentials before running:

```bash
cd pipeline
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt

# Copy and fill in credentials
cp .env.example .env
```

`.env` variables:

| Variable | Description |
|---|---|
| `BIGKINDS_EMAIL` | BigKinds account email |
| `BIGKINDS_PASSWORD` | BigKinds account password |
| `NEWS_DATA_DIR` | Override default output path (optional) |
| `COLLECT_START_DATE` | Earliest date for backfill (default: 2010-01-01) |

Run a backfill for a specific year:

```bash
python backfill.py --start 2015-01-01 --end 2015-12-31
```

Build the analysis panel:

```bash
python preprocess_oda.py
python aggregate_media.py
python build_panel.py
```

## Exploratory Dashboard

`index.html` is a browser-based prototype for exploring the ODA dataset before formal analysis. Open it directly in a modern browser — no server required.

## Methodology Notes

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the SDG classification scheme, thematic groupings, lag-structure rationale, and known limitations (Korean-language media only, single primary SDG per article, fixed lag windows).

## Status

Active data collection and classification. Panel regression analysis in progress.
