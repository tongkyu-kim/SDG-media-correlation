import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_PIPELINE_DIR = Path(__file__).parent
_PROJECT_ROOT = (_PIPELINE_DIR / "..").resolve()

# ── News data directories ─────────────────────────────────────────────────────
# NEWS_DATA_DIR  : raw BigKinds CSVs (manually downloaded, converted from xlsx)
# NEWS_CLEAN_DIR : deduplicated + author-stripped CSVs (step 2 final output)
NEWS_DATA_DIR = Path(
    os.getenv("NEWS_DATA_DIR", str(_PROJECT_ROOT / "src" / "news"))
).resolve()

NEWS_CLEAN_DIR = Path(
    os.getenv("NEWS_CLEAN_DIR", str(_PROJECT_ROOT / "src" / "news_processed"))
).resolve()

# ── Pipeline output directories ───────────────────────────────────────────────
LABELS_DIR = Path(os.getenv("LABELS_DIR", str(_PROJECT_ROOT / "src" / "labels"))).resolve()
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(_PROJECT_ROOT / "models"))).resolve()

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
