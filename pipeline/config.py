import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── BigKinds credentials ──────────────────────────────────────────────────────
BIGKINDS_EMAIL    = os.getenv("BIGKINDS_EMAIL", "")
BIGKINDS_PASSWORD = os.getenv("BIGKINDS_PASSWORD", "")

# ── BigKinds endpoints ────────────────────────────────────────────────────────
BIGKINDS_BASE_URL    = "https://www.bigkinds.or.kr"
BIGKINDS_SEARCH_URL  = f"{BIGKINDS_BASE_URL}/news/newsResult.do"
BIGKINDS_OPENAPI_URL = "https://tools.kinds.or.kr/search/news"   # official (needs key)

# ── Output path: src/news/ relative to the project root ──────────────────────
# pipeline/ sits one level below the project root, so ../src/news resolves to
# <project_root>/src/news/
_PIPELINE_DIR = Path(__file__).parent
NEWS_DATA_DIR = Path(
    os.getenv("NEWS_DATA_DIR", str(_PIPELINE_DIR / ".." / "src" / "news"))
).resolve()

# ── Collection settings ───────────────────────────────────────────────────────
COLLECT_START_DATE  = os.getenv("COLLECT_START_DATE", "2010-01-01")
DAILY_REQUEST_DELAY = float(os.getenv("DAILY_REQUEST_DELAY", "2.0"))
MAX_RETRIES         = int(os.getenv("MAX_RETRIES", "3"))
PAGE_SIZE           = 100
