"""
Playwright-based BigKinds collector.

Use this when:
  - The unofficial POST API stops working after a site redesign
  - You need bulk historical downloads (>20,000 articles per day is unlikely,
    but this handles the Excel download flow if needed)

Usage:
    client = PlaywrightClient()
    client.collect_date(date(2024, 1, 1), output_dir="downloads/")
    client.close()

The downloaded Excel is then processed by processor.py.

Requires: playwright install chromium
"""

import logging
import time
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import sync_playwright, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("playwright not installed. Run: pip install playwright && playwright install chromium")

import config
from bigkinds.media_codes import ALL_CATEGORIES, FILTER_CODES_BY_CATEGORY


class PlaywrightClient:
    """
    Controls a Chromium browser to navigate BigKinds, apply filters,
    and download the Excel export for a given date.
    """

    def __init__(
        self,
        email: str = config.BIGKINDS_EMAIL,
        password: str = config.BIGKINDS_PASSWORD,
        headless: bool = True,
    ):
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Install playwright: pip install playwright && playwright install chromium")

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self._context: BrowserContext = self._browser.new_context(
            accept_downloads=True,
            locale="ko-KR",
        )
        self._page: Page = self._context.new_page()
        self._logged_in = False

        if email and password:
            self._login(email, password)

    # ── Auth ─────────────────────────────────────────────────────────────────

    def _login(self, email: str, password: str) -> None:
        page = self._page
        page.goto(f"{config.BIGKINDS_BASE_URL}/user/loginView.do", wait_until="networkidle")

        page.fill('input[name="loginId"]', email)
        page.fill('input[name="loginPwd"]', password)
        page.click('button[type="submit"], input[type="submit"]')
        page.wait_for_load_state("networkidle")

        if "loginView" in page.url:
            logger.warning("BigKinds login may have failed — still on login page")
        else:
            self._logged_in = True
            logger.info("BigKinds Playwright login successful")

    # ── Core collection ───────────────────────────────────────────────────────

    def collect_date(self, target_date: date, output_dir: str = "downloads") -> Path | None:
        """
        Navigate to news search, apply date + media category filters,
        download the Excel export. Returns the path of the downloaded file.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        page = self._page
        date_str = target_date.strftime("%Y-%m-%d")

        logger.info("Playwright: collecting %s", date_str)

        # 1. Open news analysis page
        page.goto(
            f"{config.BIGKINDS_BASE_URL}/v2/news/search.do",
            wait_until="networkidle",
            timeout=30_000,
        )

        # 2. Set date range
        self._set_date_range(date_str, date_str)

        # 3. Select all 5 media categories
        self._select_media_categories()

        # 4. Run search
        search_btn = page.query_selector('button.btn-search, button[id*="search"], input[type="submit"]')
        if search_btn:
            search_btn.click()
            page.wait_for_load_state("networkidle")

        time.sleep(2)

        # 5. Download Excel
        return self._download_excel(output_dir, date_str)

    def _set_date_range(self, start: str, end: str) -> None:
        page = self._page
        # Selector names vary between BigKinds versions; try common patterns
        for sel in ['input[name="startDate"]', '#startDate', 'input[id*="start"]']:
            el = page.query_selector(sel)
            if el:
                el.fill(start)
                break
        for sel in ['input[name="endDate"]', '#endDate', 'input[id*="end"]']:
            el = page.query_selector(sel)
            if el:
                el.fill(end)
                break

    def _select_media_categories(self) -> None:
        """Check all 5 category checkboxes."""
        page = self._page
        # Labels / values differ between BigKinds versions
        korean_labels = ["전국일간지", "경제일간지", "전문지", "방송사", "인터넷신문"]
        for label in korean_labels:
            # Try to find a checkbox or label containing the text
            el = page.query_selector(f'label:has-text("{label}"), input[value*="{label}"]')
            if el:
                el.click()
                time.sleep(0.3)

    def _download_excel(self, output_dir: str, date_label: str) -> Path | None:
        page = self._page
        download_path = Path(output_dir) / f"bigkinds_{date_label}.xlsx"

        # Trigger download — button text varies
        for selector in [
            'button:has-text("엑셀")',
            'a:has-text("엑셀")',
            'button:has-text("Excel")',
            'button:has-text("다운로드")',
        ]:
            btn = page.query_selector(selector)
            if btn:
                with page.expect_download() as dl_info:
                    btn.click()
                download = dl_info.value
                download.save_as(str(download_path))
                logger.info("Downloaded: %s", download_path)
                return download_path

        logger.warning("Could not find download button for %s", date_label)
        return None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._context.close()
        self._browser.close()
        self._pw.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
