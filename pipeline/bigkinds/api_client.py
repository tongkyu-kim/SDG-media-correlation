"""
BigKinds unofficial POST client.

Uses the AJAX endpoint that powers the BigKinds web search UI.
No API key required; login is optional but recommended to avoid rate limiting.

If/when an official API key is obtained, see api_client_official.py.
"""

import json
import logging
import re
import time
from datetime import date
from typing import Iterator

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config
from bigkinds.media_codes import FILTER_CODE_ALL

logger = logging.getLogger(__name__)


def _date_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")


class BigKindsAPIError(Exception):
    pass


class UnofficialClient:
    """
    Wraps the BigKinds web search AJAX endpoint.

    Workflow:
      1. GET main page → collect session cookies + CSRF token
      2. POST login (optional) → authenticated session
      3. POST newsResult.do per date with pagination

    The CSRF token is a Spring Security header; BigKinds refreshes it on each
    page load so we re-fetch it if we get a 403.
    """

    MAIN_URL   = config.BIGKINDS_BASE_URL
    SEARCH_URL = config.BIGKINDS_SEARCH_URL
    LOGIN_URL  = f"{config.BIGKINDS_BASE_URL}/user/loginActor.do"

    def __init__(
        self,
        email: str = config.BIGKINDS_EMAIL,
        password: str = config.BIGKINDS_PASSWORD,
    ):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": config.BIGKINDS_BASE_URL,
            "Referer": f"{config.BIGKINDS_BASE_URL}/v2/news/search.do",
        })
        self._csrf: str = ""
        self._logged_in: bool = False

        self._init_session()
        if email and password:
            self._login(email, password)

    # ── Session init ──────────────────────────────────────────────────────────

    def _init_session(self) -> None:
        """Hit the search page to collect cookies and the CSRF token."""
        try:
            resp = self.session.get(
                f"{self.MAIN_URL}/v2/news/search.do",
                timeout=15,
            )
            resp.raise_for_status()
            self._csrf = self._extract_csrf(resp.text)
            logger.debug("Session initialised, CSRF: %s…", self._csrf[:8] if self._csrf else "none")
        except Exception as exc:
            logger.warning("Could not initialise session: %s", exc)

    @staticmethod
    def _extract_csrf(html: str) -> str:
        """Extract CSRF token from Spring Security meta tags or hidden inputs."""
        # <meta name="_csrf" content="TOKEN" />
        m = re.search(r'<meta\s+name=["\']_csrf["\']\s+content=["\'](.*?)["\']', html)
        if m:
            return m.group(1)
        # <input type="hidden" name="_csrf" value="TOKEN"/>
        m = re.search(r'name=["\']_csrf["\']\s+value=["\'](.*?)["\']', html)
        if m:
            return m.group(1)
        return ""

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _login(self, email: str, password: str) -> None:
        payload = {
            "loginId": email,
            "loginPwd": password,
            "stayLogin": "N",
            "_csrf": self._csrf,
        }
        try:
            resp = self.session.post(self.LOGIN_URL, data=payload, timeout=15)
            resp.raise_for_status()
            # Refresh CSRF after login (session may have rotated it)
            resp2 = self.session.get(f"{self.MAIN_URL}/v2/news/search.do", timeout=15)
            self._csrf = self._extract_csrf(resp2.text)
            self._logged_in = True
            logger.info("BigKinds login successful (session authenticated)")
        except Exception as exc:
            logger.warning("Login failed (%s); collecting without session", exc)

    # ── Core POST ─────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=3, max=60),
    )
    def _post(self, params: dict) -> dict:
        data = {k: str(v) for k, v in params.items()}
        data["_csrf"] = self._csrf

        resp = self.session.post(self.SEARCH_URL, data=data, timeout=30)

        # 403 usually means the CSRF token expired — refresh and retry once
        if resp.status_code == 403:
            logger.debug("403 received — refreshing CSRF token")
            self._init_session()
            data["_csrf"] = self._csrf
            resp = self.session.post(self.SEARCH_URL, data=data, timeout=30)

        resp.raise_for_status()

        try:
            return resp.json()
        except ValueError as exc:
            raise BigKindsAPIError(
                f"Non-JSON response ({resp.status_code}): {resp.text[:200]}"
            ) from exc

    # ── Response parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(data: dict) -> tuple[list, int]:
        """
        Extract (articles_list, total_count) from the response.

        BigKinds has changed its response shape across versions; we try
        several known layouts in order.
        """
        # Layout A (current as of 2024 redesign)
        if "resultList" in data:
            return data["resultList"], int(data.get("totalCount", 0))

        # Layout B (older v2 endpoint)
        inner = data.get("result", {})
        if "resultList" in inner:
            return inner["resultList"], int(inner.get("totalCount", 0))

        # Layout C (some API variants nest under "data")
        inner2 = data.get("data", {})
        if "resultList" in inner2:
            return inner2["resultList"], int(inner2.get("totalCount", 0))

        # Fallback: look for any list-valued key that looks like articles
        for key, val in data.items():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                logger.debug("Guessing article list from key %r", key)
                return val, len(val)

        return [], 0

    # ── Public API ────────────────────────────────────────────────────────────

    def search_date(
        self,
        target_date: date,
        provider_codes: str = FILTER_CODE_ALL,
        page_size: int = config.PAGE_SIZE,
    ) -> Iterator[dict]:
        """Yield all articles published on target_date, paginating automatically."""
        date_str = _date_str(target_date)
        start = 1
        total_hits: int | None = None

        while True:
            params = {
                "keyword": "",
                "startDate": date_str,
                "endDate": date_str,
                "filterProviderCode": provider_codes,
                "filterCategoryCode": "",
                "sortMethod": "date",
                "resultNumber": str(page_size),
                "startNumber": str(start),
                "keywordFilterJson": json.dumps({
                    "mainKeyword": [],
                    "subKeyword": [],
                    "mustNotKeyword": [],
                    "searchField": "all",
                }),
            }

            try:
                data = self._post(params)
            except BigKindsAPIError as exc:
                logger.error("API error for %s page starting at %d: %s", date_str, start, exc)
                break

            articles, total = self._parse_response(data)

            if total_hits is None:
                total_hits = total
                logger.info("%s: %d articles found", date_str, total_hits)

            if not articles:
                break

            yield from articles

            start += page_size
            if start > total_hits:
                break

            time.sleep(config.DAILY_REQUEST_DELAY)


def get_client() -> UnofficialClient:
    """Return a ready-to-use BigKinds client."""
    return UnofficialClient()
