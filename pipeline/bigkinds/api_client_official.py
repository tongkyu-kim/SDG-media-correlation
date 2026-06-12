"""
BigKinds Official Open API client.

Apply for an API key at: bigkinds.or.kr/v4/openApi/index.do
Once approved, set BIGKINDS_API_KEY in .env and swap the import in
collect_daily.py / scheduler.py:

    from bigkinds.api_client_official import OfficialAPIClient as Client
"""

import logging
import time
from datetime import date
from typing import Iterator

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config
from bigkinds.media_codes import FILTER_CODE_ALL

logger = logging.getLogger(__name__)


class OfficialAPIClient:
    BASE_URL = "https://tools.kinds.or.kr/search/news"

    def __init__(self, api_key: str = config.BIGKINDS_API_KEY):
        if not api_key:
            raise ValueError(
                "BIGKINDS_API_KEY not set. "
                "Apply at bigkinds.or.kr/v4/openApi/index.do"
            )
        self.api_key = api_key
        self.session = requests.Session()

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _post(self, payload: dict) -> dict:
        resp = self.session.post(self.BASE_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def search_date(
        self,
        target_date: date,
        provider_codes: str = FILTER_CODE_ALL,
        page_size: int = config.PAGE_SIZE,
    ) -> Iterator[dict]:
        date_str = target_date.strftime("%Y-%m-%d")
        page = 1
        total_hits = None

        while True:
            payload = {
                "access_key": self.api_key,
                "query": "",
                "published_at": {"from": date_str, "until": date_str},
                "provider": provider_codes.split(","),
                "category": [],
                "return_from": (page - 1) * page_size,
                "return_size": page_size,
                "sort": {"date": "desc"},
            }

            data = self._post(payload)
            docs = data.get("return_object", {}).get("docs", [])

            if total_hits is None:
                total_hits = data.get("return_object", {}).get("total_hits", 0)
                logger.info("%s: %d articles (official API)", date_str, total_hits)

            if not docs:
                break

            yield from docs

            if page * page_size >= total_hits:
                break

            page += 1
            time.sleep(config.DAILY_REQUEST_DELAY)
