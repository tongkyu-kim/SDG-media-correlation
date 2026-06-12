"""
Collect news for a single date and write to src/news/YYYY/MM/YYYY-MM-DD.json.

Usage:
    python collect_daily.py              # today
    python collect_daily.py 2024-06-01   # specific date
    python collect_daily.py --dry-run    # fetch + normalize, do not write
"""

import logging
import sys
from datetime import date, datetime

import click

from bigkinds.api_client import get_client
from processor import normalize_batch
from storage import file_store as store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def collect(target_date: date, dry_run: bool = False) -> int:
    if not dry_run:
        store.log_start(target_date)

    client = get_client()
    raw = list(client.search_date(target_date))
    logger.info("Fetched %d raw articles for %s", len(raw), target_date)

    articles = normalize_batch(raw)
    logger.info("Normalized: %d articles", len(articles))

    if dry_run:
        logger.info("[dry-run] Would write %d articles — skipping file write", len(articles))
        return len(articles)

    try:
        saved = store.save_articles(target_date, articles)
        store.log_done(target_date, saved)
        return saved
    except Exception as exc:
        store.log_failed(target_date, str(exc))
        raise


@click.command()
@click.argument("target_date", default="", metavar="[YYYY-MM-DD]")
@click.option("--dry-run", is_flag=True, help="Fetch and normalize but do not write files")
def main(target_date: str, dry_run: bool) -> None:
    if target_date:
        try:
            d = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            click.echo(f"Invalid date: {target_date}. Use YYYY-MM-DD.", err=True)
            sys.exit(1)
    else:
        d = date.today()

    count = collect(d, dry_run=dry_run)
    click.echo(f"{'[dry-run] ' if dry_run else ''}Collected {count} articles for {d}")


if __name__ == "__main__":
    main()
