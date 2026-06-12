"""
Historical backfill: collect all uncollected dates in a range.

Usage:
    python backfill.py                               # 2010-01-01 → yesterday
    python backfill.py --start 2020-01-01            # custom start
    python backfill.py --start 2020-01-01 --end 2020-12-31
    python backfill.py --retry-failed                # re-run failed dates only
    python backfill.py --dry-run                     # list pending dates, no fetch
"""

import logging
import sys
import time
from datetime import date, datetime, timedelta

import click
from tqdm import tqdm

import config
from collect_daily import collect
from storage import file_store as store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--start", default=config.COLLECT_START_DATE, show_default=True,
              help="Start date YYYY-MM-DD")
@click.option("--end", default="",
              help="End date YYYY-MM-DD (default: yesterday)")
@click.option("--retry-failed", is_flag=True,
              help="Re-run dates marked as failed in the collection log")
@click.option("--dry-run", is_flag=True,
              help="List pending dates without fetching")
@click.option("--delay", default=config.DAILY_REQUEST_DELAY, show_default=True,
              type=float, help="Seconds to sleep between days")
def main(start: str, end: str, retry_failed: bool, dry_run: bool, delay: float) -> None:
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date   = datetime.strptime(end, "%Y-%m-%d").date() if end \
                 else date.today() - timedelta(days=1)

    if retry_failed:
        pending = store.get_failed_dates()
        click.echo(f"Retrying {len(pending)} failed dates")
    else:
        pending = store.get_uncollected_dates(start_date, end_date)
        click.echo(
            f"Found {len(pending)} uncollected dates "
            f"({start_date} → {end_date})"
        )

    if not pending:
        click.echo("Nothing to do.")
        return

    if dry_run:
        for d in pending:
            click.echo(f"  {d}")
        return

    errors: list[tuple[date, str]] = []
    for d in tqdm(pending, desc="Backfill", unit="day"):
        try:
            collect(d)
        except Exception as exc:
            logger.error("Failed %s: %s", d, exc)
            errors.append((d, str(exc)))
        time.sleep(delay)

    click.echo(
        f"\nDone. {len(pending) - len(errors)} succeeded, {len(errors)} failed."
    )
    if errors:
        click.echo("Failed dates (re-run with --retry-failed):")
        for d, msg in errors:
            click.echo(f"  {d}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
