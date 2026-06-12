"""
Daily collection scheduler.

Runs collect_daily() every day at 06:00 KST (21:00 UTC previous day) so that
yesterday's articles are fully available on BigKinds before collection starts.

Usage:
    python scheduler.py          # run as a daemon (blocking)
    python scheduler.py --once   # collect today and exit (for cron / systemd)

For production, prefer a system-level scheduler:
    # crontab -e
    0 21 * * * cd /path/to/pipeline && python scheduler.py --once >> logs/collect.log 2>&1
"""

import logging
from datetime import date, timedelta

import click
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from collect_daily import collect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def collect_yesterday() -> None:
    target = date.today() - timedelta(days=1)
    logger.info("Scheduled collection starting for %s", target)
    try:
        saved = collect(target)
        logger.info("Scheduled collection complete: %d articles saved for %s", saved, target)
    except Exception as exc:
        logger.error("Scheduled collection failed for %s: %s", target, exc)


@click.command()
@click.option("--once", is_flag=True, help="Run once and exit (suitable for cron/systemd)")
@click.option("--hour", default=21, show_default=True, type=int,
              help="UTC hour to run (default 21 = 06:00 KST)")
def main(once: bool, hour: int) -> None:
    if once:
        collect_yesterday()
        return

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        collect_yesterday,
        trigger=CronTrigger(hour=hour, minute=0),
        id="daily_collect",
        name="BigKinds daily collection",
        misfire_grace_time=3600,
        coalesce=True,
    )

    logger.info("Scheduler started — daily collection at %02d:00 UTC", hour)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
