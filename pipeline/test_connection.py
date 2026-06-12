"""
Smoke test: verify BigKinds connectivity, response parsing, and file write.

Usage:
    python test_connection.py
    python test_connection.py --date 2024-01-15   # specific date
    python test_connection.py --save              # also write to src/news/
    python test_connection.py --verbose           # print raw JSON of first article

Exit 0 = success, 1 = failure.
"""

import json
import logging
import sys
from datetime import date, timedelta

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--date", "target", default="", metavar="YYYY-MM-DD",
              help="Date to test (default: 3 days ago)")
@click.option("--save", is_flag=True,
              help="Write fetched articles to src/news/ (full pipeline test)")
@click.option("--verbose", "-v", is_flag=True,
              help="Print raw JSON of first article")
def main(target: str, save: bool, verbose: bool) -> None:
    from bigkinds.api_client import UnofficialClient
    from bigkinds.media_codes import FILTER_CODES_BY_CATEGORY, NATIONAL_DAILY
    from processor import normalize
    from storage import file_store as store
    import config

    d = (
        date.fromisoformat(target)
        if target
        else date.today() - timedelta(days=3)
    )

    click.echo(f"Testing BigKinds → src/news/ pipeline for {d}\n")

    # ── 1. Client init ────────────────────────────────────────────────────────
    client = UnofficialClient()
    click.echo(f"  CSRF token fetched : {bool(client._csrf)}")
    click.echo(f"  Authenticated      : {client._logged_in}")

    # ── 2. Fetch (national dailies only, cap at 5) ────────────────────────────
    articles_raw = []
    try:
        for i, art in enumerate(
            client.search_date(d, provider_codes=FILTER_CODES_BY_CATEGORY[NATIONAL_DAILY])
        ):
            articles_raw.append(art)
            if i >= 4:
                break
    except Exception as exc:
        click.echo(f"\n  FAIL — fetch error: {exc}", err=True)
        sys.exit(1)

    if not articles_raw:
        click.echo(
            "\n  WARN — 0 articles returned.\n"
            "  Possible causes:\n"
            "    • Date falls on a public holiday or weekend with no coverage\n"
            "    • The endpoint shape changed (run with --verbose to inspect)\n"
            "    • Try a recent weekday: --date YYYY-MM-DD",
            err=True,
        )
        sys.exit(1)

    click.echo(f"\n  Fetched            : {len(articles_raw)} articles (capped at 5)")
    click.echo(f"  Response keys      : {', '.join(sorted(articles_raw[0].keys()))}")

    if verbose:
        click.echo("\nFirst article (raw):")
        click.echo(json.dumps(articles_raw[0], ensure_ascii=False, indent=2, default=str))

    # ── 3. Normalize ──────────────────────────────────────────────────────────
    first = normalize(articles_raw[0])
    if not first:
        click.echo(
            "\n  WARN — normalize() returned None.\n"
            "  The field names in the response don't match processor.py's mapping.\n"
            "  Run with --verbose and update processor.py accordingly.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"\n  Normalized OK:")
    click.echo(f"    news_id        : {first['news_id']}")
    click.echo(f"    pub_date       : {first['pub_date']}")
    click.echo(f"    provider_name  : {first['provider_name']}")
    click.echo(f"    media_category : {first['media_category']}")
    click.echo(f"    title          : {first['title'][:70]}…")
    click.echo(f"    keywords       : {first['keywords'][:5]}")

    # ── 4. Optional file write ────────────────────────────────────────────────
    if save:
        from processor import normalize_batch
        all_normed = normalize_batch(articles_raw)
        out_path = store._article_path(d)
        saved = store.save_articles(d, all_normed)
        store.log_done(d, saved)
        click.echo(f"\n  Written            : {out_path} ({saved} new articles)")
    else:
        click.echo(f"\n  Output path would  : {store._article_path(d)}")
        click.echo("  (use --save to actually write)")

    click.echo(f"\n  NEWS_DATA_DIR      : {config.NEWS_DATA_DIR}")
    click.echo("\nConnection test PASSED")


if __name__ == "__main__":
    main()
