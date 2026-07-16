"""
Split a full-corpus sampling frame CSV into one file per year, so each file
stays under Excel's ~1.05M row limit and opens fully (the combined
2007-2025 file silently truncates in Excel at row ~1,048,576, landing
partway through 2009).

Usage:
  py split_frame_by_year.py --frame src/processed/sampling_frame_full_2007_2025.csv \\
      --out-dir src/processed/sampling_frame_by_year
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option("--frame", "frame_path", required=True, type=click.Path(exists=True))
@click.option("--out-dir", required=True, type=click.Path())
@click.option("--chunksize", default=500_000, show_default=True, type=int)
def main(frame_path: str, out_dir: str, chunksize: int) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    writers: dict[str, bool] = {}  # year -> header already written?
    total = 0
    for chunk in pd.read_csv(frame_path, dtype=str, encoding="utf-8-sig", chunksize=chunksize):
        for year, group in chunk.groupby("year"):
            out_path = out / f"sampling_frame_{year}.csv"
            write_header = year not in writers
            group.to_csv(out_path, mode="a", index=False, header=write_header, encoding="utf-8-sig")
            writers[year] = True
        total += len(chunk)
        click.echo(f"  processed {total:,} rows...")

    click.echo(f"\nWrote {len(writers)} annual files to {out}:")
    for year in sorted(writers):
        p = out / f"sampling_frame_{year}.csv"
        n = sum(1 for _ in open(p, encoding="utf-8-sig")) - 1
        click.echo(f"  {year}: {n:,} rows -> {p}")


if __name__ == "__main__":
    main()
