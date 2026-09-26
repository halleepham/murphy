"""Download BTS On-Time Performance data.

The source the instructor's feedback names first: "official scheduled/actual
times, delays, cancellations, diversions; historical data since 1987".

Replaces the Kaggle-preprocessed file, which had cancellations and diversions
stripped, no tail numbers, no delay causes, and timestamps converted without
midnight rollover. Here the raw HHMM values arrive untouched and we do the
conversion ourselves, correctly.

No authentication and no forms -- BTS publishes monthly zips at a stable path.

Run:  .venv/bin/python src/murphy/fetch_bts.py --years 2023 2024
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "bts"

BASE = "https://transtats.bts.gov/PREZIP"
ONTIME = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"

# Marketing carrier lives in a separate table. Not needed for the forecaster,
# but it is what resolves "DL 3391" to the regional that actually operates it,
# so it matters once itineraries are constructed.
MARKETING = "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip"

UA = "Mozilla/5.0 (compatible; murphy-research/1.0)"


def fetch_month(template: str, year: int, month: int, dest_dir: Path) -> Path | None:
    """Download and unzip one month. Returns the CSV path, or None if skipped."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / f"{year}_{month:02d}.csv"
    if marker.exists():
        print(f"  {year}-{month:02d}  already present")
        return marker

    url = f"{BASE}/{template.format(year=year, month=month)}"
    tmp = dest_dir / f".{year}_{month:02d}.zip"

    # curl rather than urllib: Python's bundled certificate store is not always
    # wired up on macOS, and this download fails there with a verification error
    # while the system tool succeeds against the same URL.
    result = subprocess.run(
        ["curl", "-sS", "--fail", "--location", "--max-time", "900",
         "-A", UA, "-o", str(tmp), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1_000_000:
        print(f"  {year}-{month:02d}  FAILED: {result.stderr.strip() or 'short file'}")
        tmp.unlink(missing_ok=True)
        return None

    with zipfile.ZipFile(tmp) as archive:
        inner = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        with archive.open(inner) as src, marker.open("wb") as out:
            shutil.copyfileobj(src, out)
    tmp.unlink(missing_ok=True)
    print(f"  {year}-{month:02d}  {marker.stat().st_size / 1e6:>6.0f} MB")
    return marker


def main():
    ap = argparse.ArgumentParser(description="Download BTS On-Time Performance data.")
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024])
    ap.add_argument("--marketing", action="store_true",
                    help="also fetch the marketing-carrier table")
    a = ap.parse_args()

    for year in a.years:
        print(f"\nOn-Time Performance {year}")
        for month in range(1, 13):
            fetch_month(ONTIME, year, month, RAW / "ontime")

    if a.marketing:
        for year in a.years:
            print(f"\nMarketing carrier {year}")
            for month in range(1, 13):
                fetch_month(MARKETING, year, month, RAW / "marketing")

    total = sum(p.stat().st_size for p in RAW.rglob("*.csv"))
    files = len(list(RAW.rglob("*.csv")))
    print(f"\n{files} files, {total / 1e9:.1f} GB in {RAW}")


if __name__ == "__main__":
    main()
