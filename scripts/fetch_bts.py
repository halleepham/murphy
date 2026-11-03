"""Download BTS Marketing Carrier On-Time Performance monthly zips.

Source and window: docs/decisions/0004 (train 2023-2025, test 2026).
Each download is recorded in data/raw/bts/MANIFEST.csv with its URL, SHA-256,
size and download time, so every Parquet row can be traced to a source file.

Usage: python scripts/fetch_bts.py 2023-01 2026-08
"""

import csv
import hashlib
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{y}_{m}.zip"
)
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "bts"
MANIFEST = RAW / "MANIFEST.csv"
FIELDS = ["file", "url", "sha256", "bytes", "downloaded_at_utc"]


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch(y: int, m: int) -> dict | None:
    dest = RAW / f"mkt_{y}_{m:02d}.zip"
    if dest.exists():
        return None  # already downloaded; manifest row was written then
    url = URL.format(y=y, m=m)
    tmp = dest.with_suffix(".part")
    # urlretrieve has no timeout: a dropped connection hung it for 30+ min.
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
            break
        except OSError as e:  # includes timeouts and URLError
            print(f"{dest.name}: attempt {attempt} failed ({e})", flush=True)
            if attempt == 3:
                raise
    data = tmp.read_bytes()
    if not data.startswith(b"PK"):
        tmp.unlink()
        raise RuntimeError(f"{url} did not return a zip file")
    tmp.rename(dest)
    return {
        "file": dest.name,
        "url": url,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(start: str, end: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for y, m in months(start, end):
            row = fetch(y, m)
            if row:
                w.writerow(row)
                f.flush()
                print(f"{row['file']}  {row['bytes']:,} bytes  {row['sha256'][:12]}", flush=True)
            else:
                print(f"mkt_{y}_{m:02d}.zip  exists, skipped", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
