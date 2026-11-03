"""Download IEM weather for the 30 busiest airports: ASOS observations and
TAF forecasts as issued. Source: docs/decisions/0004; scope: 0007.

One CSV per station per year, in data/raw/iem/{asos,taf}/. Each download is
recorded in data/raw/iem/MANIFEST.csv with its URL, SHA-256, size and time.
Requests run one at a time, to be polite to IEM.

The window starts on 31 Dec 2022, so a TAF issued the day before the first
flight (1 Jan 2023) is available.

Usage: python scripts/fetch_iem.py            # all 30 airports, 2022-2026
       python scripts/fetch_iem.py ATL ORD     # just these
"""

import csv
import hashlib
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from murphy.datalayer.weather import AIRPORTS  # noqa: E402

RAW = ROOT / "data" / "raw" / "iem"
MANIFEST = RAW / "MANIFEST.csv"
FIELDS = ["file", "url", "sha256", "bytes", "downloaded_at_utc"]
FIRST, LAST = "2022-12-31", "2026-09-01"  # LAST is exclusive

ASOS = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={s}"
    "&data=all&sts={a}T00:00Z&ets={b}T00:00Z&tz=Etc/UTC&format=onlycomma"
    "&latlon=no&missing=empty&trace=T&direct=no&report_type=3&report_type=4"
)
TAF = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py?station=K{s}"
    "&sts={a}T00:00Z&ets={b}T00:00Z&fmt=csv"
)


def years():
    """(year, start, end) chunks covering [FIRST, LAST)."""
    for y in range(int(FIRST[:4]), int(LAST[:4]) + 1):
        a, b = max(FIRST, f"{y}-01-01"), min(LAST, f"{y + 1}-01-01")
        if a < b:
            yield y, a, b


def fetch(url: str, dest: Path, header: bytes) -> dict | None:
    if dest.exists():
        return None  # already downloaded; manifest row was written then
    tmp = dest.with_suffix(".part")
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
            break
        except OSError as e:  # includes timeouts and URLError
            print(f"{dest.name}: attempt {attempt} failed ({e})", flush=True)
            if attempt == 3:
                raise
    data = tmp.read_bytes()
    if not data.startswith(header):
        tmp.unlink()
        raise RuntimeError(f"{url} did not return the expected CSV: {data[:200]!r}")
    tmp.rename(dest)
    return {
        "file": f"{dest.parent.name}/{dest.name}",
        "url": url,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(stations: list[str]) -> None:
    for kind in ("asos", "taf"):
        (RAW / kind).mkdir(parents=True, exist_ok=True)
    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as m:
        w = csv.DictWriter(m, FIELDS)
        if new:
            w.writeheader()
        for s in stations:
            for y, a, b in years():
                for kind, tpl, header in (("asos", ASOS, b"station,valid"),
                                          ("taf", TAF, b"station,valid")):
                    row = fetch(tpl.format(s=s, a=a, b=b), RAW / kind / f"{s}_{y}.csv", header)
                    if row:
                        w.writerow(row)
                        m.flush()
                        print(f"{row['file']}: {row['bytes']:,} bytes", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(AIRPORTS))
