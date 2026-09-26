"""Fetch NOAA Integrated Surface Database observations for each airport.

The weather source the instructor's feedback names. Unlike reanalysis, these are
observations taken at the field, and they carry the three things reanalysis does
not: visibility, cloud ceiling, and present-weather codes for fog and
thunderstorm. Those are what actually trigger ground stops.

Open-Meteo's ERA5 archive was tried first and rejected: it returns visibility
empty for every hour and never emits a fog or thunderstorm code, which is three
of the four weather attributes the feedback asks for. Measured on this data, low
visibility triples the cancellation rate -- so that gap was not survivable.

Streams rather than hoards: each station-year is downloaded, parsed down to the
handful of fields we need, written out compactly, and the raw file deleted. Peak
disk use is one 8 MB file, and the finished table is a few hundred MB rather
than the ~5 GB the raw archive would occupy.

ISD packs several values into one comma-separated field, with sentinel values
for missing data -- 9999 and friends -- which is why the parsing below is
explicit rather than clever.

Run:  .venv/bin/python src/murphy/fetch_isd.py --years 2023 2024
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import duckdb

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
COORDS = ROOT / "data" / "raw" / "airport_coords.csv"
HISTORY = ROOT / "data" / "raw" / "isd-history.csv"
OUT = ROOT / "data" / "raw" / "isd"

HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
DATA_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"

MAX_STATION_KM = 15     # beyond this the observation is not this airport's weather

# Present-weather codes that matter operationally. Fog and thunderstorm are the
# two conditions reanalysis cannot see and that close airports.
FOG_CODES = {"10", "11", "12", "28", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49"}
THUNDER_CODES = {"17", "29", "91", "92", "93", "94", "95", "96", "97", "98", "99"}


def curl(url: str, dest: Path) -> bool:
    result = subprocess.run(
        ["curl", "-sS", "--fail", "--location", "--max-time", "300", "-o", str(dest), url],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def km_between(lat1, lon1, lat2, lon2) -> float:
    return math.hypot((lat1 - lat2) * 111,
                      (lon1 - lon2) * 111 * math.cos(math.radians((lat1 + lat2) / 2)))


def _scaled(field: str, index: int, missing: str, scale: float):
    """Pull one sub-field out of an ISD group, honouring its missing sentinel."""
    if not field:
        return None
    parts = field.split(",")
    if len(parts) <= index:
        return None
    raw = parts[index].strip()
    if not raw or raw.lstrip("+-") == missing.lstrip("+-"):
        return None
    try:
        return int(raw) * scale
    except ValueError:
        return None


def match_stations() -> list[tuple[str, str]]:
    """Pair each airport with the nearest station that ran for the whole period."""
    if not HISTORY.exists():
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        if not curl(HISTORY_URL, HISTORY):
            raise SystemExit("could not download the ISD station history")

    stations = []
    with HISTORY.open(encoding="latin-1") as f:
        for row in csv.DictReader(f):
            try:
                lat, lon = float(row["LAT"]), float(row["LON"])
            except (ValueError, TypeError):
                continue
            if not (row["BEGIN"] <= "20230101" and row["END"] >= "20241231"):
                continue
            stations.append((f"{row['USAF']}{row['WBAN']}", lat, lon))

    pairs, unmatched = [], []
    with COORDS.open() as f:
        for row in csv.DictReader(f):
            code = row["airport"]
            lat, lon = float(row["lat"]), float(row["lon"])
            best = min(stations, key=lambda s: km_between(lat, lon, s[1], s[2]))
            if km_between(lat, lon, best[1], best[2]) <= MAX_STATION_KM:
                pairs.append((code, best[0]))
            else:
                unmatched.append(code)
    if unmatched:
        print(f"no station within {MAX_STATION_KM} km: {', '.join(unmatched)}")
    return pairs


def parse_station_year(path: Path, airport: str) -> list[tuple]:
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            wnd, tmp, vis, cig = r.get("WND", ""), r.get("TMP", ""), r.get("VIS", ""), r.get("CIG", "")
            aa1, aw1 = r.get("AA1", ""), r.get("AW1", "")

            precip = _scaled(aa1, 1, "9999", 0.1)          # tenths of a mm
            code = aw1.split(",")[0].strip() if aw1 else ""

            rows.append((
                airport,
                r["DATE"],
                _scaled(tmp, 0, "9999", 0.1),               # °C
                _scaled(wnd, 3, "9999", 0.1),               # m/s
                _scaled(wnd, 0, "999", 1.0),                # degrees
                _scaled(vis, 0, "999999", 1.0),             # metres
                _scaled(cig, 0, "99999", 1.0),              # metres
                precip,
                1 if code in FOG_CODES else 0,
                1 if code in THUNDER_CODES else 0,
            ))
    return rows


COLUMNS = ["airport", "obs_time", "temp_c", "wind_ms", "wind_dir_deg",
           "visibility_m", "ceiling_m", "precip_mm", "fog", "thunderstorm"]


def main():
    ap = argparse.ArgumentParser(description="Fetch NOAA ISD observations per airport.")
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024])
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pairs = match_stations()
    print(f"{len(pairs)} airports matched to stations\n")

    con = duckdb.connect()
    done, failed, peak_mb = 0, [], 0.0

    for i, (airport, station) in enumerate(pairs, 1):
        dest = OUT / f"{airport}.parquet"
        if dest.exists():
            done += 1
            continue

        rows = []
        for year in a.years:
            tmp = OUT / f".{airport}_{year}.csv"
            if not curl(DATA_URL.format(year=year, station=station), tmp):
                tmp.unlink(missing_ok=True)
                continue
            peak_mb = max(peak_mb, tmp.stat().st_size / 1e6)
            try:
                rows.extend(parse_station_year(tmp, airport))
            finally:
                tmp.unlink(missing_ok=True)     # discard immediately

        if not rows:
            failed.append(airport)
            continue

        con.execute(f"CREATE OR REPLACE TABLE w ({', '.join(f'{c} VARCHAR' for c in COLUMNS)})")
        con.executemany(f"INSERT INTO w VALUES ({', '.join('?' * len(COLUMNS))})", rows)
        con.execute(f"""
            COPY (SELECT airport,
                         strptime(obs_time, '%Y-%m-%dT%H:%M:%S') AS obs_time,
                         temp_c::DOUBLE        AS temp_c,
                         wind_ms::DOUBLE       AS wind_ms,
                         wind_dir_deg::DOUBLE  AS wind_dir_deg,
                         visibility_m::DOUBLE  AS visibility_m,
                         ceiling_m::DOUBLE     AS ceiling_m,
                         precip_mm::DOUBLE     AS precip_mm,
                         fog::INT              AS fog,
                         thunderstorm::INT     AS thunderstorm
                  FROM w)
            TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        done += 1
        if i % 20 == 0:
            held = sum(p.stat().st_size for p in OUT.glob("*.parquet")) / 1e6
            print(f"  {i}/{len(pairs)}  done={done}  failed={len(failed)}  "
                  f"kept={held:,.0f} MB  peak temp file={peak_mb:.0f} MB")

    held = sum(p.stat().st_size for p in OUT.glob("*.parquet")) / 1e6
    print(f"\n{done} airports parsed, {held:,.0f} MB kept on disk")
    print(f"largest temporary file at any moment: {peak_mb:.0f} MB")
    if failed:
        print(f"failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
