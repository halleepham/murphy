"""Airport -> IANA time zone table.

BTS gives local clock times with no time zone, so every UTC conversion
depends on this table. Sources (both public domain, docs/decisions/0004):

- OurAirports airports.csv: maps the IATA code BTS uses to the ICAO code.
- IEM ASOS station metadata: gives `tzname` per station. Mainland stations
  are keyed by the 3-letter FAA code; territories by ICAO (TJSJ, PHNL).

Every row records where its time zone came from (`tz_source`).
"""

import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OURAIRPORTS = RAW / "ourairports" / "airports.csv"
IEM_CACHE = RAW / "iem" / "asos_stations.json"
OUT = ROOT / "data" / "airports.parquet"

IEM = "https://mesonet.agron.iastate.edu/geojson/"
US_COUNTRIES = {"US", "PR", "VI", "GU", "AS", "MP"}

# Corrections, each with its reason. Applied after the IEM lookup; an
# override for a code OurAirports lacks adds the airport.
OVERRIDES = {
    # IEM lists TIST and TISX as Atlantic/Bermuda, which observes DST.
    # The US Virgin Islands stay on UTC-4 all year.
    "STT": ("America/St_Thomas", "override: IEM says Atlantic/Bermuda; USVI has no DST"),
    "STX": ("America/St_Thomas", "override: IEM says Atlantic/Bermuda; USVI has no DST"),
    # OurAirports lists West Palm Beach as DJT/KDJT; BTS still codes it PBI.
    # IEM has no PBI station.
    "PBI": ("America/New_York", "override: OurAirports lists it as DJT; no IEM station"),
    # No IEM station at Saipan; IEM's other Saipan-area stations use this zone.
    "SPN": ("Pacific/Saipan", "override: no IEM station; zone of IEM PGWT/PGRO"),
}


def fetch_iem_stations() -> dict[str, str]:
    """Download every US ASOS network once; return {station id: tzname}."""
    if IEM_CACHE.exists():
        return json.loads(IEM_CACHE.read_text())
    with urllib.request.urlopen(IEM + "networks.geojson", timeout=60) as r:
        nets = [f["id"] for f in json.load(r)["features"]]
    nets = [n for n in nets if re.fullmatch(r"[A-Z]{2}_ASOS", n)]
    tz = {}
    for n in nets:
        with urllib.request.urlopen(f"{IEM}network/{n}.geojson", timeout=60) as r:
            for f in json.load(r)["features"]:
                tz[f["id"]] = f["properties"]["tzname"]
    IEM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    IEM_CACHE.write_text(json.dumps(tz, sort_keys=True))
    return tz


def build() -> pd.DataFrame:
    station_tz = fetch_iem_stations()
    oa = pd.read_csv(OURAIRPORTS, keep_default_na=False)
    oa = oa[(oa.iata_code != "") & oa.iso_country.isin(US_COUNTRIES)]
    # Two OurAirports rows can share an IATA code; prefer scheduled service.
    oa = oa.sort_values("scheduled_service", ascending=False).drop_duplicates("iata_code")

    rows = []
    for a in oa.itertuples():
        tz = src = None
        # local_code is the FAA code, which IEM uses when it differs from
        # IATA (FCA is FAA "GPI").
        for sid in (a.iata_code, a.local_code, a.icao_code, a.gps_code):
            if sid and sid in station_tz:
                tz, src = station_tz[sid], f"IEM station {sid}"
                break
        if a.iata_code in OVERRIDES:
            tz, src = OVERRIDES[a.iata_code]
        rows.append({
            "airport": a.iata_code, "icao": a.icao_code or a.gps_code, "name": a.name,
            "lat": a.latitude_deg, "lon": a.longitude_deg, "tz": tz, "tz_source": src,
        })
    present = {r["airport"] for r in rows}
    for code, (tz, src) in OVERRIDES.items():
        if code not in present:
            rows.append({"airport": code, "icao": None, "name": None,
                         "lat": None, "lon": None, "tz": tz, "tz_source": src})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"{len(df)} airports, {df.tz.isna().sum()} without a time zone -> {OUT}")
