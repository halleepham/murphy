"""IEM weather CSVs -> data/weather/{asos,taf}.parquet.

Scope: the 30 airports with the most departures in our BTS data (Jan 2023 -
Aug 2026, duplicates excluded); they carry 64.6% of departures. Owner chose
this. A leg at any other airport has no weather, and says so.

When each row becomes known (used by murphy.datalayer.asof):
  ASOS  observation time + METAR_LAG. IEM keeps no receipt time. 10 min
        covered 99.25% of 400 reports in an AWC receipt-time sample.
  TAF   its issue time: the later of IEM's `valid` column and the
        timestamp in product_id. They differ on 84 of 374,113 products
        (0.02%), either way round, by 1 to 160 minutes.

Every row keeps source_file (provenance) and the raw text (METAR or TAF
group), so a value can be traced to the report it came from.

Usage: python -m murphy.datalayer.weather
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "iem"
OUT = ROOT / "data" / "weather"

# Ranked by departures; FAA code = IATA code for
# all 30, and the TAF station is "K" + the code.
AIRPORTS = (
    "ATL", "ORD", "DEN", "DFW", "CLT", "PHX", "LAX", "SEA", "LGA", "IAH",
    "LAS", "DCA", "SFO", "MCO", "BOS", "EWR", "JFK", "MSP", "DTW", "PHL",
    "SLC", "MIA", "BNA", "SAN", "BWI", "AUS", "IAD", "MDW", "TPA", "DAL",
)
METAR_LAG = "INTERVAL 10 MINUTE"

ASOS_SQL = f"""
SELECT DISTINCT
  station AS airport,
  CAST(valid AS TIMESTAMP) AS obs_utc,
  tmpf, dwpf, relh, drct, sknt, gust, alti, vsby,
  -- IEM writes a trace amount as "T": stored as 0 with a trace flag.
  -- A missing reading stays NULL; it is not zero.
  CASE WHEN p01i = 'T' THEN 0 ELSE TRY_CAST(p01i AS DOUBLE) END AS p01i,
  p01i = 'T' AS p01i_trace,
  skyc1, skyc2, skyc3, skyc4, skyl1, skyl2, skyl3, skyl4,
  wxcodes, snowdepth,
  CASE WHEN ice_accretion_1hr = 'T' THEN 0
    ELSE TRY_CAST(ice_accretion_1hr AS DOUBLE) END AS ice_1hr,
  ice_accretion_1hr = 'T' AS ice_1hr_trace,
  metar,
  CAST(valid AS TIMESTAMP) + {METAR_LAG} AS known_at,
  parse_filename(filename) AS source_file
FROM read_csv($src, filename = true, union_by_name = true,
              types = {{'wxcodes': 'VARCHAR', 'metar': 'VARCHAR', 'p01i': 'VARCHAR',
                        'ice_accretion_1hr': 'VARCHAR',
                        'skyc1': 'VARCHAR', 'skyc2': 'VARCHAR',
                        'skyc3': 'VARCHAR', 'skyc4': 'VARCHAR'}})
ORDER BY airport, obs_utc
"""

TAF_SQL = """
SELECT DISTINCT
  substr(station, 2) AS airport,
  CAST(valid AS TIMESTAMP) AS issued_utc,
  product_id,
  is_amendment,
  -- IEM calls a TAF's opening group "Observation"; it is a forecast.
  CASE ftype WHEN 'Observation' THEN 'initial' WHEN 'Forecast' THEN 'FM'
    WHEN 'Temporary' THEN 'TEMPO' WHEN 'Probability 30' THEN 'PROB30'
    WHEN 'Probability 40' THEN 'PROB40' ELSE ftype END AS period_type,
  CAST(fx_valid AS TIMESTAMP) AS period_start_utc,
  CAST(fx_valid_end AS TIMESTAMP) AS period_end_utc,
  is_tempo,
  sknt, drct, gust, visibility, presentwx, skyc, skyl,
  ws_level, ws_drct, ws_sknt,
  raw,
  -- The later of IEM's issue time and the product_id time: when
  -- they disagree, the later one can never leak.
  greatest(CAST(valid AS TIMESTAMP),
           strptime(substr(product_id, 1, 12), '%Y%m%d%H%M')) AS known_at,
  parse_filename(filename) AS source_file
FROM read_csv($src, filename = true, union_by_name = true,
              types = {'raw': 'VARCHAR', 'presentwx': 'VARCHAR',
                       'skyc': 'VARCHAR', 'skyl': 'VARCHAR',
                       'fx_valid_end': 'VARCHAR'})
ORDER BY airport, issued_utc, period_start_utc
"""


def build(raw: Path = RAW, out: Path = OUT) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for kind, sql in (("asos", ASOS_SQL), ("taf", TAF_SQL)):
        dest = out / f"{kind}.parquet"
        rel = duckdb.sql(sql, params={"src": str(raw / kind / "*.csv")})
        rel.write_parquet(str(dest))
        counts[kind] = duckdb.sql(f"SELECT count(*) FROM '{dest}'").fetchall()[0][0]
    return counts


if __name__ == "__main__":
    print(build())
