# Data

## Sources

| Source | Coverage | How it's fetched |
|---|---|---|
| [BTS Marketing Carrier On-Time Performance](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGK&QO_fu146_anzr=b0-gvzr) | Jan 2023 – Aug 2026. Flights marketed by AA, DL, UA, WN, AS, B6, F9 | `scripts/fetch_bts.py 2023-01 2026-08`. Each zip's URL and SHA-256 go in `data/raw/bts/MANIFEST.csv` |
| Iowa Environmental Mesonet (IEM) | [ASOS](https://mesonet.agron.iastate.edu/request/download.phtml) observations and [TAF](https://mesonet.agron.iastate.edu/request/taf.php) forecasts as issued, the 30 busiest airports, 31 Dec 2022 – 31 Aug 2026 | `scripts/fetch_iem.py` |
| [OurAirports](https://ourairports.com/data/) [`airports.csv`](https://davidmegginson.github.io/ourairports-data/airports.csv) | IATA ↔ ICAO codes, locations | Manual download to `data/raw/ourairports/` |
| [BTS DB1B Market](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FHK&QO_fu146_anzr=b0-gvzr) | Q2 2025 (`DB1BMarket_2025_2.zip`) | Manual download to `data/raw/db1b/` |
| eCFR: [14 CFR 250](https://www.ecfr.gov/current/title-14/part-250), [259](https://www.ecfr.gov/current/title-14/part-259), [260](https://www.ecfr.gov/current/title-14/part-260) | Oversales, customer service, refunds | Manual download, see below |
| [Delta](https://www.delta.com/us/en/legal/contract-of-carriage-dgr) and [Southwest](https://www.southwest.com/swa-resources/pdfs/corporate-commitments/contract-of-carriage.pdf) contracts of carriage | As published on 7 Nov 2026 | Manual download, see below |

The federal rules and airline contracts are saved under `private/policies/raw/`,
using the file names in `SOURCES` in `murphy/policies.py`. `private/` is
gitignored because the airline contracts are copyrighted and are not
redistributed.

## What a forecast row is

A **leg** is one scheduled flight. A fixed hash of the flight ID picks about
2 million legs from 2023–2025, and the same share from 2026. Hashing means the
same legs come back on every run. Each leg gives three forecasts, made 24, 6
and 2 hours before departure. Every input is computed **as of** the forecast
time: nothing that happened later is visible.

The forecast covers 16 outcomes: early, then 15-minute steps from on time to
180+ minutes late, plus cancelled and diverted.

## Splits

| Split | Years | Used for |
|---|---|---|
| Fit | 2023–2024 | Training |
| Validation | 2025 | Every setting, threshold and comparison |
| Test | Jan–Aug 2026 | Scored once, after everything was frozen |

## Model inputs

| Group | Inputs |
|---|---|
| A, schedule | Carrier, local hour, weekday, month, distance, duration, lead time |
| B, history | Trailing 365-day late and cancel rates by origin, destination, carrier and route |
| C, current state | Last 2 hours of departures and arrivals at both airports |
| D, aircraft | The aircraft's previous leg, hidden where a traveler could not have known it |
| E, weather | Latest observation at the origin; the TAF in force at departure and arrival |
| F, M, network | Regional state from the communities, and the GCN's predicted delay share |

## The release download

`scripts/download_data.py` fetches zips from this repository's GitHub release,
checks their SHA-256, and unpacks them into `data/`. Everything in them is
derived from public sources.

**App data** (`python scripts/download_data.py`): what the app reads.

| File | What |
|---|---|
| `data/forecast/app/replay.duckdb` | Precomputed 2025 trips, alternatives, forecasts and track record |
| `data/airports.parquet` | Airport time zones and locations |
| `data/weather/{asos,taf}.parquet` | Weather for the map and the knowledge graph |
| `data/forecast/network/` | Communities, co-disruption links, GCN scores |
| `data/pastdays/corpus.parquet` | Airport-days for similar-day search |
| `data/graphrag/llm_cache.jsonl` | Cached LLM explanations: an answer already explained needs no new LLM call |
| `data/forecast/alternatives/final_network.joblib` | The trained forecast model that was evaluated |

**Graph data** (`python scripts/download_data.py --graph`): only needed to load
the knowledge graph for the Ask tab.

| File | What |
|---|---|
| `data/flights/2023_*.parquet`, `2024_*`, `2025_01` | Flights for the graph's routes, links and January 2025 slice |
| `data/forecast/rows.parquet` | Forecast rows |
| `data/forecast/network/full/m.parquet` | The GCN inputs each forecast saw |
