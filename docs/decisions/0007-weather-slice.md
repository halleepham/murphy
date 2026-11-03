# 0007 — The weather slice: IEM ASOS and TAF as issued

**Date:** 3 Nov 2026
**Status:** scope accepted by the owner, 3 Nov 2026. The METAR lag
(Decision 3) is an AI recommendation, **awaiting the owner**.

Code: `scripts/fetch_iem.py`, `murphy/datalayer/weather.py`,
`asof.asos_as_of`, `asof.taf_as_of`. Tests: `tests/test_weather.py`.

## Decision 1 — Which airports: the 30 busiest

Measured on our BTS data (Jan 2023 – Aug 2026, `duplicate_flag` excluded,
26,332,839 legs, 356 origins), share of departures by origin rank:

| Airports | Share of departures |
|---|---|
| top 15 | 43.4% |
| top 30 | 64.6% |
| top 100 | 90.9% |
| all 356 | 100% |

Query: count legs by `origin`, rank by count, cumulative share.

**Options put to the owner:** top 15, top 30, top 100, all 356.
**AI recommended top 30.** It is two thirds of departures, the download is
small (measured: ATL took 29 s for ~23 MB), and the list can widen later.
Top 100 adds small fields, many of which have no TAF. **Owner: accepted.**

The 30, ranked: ATL ORD DEN DFW CLT PHX LAX SEA LGA IAH LAS DCA SFO MCO BOS
EWR JFK MSP DTW PHL SLC MIA BNA SAN BWI AUS IAD MDW TPA DAL. All are mainland
stations whose IEM ASOS ID is the IATA code and whose TAF ID is `K` + code.

A leg touching any other airport has no weather. `asos_as_of` and
`taf_as_of` raise for such an airport, so missing weather cannot be read as
calm weather. How the model handles the gap is a Step 2 decision.

## Decision 2 — Fares and DOT data move to Steps 3–4

`docs/PLAN.md` put DB1B fares and DOT policy and consumer data in Step 1.
**AI recommended** fetching each where it is first used: fares in Step 3
(ranking alternatives), policy in Step 4 (retrieval). **Owner: accepted.**
Step 1 ends with weather.

## Decision 3 — When weather becomes known

- **TAF: at its issue time, the later of two.** IEM's `valid` column and
  the timestamp at the start of `product_id` usually agree. They differ on
  84 of 374,113 products (0.02%), either way round: `product_id` later on
  69 (by 1–160 min, median 4), IEM later on 15 (3–60 min, median 32). The
  later time can never leak, so `known_at` takes it. One product (LAX,
  `202309101740`) has rows under both 17:20 and 17:40; with this rule all
  its rows share one `known_at`. Amendments are separate products with
  their own time, so a later amendment never reaches back.
- **ASOS: observation time + 10 minutes.** IEM stores no receipt time. The
  10 minutes is an **assumption, not a sourced figure.** A visibility rule is
  needed either way, and zero lag would let a model see a METAR before it
  was sent. *Awaiting the owner.*

As with flights, visibility is strict: `known_at < as_of`.

## What was built

1. **Fetch.** One CSV per station per year, 31 Dec 2022 – 31 Aug 2026,
   ASOS routine and special reports (`report_type` 3 and 4), and every TAF.
   Each file is logged in `data/raw/iem/MANIFEST.csv` with URL, SHA-256,
   size and download time. The window starts a day early so a TAF issued on
   31 Dec covers 1 Jan flights.
2. **Build.** `data/weather/asos.parquet` and `data/weather/taf.parquet`,
   each row with `known_at` and `source_file`, plus the raw METAR or TAF
   group as evidence. Nothing under `data/` is committed.
3. **Cleaning, each with its reason.**
   - IEM writes a trace of precipitation or ice as `T`. Stored as 0 with a
     `*_trace` flag. A missing reading stays NULL; it is not zero.
   - IEM labels a TAF's opening group `Observation`. It is a forecast, so it
     is renamed `initial`. `Forecast` → `FM`, `Temporary` → `TEMPO`,
     `Probability 30/40` → `PROB30/40`. Unknown labels pass through as-is.
4. **`asos_as_of(as_of, airport, lookback_hours=24)`** and
   **`taf_as_of(as_of, airport)`** (latest TAF issued before `as_of`, all
   its periods; empty if none yet).

## Facts verified against the data

Measured 3 Nov 2026 after the full download and build.

- **Download:** 300 files (150 ASOS, 150 TAF), 556 MB raw, about 15 min.
  One request failed once (AUS 2024 ASOS, "Remote end closed connection")
  and succeeded on retry.
- **ASOS:** 1,123,438 observations, 31 Dec 2022 00:08 – 31 Aug 2026 23:56
  UTC. Per airport 33,336 (PHX) to 41,063 (PHL). Precipitation missing on
  974 rows and trace on 82,530; wind speed missing on 773.
- **Gaps:** the longest gaps between observations are 13 h, 11 h and 8 h,
  all at AUS (from 8 Jan 2024, 16 Jul 2026, 10 Jan 2024), then 7 h at MIA
  (3 Jun 2025). In a gap `asos_as_of` returns old or no rows; the caller
  sees the observation times.
- **TAF:** 374,113 products, 1,830,120 period rows: 1,304,732 `FM`,
  374,184 `initial`, 98,475 `TEMPO`, 52,729 `PROB30`. Products per airport
  7,961 (BNA) to 16,952 (ATL). The longest gap between issues is 11.8 h
  (BNA).
- **Two versions of a group:** 71 products (0.02%) hold two different
  `initial` groups under one `product_id`, e.g. SLC 1 Jan 2023 23:20,
  `1SM -SN` vs `3/4SM -SN`. Both are kept and both are returned, so a
  caller sees the conflict. Nothing here picks one.
- **Parquet:** 106 MB in `data/weather/`, built in 14 s.

## Tests

`tests/test_weather.py`: the strict METAR boundary; the lookback window;
trace vs missing precipitation; latest TAF before `as_of` with the strict
boundary; **thin evidence:** no TAF issued yet returns empty, not a guess;
leakage invariance (future rows change nothing); **failures:** an airport
outside the 30 raises, and a naive `as_of` is rejected; **baseline
comparison:** a naive nearest-in-time join returns a report from 2 minutes
in the future, which `asos_as_of` hides; `known_at` is the later of two
disagreeing issue times; a real-data check (all 30 airports in both tables;
no TAF row known before either issue time; one `known_at` per product;
disagreement under 0.1%).

Run: `.venv/bin/python -m pytest -q tests/` → `33 passed` (22 earlier, 11
here), 3 Nov 2026. The 39 warnings are PyTorch deprecation warnings.

## AI-recommendation instances

| AI recommended | Owner | Why |
|---|---|---|
| Weather for the 30 busiest airports | accepted | 64.6% of departures; small download; can widen |
| Fares and DOT data move to Steps 3–4 | accepted | fetch each where it is first used |
| METAR visible 10 min after observation | awaiting | no receipt time in IEM; zero lag would leak |
| Rename IEM's TAF `Observation` group to `initial` | not asked | it is a forecast; the IEM label misleads |
| TAF known at the later of IEM time and `product_id` time | not asked | found after the build: 84 products disagree; later cannot leak |
