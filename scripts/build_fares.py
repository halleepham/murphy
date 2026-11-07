"""Median fares by route and airline from one quarter of the DB1B (docs/decisions/0015).

Keeps nonstop, non-bulk tickets of $10 or more. Rows are counted, not passengers.
Reads data/raw/db1b/DB1BMarket_2025_2.zip; writes data/forecast/fares.parquet (both gitignored).
Usage: PYTHONPATH=. python scripts/build_fares.py
"""
import tempfile, zipfile
import duckdb
from murphy.forecast.sample import ROOT

ZIP = ROOT / "data" / "raw" / "db1b" / "DB1BMarket_2025_2.zip"
OUT = ROOT / "data" / "forecast" / "fares.parquet"

with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(ZIP) as z:
        csv = z.extract("Origin_and_Destination_Survey_DB1BMarket_2025_2.csv", tmp)
    duckdb.execute(f"""COPY (
        WITH t AS (SELECT Origin AS origin, Dest AS dest, TkCarrier AS carrier, MktFare AS f
                   FROM read_csv('{csv}', header = true)
                   WHERE MktCoupons = 1 AND BulkFare = 0 AND MktFare >= 10)
        SELECT origin, dest, carrier, count(*) AS tickets, median(f) AS fare FROM t WHERE carrier <> '99' GROUP BY ALL
        UNION ALL
        SELECT origin, dest, NULL, count(*), median(f) FROM t GROUP BY origin, dest
    ) TO '{OUT}' (FORMAT parquet)""")
print(duckdb.sql(f"SELECT count(*) AS rows, count(carrier) AS pairs, sum(CASE WHEN carrier IS NULL THEN tickets END) AS tickets FROM '{OUT}'").fetchall())
