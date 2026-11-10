"""Export the co-disruption links (CO_DISRUPTS, 2023-24) and each region's busiest airports from
Neo4j to local files.

The map reads links from this file so the app does not need Neo4j at request time.
Prints the row count and the time taken.

  PYTHONPATH=. .venv/bin/python scripts/export_links.py
"""
import time
import pandas as pd
from murphy import graph as kg

OUT = kg.ROOT / "data" / "forecast" / "network" / "co_disrupts.parquet"
NAMES = kg.ROOT / "data" / "forecast" / "network" / "community_busiest.parquet"  # big airports by 2023-24 departures
QUERY = ("MATCH (a:Airport)-[r:CO_DISRUPTS]->(b:Airport) "
         "RETURN a.iata AS a, b.iata AS b, r.weight AS weight, r.years AS years")


def main() -> None:
    t0 = time.perf_counter()
    driver, db = kg.connect()
    with driver:
        recs, _, _ = driver.execute_query(QUERY, database_=db)
        names, _, _ = driver.execute_query("MATCH (c:Community) RETURN c.id AS community, c.busiest AS busiest", database_=db)
    df = pd.DataFrame([r.data() for r in recs], columns=["a", "b", "weight", "years"])
    df.to_parquet(OUT, index=False)
    nm = pd.DataFrame([r.data() for r in names]).sort_values("community")
    nm.to_parquet(NAMES, index=False)
    print(nm.assign(busiest=nm.busiest.map(lambda b: ", ".join(b[:4]))).to_string(index=False))
    print(f"wrote {OUT.relative_to(kg.ROOT)}: {len(df)} links, "
          f"{df[['a', 'b']].stack().nunique()} airports, {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
