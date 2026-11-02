"""Export everything in the Neo4j instance to a JSON file in private/.

Why: Aura Free has no backups, and the existing instance holds data from
before the rebuild. This saves it before the instance is cleared. Prints
counts only, never the contents (the old data is not a reference, 0000).

Usage: .venv/bin/python scripts/neo4j_backup.py
"""

import datetime as dt
import json
import os
import pathlib

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv(ROOT / ".env")
    uri, user, pw = (os.environ.get(k) for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"))
    if not (uri and user and pw):
        raise SystemExit("NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD are empty in .env")
    db = os.environ.get("NEO4J_DATABASE") or None

    with GraphDatabase.driver(uri, auth=(user, pw)) as driver:
        nodes, _, _ = driver.execute_query(
            "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props",
            database_=db)
        rels, _, _ = driver.execute_query(
            "MATCH (a)-[r]->(b) RETURN elementId(r) AS id, type(r) AS type, "
            "elementId(a) AS start, elementId(b) AS end, properties(r) AS props",
            database_=db)

    out = ROOT / "private" / f"neo4j-backup-{dt.date.today().isoformat()}.json"
    payload = {"exported_at": dt.datetime.now().isoformat(timespec="seconds"), "uri": uri,
               "nodes": [r.data() for r in nodes], "relationships": [r.data() for r in rels]}
    out.write_text(json.dumps(payload, default=str))
    print(f"wrote {out.relative_to(ROOT)}: {len(payload['nodes'])} nodes, "
          f"{len(payload['relationships'])} relationships")


if __name__ == "__main__":
    main()
