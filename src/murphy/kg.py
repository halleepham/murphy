"""Connection to the Neo4j knowledge graph.

The graph is a second index over the same facts the Parquet store holds, built for
a different question shape: DuckDB answers "what happened to flights like this one"
by aggregating millions of rows, and Neo4j answers "what is this flight connected
to" by traversing a few dozen nodes. Neither does the other's job well, so both
exist.

Credentials come from .env (see .env.example). Aura names the database and the user
after the instance id rather than 'neo4j', so nothing here assumes those defaults.
"""

import os

from .config import load_env


def _trust_system_certificates() -> None:
    """Give Python a CA bundle on machines where it has none.

    The python.org macOS installer ships without wiring up a certificate store, so
    ssl.get_default_verify_paths().cafile is None and *every* TLS verification
    fails -- including Aura's, whose certificate is perfectly valid. The symptom is
    a confusing "self signed certificate in certificate chain" error that looks
    like a server problem and is not.

    certifi is already present as a transitive dependency, so pointing OpenSSL at
    its bundle fixes this without asking anyone to run Install Certificates.command
    before the repository works. An existing SSL_CERT_FILE is left alone, because a
    caller who set one meant it.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def settings() -> dict:
    """The four values needed to reach the graph, read from the environment."""
    missing = [n for n in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
               if not load_env(n)]
    if missing:
        raise RuntimeError(
            f"Missing Neo4j credentials: {', '.join(missing)}. "
            "Copy them from the Aura console into .env -- see .env.example."
        )
    uri = load_env("NEO4J_URI")
    return {
        "uri": uri,
        "user": load_env("NEO4J_USERNAME"),
        "password": load_env("NEO4J_PASSWORD"),
        # Aura instances created recently name the database after the instance
        # rather than 'neo4j', and asking for the wrong one fails at routing with
        # a message that does not mention the database.
        "database": load_env("NEO4J_DATABASE") or "neo4j",
    }


def driver():
    """Open a driver. The caller closes it, so use it as a context manager."""
    _trust_system_certificates()
    from neo4j import GraphDatabase  # imported late: SSL_CERT_FILE must be set first

    cfg = settings()
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def run(cypher: str, **parameters) -> list:
    """Run one parameterised query and return its records.

    Parameterised, never interpolated. The schema is known and the queries are
    fixed, so there is no text-to-Cypher step here for the same reason there is no
    text-to-SQL in retrieval.py: it would add a failure mode to something that has
    none, and a malformed graph query fails silently as an empty result rather than
    loudly as an error.
    """
    cfg = settings()
    with driver() as drv:
        with drv.session(database=cfg["database"]) as session:
            return list(session.run(cypher, **parameters))


def counts() -> tuple[int, int]:
    """Nodes and relationships currently stored. Aura Free caps both."""
    record = run(
        "CALL () { MATCH (n) RETURN count(n) AS nodes } "
        "CALL () { MATCH ()-[r]->() RETURN count(r) AS rels } "
        "RETURN nodes, rels"
    )[0]
    return record["nodes"], record["rels"]


def main():
    """Smoke test: .venv/bin/python -m murphy.kg"""
    cfg = settings()
    print(f"uri      : {cfg['uri']}")
    print(f"database : {cfg['database']}")
    with driver() as drv:
        drv.verify_connectivity()
    nodes, rels = counts()
    print(f"connected: {nodes} nodes, {rels} relationships")


if __name__ == "__main__":
    main()
