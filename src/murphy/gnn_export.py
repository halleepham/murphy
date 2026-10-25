"""Export the Southwest subgraph from Neo4j as plain Python records.

Neo4j is the graph of record, so the GNN reads from it rather than from Parquet. This
module does nothing but read and shape: no features are computed here (that is
`gnn_features`) and no tensors are built (that is `gnn_model`). Keeping the boundary
sharp is what lets the feature assertions in `gnn_features` be tested without a
database.

**Scope is one carrier, WN.** Across all nine, carrier alone nearly separates the
classes -- Delta 91.9% late against Alaska 19.5% on the seed day -- so a model can
memorise the carrier and score well while learning nothing from structure, leaving no
headroom to measure what the graph adds. Decision N2 in `challenge3_log.md`.

**Outcome properties are read, but only two of them, and only as labels.** The export
carries `outcome_arr_delay_min` and `outcome_cancelled` because the label and the
exclusion rule both need them. Every other `outcome_*` property is left in the database.
`gnn_features` asserts that none of them reaches a feature column.

`Service.pct_late15` is exported for one purpose only: it *is* rung G0, the stored
baseline rate that every learned model has to beat. It is deliberately kept out of the
feature matrix, because it is computed over all of 2024 including the seed day and so
carries a trace of the label it predicts (decision N7). Used as the baseline it is
honest, because the contamination is stated; used as a feature it would be hidden.

Run:  PYTHONPATH=src .venv/bin/python -m murphy.gnn_export
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import kg

CARRIER = "WN"

# Flights of one carrier, with the three things a feature needs that live on another
# node: the carrier (to filter), and the two airport codes (which are edges, not
# properties -- decision S6). Ordered by flight_id so the export is reproducible:
# node ordering has to be stable or an index built in one run means something else in
# the next.
FLIGHTS = """
MATCH (f:Flight)-[:INSTANCE_OF]->(s:Service)-[:OPERATED_BY]->(a:Airline {carrier: $carrier})
MATCH (f)-[:DEPARTS_FROM]->(o:Airport)
MATCH (f)-[:ARRIVES_AT]->(d:Airport)
RETURN f.flight_id             AS flight_id,
       o.iata                  AS origin,
       d.iata                  AS dest,
       f.sched_dep_utc         AS sched_dep_utc,
       f.sched_arr_utc         AS sched_arr_utc,
       f.sched_dep_hour        AS sched_dep_hour,
       f.sched_duration_min    AS sched_duration_min,
       f.distance_mi           AS distance_mi,
       f.outcome_arr_delay_min AS outcome_arr_delay_min,
       f.outcome_cancelled     AS outcome_cancelled,
       s.pct_late15            AS service_pct_late15
ORDER BY f.flight_id
"""

# The rotation chain, one hop. Direction is preserved exactly as stored: `up` flew
# before `f` on the same tail. The arrival time of `up` comes along because the mask in
# `gnn_features` needs it -- a feature may only be used where the inbound leg had landed
# before this flight was due out (decision N6).
PRECEDES = """
MATCH (up:Flight)-[:PRECEDES]->(f:Flight)-[:INSTANCE_OF]->(:Service)
                                           -[:OPERATED_BY]->(:Airline {carrier: $carrier})
RETURN up.flight_id             AS up_id,
       f.flight_id              AS flight_id,
       up.sched_arr_utc         AS up_sched_arr_utc,
       up.outcome_arr_delay_min AS up_outcome_arr_delay_min
ORDER BY f.flight_id, up.flight_id
"""

# Both ends of every flight. `kind` keeps them in one query while staying separable:
# departing into a congested hour and arriving into one are different facts.
HOURS = """
MATCH (f:Flight)-[r:DEPARTS_IN|ARRIVES_IN]->(h:AirportHour)
MATCH (f)-[:INSTANCE_OF]->(:Service)-[:OPERATED_BY]->(:Airline {carrier: $carrier})
RETURN f.flight_id  AS flight_id,
       h.weather_id AS weather_id,
       type(r)      AS kind
ORDER BY f.flight_id, h.weather_id
"""

AIRPORT_HOURS = """
MATCH (f:Flight)-[:DEPARTS_IN|ARRIVES_IN]->(h:AirportHour)
MATCH (f)-[:INSTANCE_OF]->(:Service)-[:OPERATED_BY]->(:Airline {carrier: $carrier})
WITH DISTINCT h
RETURN h.weather_id         AS weather_id,
       h.iata               AS iata,
       h.n_sched_departures AS n_sched_departures,
       h.n_sched_arrivals   AS n_sched_arrivals,
       h.n_sched_movements  AS n_sched_movements,
       h.temp_c             AS temp_c,
       h.wind_ms            AS wind_ms,
       h.precip_mm          AS precip_mm,
       h.visibility_m       AS visibility_m,
       h.ceiling_m          AS ceiling_m,
       h.fog                AS fog,
       h.thunderstorm       AS thunderstorm
ORDER BY h.weather_id
"""


@dataclass
class Export:
    """Everything the feature builder needs, and nothing else."""

    carrier: str
    flights: list[dict] = field(default_factory=list)
    precedes: list[dict] = field(default_factory=list)
    hour_edges: list[dict] = field(default_factory=list)
    airport_hours: list[dict] = field(default_factory=list)

    @property
    def labelled(self) -> list[dict]:
        """Operated flights with an arrival delay on record.

        Cancelled flights are excluded: `kg_build` creates PRECEDES only where
        `cancelled = 0 AND diverted = 0`, so a cancelled flight is an isolated node the
        GNN cannot speak to, and "how late did it arrive" has no answer for a flight
        that never arrived (decision N2/§4).
        """
        return [f for f in self.flights
                if not f["outcome_cancelled"] and f["outcome_arr_delay_min"] is not None]

    def summary(self) -> dict:
        lab = self.labelled
        with_up = {e["flight_id"] for e in self.precedes}
        late = sum(1 for f in lab if f["outcome_arr_delay_min"] >= 15)
        return {
            "flights": len(self.flights),
            "labelled": len(lab),
            "late15": late,
            "late15_pct": round(100 * late / max(len(lab), 1), 1),
            "with_upstream": sum(1 for f in lab if f["flight_id"] in with_up),
            "precedes_edges": len(self.precedes),
            "hour_edges": len(self.hour_edges),
            "airport_hours": len(self.airport_hours),
            "airports": len({f["origin"] for f in self.flights}
                            | {f["dest"] for f in self.flights}),
        }


def export(carrier: str = CARRIER) -> Export:
    def rows(cypher):
        return [dict(r) for r in kg.run(cypher, carrier=carrier)]

    return Export(
        carrier=carrier,
        flights=rows(FLIGHTS),
        precedes=rows(PRECEDES),
        hour_edges=rows(HOURS),
        airport_hours=rows(AIRPORT_HOURS),
    )


def main():
    e = export()
    print(f"export: carrier {e.carrier}")
    for k, v in e.summary().items():
        print(f"  {k:<16} {v:>8,}" if isinstance(v, int) else f"  {k:<16} {v:>8}")


if __name__ == "__main__":
    main()
