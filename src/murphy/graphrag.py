"""GraphRAG over the Murphy knowledge graph.

    question -> Neo4j retrieval -> retrieved context -> LLM -> grounded answer

Three retrievers, one per shape of question a traveller actually asks:

  * ``retrieve_itinerary`` -- what is my trip?            one entity and its neighbours
  * ``retrieve_cascade``   -- what else did this plane break?   multi-hop
  * ``retrieve_policy``    -- what am I owed?             full-text, narrowed by the graph

The third is the one worth understanding. Full text alone answers "what does any
airline say about hotels"; the graph supplies *which* carrier operated the flight, how
late it was, and whether the cause was within the airline's control -- and only then is
the clause the right clause. The graph narrows, the index reads.

Two rules carried over from the rest of the project, both enforced here rather than
hoped for:

  * **No Text2Cypher.** Queries are fixed, parameterised templates. Building this graph
    produced three Cypher mistakes that returned *wrong answers rather than errors*, so
    a model generating Cypher has no signal to correct against.
  * **The LLM never produces a value.** Every number in an answer is a value retrieved
    from the graph, and the evidence travels beside the answer so a reader can check.

Run:  PYTHONPATH=src .venv/bin/python -m murphy.graphrag
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

from . import kg


@dataclass
class Retrieval:
    """Rows, and the query that produced them.

    The Cypher travels with the data because provenance is the point: an answer the
    traveller cannot check is not grounded, it is merely fluent.
    """

    tool: str
    question: str
    cypher: str
    params: dict
    rows: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def empty(self) -> bool:
        return not self.rows

    def as_context(self, limit: int = 25, width: int = 76) -> str:
        """Serialise for the prompt: key=value pairs, wrapped to a readable width.

        Deliberately flat and dull. A subgraph rendered as prose invites the model to
        read relationships that are not there.

        Wrapping is not cosmetic. A row with six aggregates runs past 200 characters,
        which is unreadable on a screen and in a screenshot, and a continuation line
        is indented so it still reads as one row rather than two.
        """
        if self.empty:
            return "(no rows returned)"
        lines = []
        for r in self.rows[:limit]:
            fields = [f"{k}={v}" for k, v in r.items() if v is not None]
            line, out = "", []
            for f in fields:
                candidate = f if not line else f"{line} · {f}"
                if len(candidate) > width and line:
                    out.append(line)
                    line = f"    {f}"          # continuation, visibly indented
                else:
                    line = candidate
            out.append(line)
            lines.extend(out)
        if len(self.rows) > limit:
            lines.append(f"... {len(self.rows) - limit} more rows not shown")
        return "\n".join(lines)


def _run(tool: str, question: str, cypher: str, **params) -> Retrieval:
    rows = [dict(r) for r in kg.run(cypher, **params)]
    return Retrieval(tool=tool, question=question, cypher=cypher.strip(),
                     params=params, rows=rows)


# ------------------------------------------------------------------ basic lookup

ITINERARY = """
MATCH (t:Traveler)-[:BOOKED]->(i:Itinerary {itinerary_id: $itinerary_id})
MATCH (i)-[leg:CONTAINS]->(f:Flight)
MATCH (f)-[:DEPARTS_FROM]->(from:Airport)
MATCH (f)-[:ARRIVES_AT]->(to:Airport)
MATCH (f)-[:INSTANCE_OF]->(s:Service)
RETURN leg.leg_order        AS leg,
       f.name               AS flight,
       from.iata            AS from,
       to.iata              AS to,
       f.sched_dep_local    AS scheduled_departure,
       f.sched_arr_local    AS scheduled_arrival,
       f.outcome_arr_delay_min AS arrived_late_by_min,
       s.p50_arr_delay      AS typical_delay_for_this_service,
       s.p90_arr_delay      AS bad_day_delay_for_this_service,
       s.n_flights          AS service_sample_size,
       t.arrive_by_utc      AS traveler_deadline
ORDER BY leg
"""


def retrieve_itinerary(itinerary_id: str, question: str = "") -> Retrieval:
    """One trip, its legs, and what each leg's service normally does.

    The Service quantiles come along because they are the context that makes a delay
    mean something: 115 minutes late is a different story on a route whose 90th
    percentile is 87 than on one whose 90th percentile is 20.
    """
    return _run("itinerary", question, ITINERARY, itinerary_id=itinerary_id)


# ------------------------------------------------------------------- multi-hop

CASCADE = """
MATCH path = (origin:Flight {flight_id: $flight_id})-[:PRECEDES*1..4]->(later:Flight)
MATCH (ac:Aircraft)-[:FLIES]->(origin)
MATCH (later)-[:DEPARTS_FROM]->(from:Airport)
MATCH (later)-[:ARRIVES_AT]->(to:Airport)
RETURN ac.tail_number AS aircraft,
       length(path)   AS hops_downstream,
       later.name     AS later_flight,
       from.iata      AS from,
       to.iata        AS to,
       later.outcome_arr_delay_min           AS arrived_late_by_min,
       later.outcome_delay_carrier_min       AS blamed_on_carrier_min,
       later.outcome_delay_late_aircraft_min AS blamed_on_late_aircraft_min
ORDER BY hops_downstream
"""


def retrieve_cascade(flight_id: str, question: str = "") -> Retrieval:
    """What the same airframe flew next, up to four legs on.

    This is the multi-hop retrieval. The cause columns ride along because they are
    what makes the path an explanation rather than a coincidence: BTS attributes the
    downstream delay to "late aircraft" independently of anything computed here.
    """
    return _run("cascade", question, CASCADE, flight_id=flight_id)


# -------------------------------------------------- documents, narrowed by the graph

POLICY_CONTEXT = """
MATCH (f:Flight {flight_id: $flight_id})-[:INSTANCE_OF]->(:Service)-[:OPERATED_BY]->(a:Airline)
RETURN a.carrier AS carrier, a.name AS airline,
       f.outcome_arr_delay_min AS delay_min,
       coalesce(f.outcome_delay_carrier_min, 0) > 0 AS within_airline_control,
       f.outcome_cancelled AS cancelled
"""

POLICY_SEARCH = """
CALL db.index.fulltext.queryNodes('policy_text', $q) YIELD node, score
MATCH (a:Airline)-[:PUBLISHES]->(node)
WHERE ($carrier IS NULL OR a.carrier = $carrier)
  AND ($controllable IS NULL OR node.controllable = $controllable)
  AND ($delay_min IS NULL OR node.threshold_hours IS NULL
       OR $delay_min >= node.threshold_hours * 60)
RETURN a.name            AS airline,
       node.commitment   AS commitment,
       node.disruption_type AS applies_to,
       node.threshold_hours AS threshold_hours,
       node.text         AS clause_text,
       node.source       AS source,
       node.url          AS url,
       round(score * 100) / 100 AS match_score
ORDER BY match_score DESC
LIMIT $limit
"""


def retrieve_policy(question: str, flight_id: str | None = None,
                    limit: int = 5) -> Retrieval:
    """Published commitments matching the question, narrowed by the traveller's facts.

    Without ``flight_id`` this is plain full-text search: it answers "what does anyone
    say about hotels". With it, the graph first establishes who operated the flight,
    how late it was, and whether the cause was within the airline's control, and those
    three facts filter the clauses. That filtering step is the whole of GraphRAG --
    the graph narrows, the index reads.

    A clause with a threshold is only returned when the delay actually reaches it, so
    a 56-minute delay never surfaces a commitment that requires three hours.
    """
    carrier = controllable = delay_min = None
    context_rows: list[dict] = []
    if flight_id:
        ctx = _run("policy-context", question, POLICY_CONTEXT, flight_id=flight_id)
        if ctx.rows:
            c = ctx.rows[0]
            carrier = c["carrier"]
            controllable = c["within_airline_control"]
            delay_min = c["delay_min"]
            context_rows = ctx.rows

    found = _run("policy", question, POLICY_SEARCH, q=question, carrier=carrier,
                 controllable=controllable, delay_min=delay_min, limit=limit)
    # The narrowing facts are evidence too -- they are why these clauses and not others.
    found.params = {"question": question, "flight_id": flight_id,
                    "narrowed_by": context_rows or "nothing (unnarrowed search)"}
    return found


# ------------------------------------------------- analytics over a retrieved set

ROUTE_RISK = """
MATCH (alt:Flight)-[:DEPARTS_FROM]->(hub:Airport {iata: $hub})
MATCH (alt)-[:ARRIVES_AT]->(dest:Airport {iata: $dest})
OPTIONAL MATCH (upstream:Flight)-[:PRECEDES]->(alt)
WITH alt, upstream, upstream IS NOT NULL
          AND upstream.outcome_arr_delay_min >= $late_threshold AS compromised
RETURN count(alt)                                             AS total_options,
       count(upstream)                                        AS options_with_traceable_upstream,
       sum(CASE WHEN compromised THEN 1 ELSE 0 END)           AS options_whose_aircraft_was_already_late,
       round(avg(alt.outcome_arr_delay_min))                  AS avg_delay_across_ALL_options,
       round(avg(CASE WHEN compromised
                      THEN alt.outcome_arr_delay_min END))    AS avg_delay_among_COMPROMISED_options,
       round(avg(CASE WHEN NOT compromised
                      THEN alt.outcome_arr_delay_min END))    AS avg_delay_among_OTHER_options
"""


def retrieve_route_risk(hub: str, dest: str, late_threshold: int = 45,
                        question: str = "") -> Retrieval:
    """Aggregate over every option on one leg of a route.

    The analytics question. It walks one hop upstream from each candidate to ask where
    that aircraft is coming from, then aggregates -- so the answer is about the whole
    option set rather than any single flight. Counting is done in Cypher precisely so
    the model does not have to: it reports a number it was handed.
    """
    return _run("route-risk", question, ROUTE_RISK,
                hub=hub, dest=dest, late_threshold=late_threshold)


# ------------------------------------------------------------------ the LLM step

# The parser already uses this model, so the project depends on one provider rather
# than two. Temperature 0 because narration of retrieved rows should not vary.
LLM_MODEL = "openai/gpt-oss-120b"
GROQ_BASE = "https://api.groq.com/openai/v1"

SYSTEM = """You explain flight disruption evidence to a traveller.

You are given rows retrieved from a knowledge graph. Follow these rules exactly:

1. Use ONLY the retrieved rows. Every number you write must appear in them.
2. Never calculate, estimate, average or infer a figure that is not present.
3. If the rows do not answer the question, say "the graph does not contain this"
   and stop. Do not fill the gap from general knowledge.
4. When you cite a policy clause, quote it and give its URL.
5. Delay figures are minutes. A "carrier" delay means a cause within the airline's
   control; it does not name an incident.
6. Be brief and concrete. Three or four sentences unless the rows demand more."""


def _llm():
    from neo4j_graphrag.llm import OpenAILLM

    from . import config
    kg._trust_system_certificates()
    key = config.load_env("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing from .env -- see .env.example")
    return OpenAILLM(model_name=LLM_MODEL, api_key=key, base_url=GROQ_BASE,
                     model_params={"temperature": 0})


@dataclass
class Answer:
    """An answer and everything needed to check it."""

    question: str
    tool: str
    cypher: str
    evidence: list[dict]
    answer: str
    elapsed_s: float

    def show(self, rows: int = 8) -> None:
        print(f"\nQ: {self.question}")
        print(f"   tool: {self.tool}   evidence: {len(self.evidence)} rows"
              f"   {self.elapsed_s:.1f}s\n")
        # Reuse the wrapped serialisation rather than truncating: a row cut at 150
        # characters hides exactly the columns that carry the argument.
        shown = Retrieval(tool=self.tool, question=self.question, cypher="",
                          params={}, rows=self.evidence[:rows])
        for line in shown.as_context(limit=rows).splitlines():
            print("   " + line)
        if len(self.evidence) > rows:
            print(f"   ... {len(self.evidence) - rows} more")
        # Wrapped at a fixed width so what is on screen does not depend on how wide
        # the window happens to be -- screenshots of this need to be reproducible.
        print()
        for i, line in enumerate(textwrap.wrap(self.answer, 76)):
            print(("A: " if i == 0 else "   ") + line)
        print()


def answer(retrieval: Retrieval, question: str | None = None, llm=None) -> Answer:
    """Turn retrieved rows into a grounded answer.

    The rows go into the prompt verbatim and come back out in the Answer, so the
    evidence sits beside the claim rather than behind it. Nothing here checks that the
    model obeyed -- that is a hand check, and the evaluation says so.
    """
    import time

    q = question or retrieval.question
    llm = llm or _llm()
    prompt = (f"Question: {q}\n\n"
              f"Retrieved from the knowledge graph "
              f"({len(retrieval)} rows, via the '{retrieval.tool}' retriever):\n"
              f"{retrieval.as_context()}\n\nAnswer:")
    started = time.time()
    reply = llm.invoke(prompt, system_instruction=SYSTEM)
    return Answer(question=q, tool=retrieval.tool, cypher=retrieval.cypher,
                  evidence=retrieval.rows, answer=reply.content.strip(),
                  elapsed_s=time.time() - started)


# ------------------------------------------------------------------- the router

# Defaults let the router answer about the sample traveller without having to invent
# identifiers. A production version would resolve these from the parsed booking; here
# they stand in for that step, which Challenge 2 already built.
DEFAULT_ITINERARY = "itin-no-escape"
DEFAULT_FLIGHT = "2024-07-19_WN4854_LBB_DAL"


def _tools():
    """Wrap the retrievers as library tools so an LLM can choose between them.

    This is "agentic" only in the sense the lecture uses: the model selects a tool.
    There is no verification loop, no retry, and no abstention -- if it picks wrongly
    it answers the wrong question confidently. That limit is stated rather than hidden,
    and it is the failure the evaluation demonstrates.
    """
    from neo4j_graphrag.tool import ObjectParameter, StringParameter, Tool

    def _itinerary(itinerary_id: str = DEFAULT_ITINERARY, **_):
        return retrieve_itinerary(itinerary_id)

    def _cascade(flight_id: str = DEFAULT_FLIGHT, **_):
        return retrieve_cascade(flight_id)

    def _route_risk(hub: str = "ATL", dest: str = "EWR", **_):
        return retrieve_route_risk(hub, dest)

    def _policy(question: str = "", flight_id: str = "", **_):
        return retrieve_policy(question or "passenger rights", flight_id or None)

    return [
        Tool(name="itinerary_lookup",
             description="A traveller's booked trip: its legs, airports, scheduled and "
                         "actual times, the deadline they must meet, and how that route "
                         "normally performs. Use for questions about one specific trip.",
             execute_func=_itinerary,
             parameters=ObjectParameter(description="itinerary lookup", properties={
                 "itinerary_id": StringParameter(description="e.g. itin-no-escape")})),
        Tool(name="aircraft_cascade",
             description="Follows one aircraft forward through the flights it operated "
                         "next, up to four legs, with how much of each delay was blamed "
                         "on the late aircraft. Use when asked what a delay caused "
                         "downstream, or what else an aircraft went on to fly.",
             execute_func=_cascade,
             parameters=ObjectParameter(description="cascade lookup", properties={
                 "flight_id": StringParameter(description="e.g. 2024-07-19_WN4854_LBB_DAL")})),
        Tool(name="route_risk_summary",
             description="Counts and averages across every flight on one airport pair: "
                         "how many options existed, how many were flown by an aircraft "
                         "already running late, and the average arrival delay. Use for "
                         "how many / on average / across all questions.",
             execute_func=_route_risk,
             parameters=ObjectParameter(description="route aggregate", properties={
                 "hub": StringParameter(description="departure airport, e.g. ATL"),
                 "dest": StringParameter(description="arrival airport, e.g. EWR")})),
        Tool(name="policy_lookup",
             description="Published airline commitments and contract-of-carriage text: "
                         "meals, hotels, rebooking, compensation. Use when asked what "
                         "the traveller is owed or entitled to.",
             execute_func=_policy,
             parameters=ObjectParameter(description="policy search", properties={
                 "question": StringParameter(description="the traveller's words"),
                 "flight_id": StringParameter(description="optional, narrows to one carrier")})),
    ]


ROUTER_SYSTEM = """Choose exactly one tool to answer the traveller's question.
Prefer route_risk_summary for questions asking how many, how often, or on average.
Prefer aircraft_cascade for questions about knock-on effects of a delay.
Prefer policy_lookup for questions about entitlements, refunds, meals or hotels.
Prefer itinerary_lookup for questions about one specific booked trip."""


def route(question: str, llm=None) -> Retrieval:
    """Let the LLM pick a retriever, then run it.

    Returns the Retrieval unchanged so the caller still sees which tool ran and the
    Cypher behind it -- routing must not cost provenance.
    """
    llm = llm or _llm()
    tools = {t.get_name(): t for t in _tools()}
    listing = "\n".join(f"- {n}: {t.get_description()}" for n, t in tools.items())
    reply = llm.invoke(
        f"Tools:\n{listing}\n\nQuestion: {question}\n\n"
        "Reply with the tool name alone, nothing else.",
        system_instruction=ROUTER_SYSTEM)
    # The reply is occasionally empty, and splitlines()[0] on an empty string raises.
    # A router that crashes is worse than one that chooses badly, so every unusable
    # reply falls back to a named tool rather than taking the demo down with it.
    lines = [l for l in reply.content.strip().strip("`\"' .").splitlines() if l.strip()]
    name = lines[0].strip() if lines else ""
    if name not in tools:
        name = "itinerary_lookup"
    result = tools[name].execute(question=question)
    result.question = question
    return result


# -------------------------------------------------------------------------- main

SAMPLES = {
    "itinerary": lambda: retrieve_itinerary(
        "itin-no-escape", "What is on my trip and when do I need to arrive?"),
    "cascade": lambda: retrieve_cascade(
        "2024-07-19_WN4854_LBB_DAL",
        "This flight ran late -- what else did the same aircraft fly?"),
    "policy-narrowed": lambda: retrieve_policy(
        "can I get a hotel or a meal", flight_id="2024-07-19_DL2498_ATL_EWR"),
    "policy-open": lambda: retrieve_policy("can I get a hotel or a meal"),
}


def main():
    for name, make in SAMPLES.items():
        r = make()
        print(f"\n{'=' * 74}\n{name}  ->  {len(r)} rows")
        if r.params.get("narrowed_by"):
            print(f"narrowed by: {r.params['narrowed_by']}")
        print(r.as_context(limit=6)[:1400])


if __name__ == "__main__":
    main()
