"""GraphRAG over the knowledge graph.

Suggested question -> fixed Cypher queries (murphy.graph.QUERIES) -> evidence rows tagged [E1]..
-> LLM explains, citing tags -> code checks the answer. The LLM computes nothing: code formats
every number before the LLM sees it, and `check` flags any number in the answer that is not in
the evidence.

Four ways to answer, for comparison:
  graph          the fixed queries
  graph_no_comm  the same, without community and co-disruption neighbors (the ablation)
  bm25           BM25 over the same graph facts written as sentences, no time filter
  none           the LLM alone, no evidence
"""
import hashlib
import json
import os
import re
import time

import httpx
from rank_bm25 import BM25Okapi

from murphy import graph as kg

MODEL = "openai/gpt-oss-120b"  # Groq free tier; listed by Groq's /models on 7 Nov 2026
URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE = kg.ROOT / "data" / "graphrag" / "llm_cache.jsonl"

# The suggested questions shown beside a trip and the queries each one runs.
QUESTIONS = {
    "chance": ("How likely is this flight to make me late?", ["flight"]),
    "better": ("Is there a better flight?", ["alternatives"]),
    "why": ("Why is this flight at risk?", ["why", "taf"]),
    "region": ("How are the regions doing today?", ["community_day"]),
}
COMMUNITY_KEYS = {"community", "community_busiest", "co_disruption_weight", "gcn_late_share_next3h"}
GCN_TOP = 3  # neighbors kept when the GCN orders them


def gcn_rank_on() -> bool:
    """Off unless MURPHY_GCN_RANK=1, so cached answers stay valid until the `why`
    answers are rerun. Changing the evidence changes the cache key, and the app would call Groq."""
    return os.environ.get("MURPHY_GCN_RANK") == "1"

# What each evidence field means, sent with the evidence (without it the LLM read
# a fixed 2023-24 link weight as today's risk, and mixed up the trip's and the alternatives' chances).
GLOSSARY = {
    "p_miss": f"chance this flight arrives more than {kg.DEADLINE_MIN} min after its scheduled arrival, "
              "or is cancelled or diverted. This is what 'late' means here",
    "trip_p_miss": "p_miss of the traveler's own flight", "alt_p_miss": "p_miss of this alternative flight",
    "arr_ahead_min": "how many minutes earlier the alternative is scheduled to arrive than the traveler's flight",
    "as_of": "when the forecast was made (UTC)", "lead_h": "hours before departure the forecast was made",
    "dep_local": "scheduled departure, local time, HHMM",
    "community": "a group of airports whose disruptions often happened together in 2023-24",
    "co_disruption_weight": "how strongly this airport's disruptions moved with the origin's in 2023-24. "
                            "A fixed link, not today's risk; its METAR shows its weather at as_of",
    "metar": "raw weather observation; known_at says when it was available",
    "raw": "raw terminal aerodrome forecast (TAF) for the origin",
    "mean_p_miss": "average p_miss of the day's flights leaving that community, at the same lead",
    "flights_p_miss_50": "how many of those flights had p_miss of 50% or more",
    "trip_role": "whether this community holds the traveler's origin or destination airport",
    "gcn_late_share_next3h": "a model's forecast, made at as_of, of the share of this airport's departures in "
                             "the next 3 hours that leave 15+ min late or are cancelled. Used to pick which "
                             "neighbors to show",
}

SYSTEM = """You explain flight-disruption evidence to a traveler.
Rules:
- Use only the evidence given. Cite each fact with its tag, like [E2].
- Copy numbers exactly as written in the evidence. Do not compute, convert, round, add, subtract or compare numbers.
- Write times exactly as in the evidence. Do not convert time zones.
- In raw METAR or TAF text you may name the weather (snow, freezing rain); do not decode its numbers.
- A fact whose known_at is after the forecast's as_of was learned later: do not use it.
- If the evidence does not answer the question, say so plainly.
- At most 5 short sentences. Plain words."""


def fmt(v, key=""):
    """Code decides how every value is written: chances as whole percents, weights to 3 places."""
    if isinstance(v, dict):
        return {k: fmt(x, k) for k, x in v.items()}
    if isinstance(v, list):
        return [fmt(x, key) for x in v]
    if isinstance(v, float):
        return f"{round(100 * v)}%" if "p_miss" in key or key.startswith("gcn_") else round(v, 3)
    if hasattr(v, "iso_format"):  # neo4j DateTime
        return v.iso_format()[:16] + "Z"
    return v


def evidence(session, key: str, flight_id: str, lead_h: int, communities=True, max_earlier_h: float = 24) -> list[dict]:
    """Rows from the question's fixed queries, each tagged. Without communities, the `why` query
    keeps only the origin's own weather row and drops the community fields."""
    out = []
    for q in QUESTIONS[key][1]:
        if q == "community_day" and not communities:
            continue
        rows = kg.ask(session, q, flight_id, lead_h, max_earlier_h)
        if q == "why" and communities and gcn_rank_on():
            rows = gcn_neighbors(rows)
        for row in rows:
            if q == "why" and row.get("co_disruption_weight") is not None and row.get("metar_id") is None:
                continue
            if not communities:
                if row.get("co_disruption_weight") is not None:
                    continue
                row = {k: v for k, v in row.items() if k not in COMMUNITY_KEYS}
            out.append({"tag": f"E{len(out) + 1}", "query": q, **fmt(row)})
    return out


def gcn_neighbors(rows: list[dict], top: int = GCN_TOP) -> list[dict]:
    """The origin's own row, then the `top` neighbors with weather, highest GCN score first.
    Neighbors with no score follow those with one. Without the score files, rows are unchanged."""
    from murphy.forecast import rank
    if rank.gcn_scores() is None:
        return rows
    own = [r for r in rows if r.get("co_disruption_weight") is None]
    nb = [dict(r) for r in rows if r.get("co_disruption_weight") is not None and r.get("metar_id") is not None]
    for r in nb:
        r["gcn_late_share_next3h"] = rank.gcn_lookup(r["airport"], r.get("as_of"))
    key = lambda r: -r["gcn_late_share_next3h"] if r["gcn_late_share_next3h"] is not None else float("inf")
    return own + sorted(nb, key=key)[:top]  # sorted is stable: ties and unscored keep link-weight order


def neighbor_order() -> str:
    from murphy.forecast import rank
    if gcn_rank_on() and rank.gcn_scores() is not None:
        return f"GCN next-3-hour late share at as_of, top {GCN_TOP}"
    return "co-disruption link weight, all with weather"


def as_text(ev: list[dict]) -> str:
    return "\n".join(f"[{e['tag']}] " + json.dumps({k: v for k, v in e.items() if k != "tag"}) for e in ev)


def messages(question: str, flight_id: str, lead_h: int, ev: list[dict] | None) -> list[dict]:
    head = f"Trip flight: {flight_id}. Forecast lead: {lead_h} h before departure.\nQuestion: {question}"
    if ev is None:
        return [{"role": "system", "content": SYSTEM.replace("Use only the evidence given. ", "")},
                {"role": "user", "content": head}]
    body = as_text(ev) if ev else "(no evidence found)"
    used = {k for e in ev for k in e} | {k for e in ev for f in e.get("forecasts", []) for k in f}
    terms = "\n".join(f"- {k}: {v}" for k, v in GLOSSARY.items() if k in used or k == "p_miss")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{head}\n\nWhat the fields mean:\n{terms}\n\nEvidence:\n{body}"}]


class Paused(Exception):
    """Groq's free daily limit is reached; the app shows the evidence without an explanation."""


def llm(msgs: list[dict], wait=True) -> str:
    """One Groq call, temperature 0, cached on disk by the exact request. With wait=False (the app),
    a rate limit raises Paused at once instead of sleeping."""
    req = {"model": MODEL, "messages": msgs, "temperature": 0}
    h = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
    if CACHE.exists():
        for line in CACHE.open():
            r = json.loads(line)
            if r["hash"] == h:
                return r["answer"]
    from dotenv import load_dotenv
    load_dotenv(kg.ROOT / ".env")
    for attempt in range(200):
        r = httpx.post(URL, json=req, timeout=120,
                       headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
        if r.status_code != 429:
            break
        if not wait:
            raise Paused(r.headers.get("retry-after", ""))
        pause = float(r.headers.get("retry-after", 20))
        if pause > 3600:  # stop; the cache keeps what was done, so a rerun resumes
            raise SystemExit(f"Groq rate limit: retry after {pause:.0f} s. {r.text[:200]}")
        time.sleep(pause + 1)
    r.raise_for_status()
    answer = r.json()["choices"][0]["message"]["content"].strip()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a") as f:
        f.write(json.dumps({"hash": h, "model": MODEL, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "answer": answer}) + "\n")
    return answer


def answer(session, key: str, flight_id: str, lead_h: int, max_earlier_h: float) -> dict:
    """What the app shows for one suggested question: the evidence always; the explanation and its
    checks when Groq answers. Answer + Evidence + Provenance + Uncertainty."""
    question = QUESTIONS[key][0]
    out = {"question": question, "evidence": [], "explanation": None, "checks": None,
           "provenance": {"queries": QUESTIONS[key][1], "llm": MODEL, "graph": "Neo4j, flights 4-12 Jan 2025"}}
    if key == "why":
        out["provenance"]["neighbor_order"] = neighbor_order()
    if not kg.ask(session, "flight", flight_id, lead_h):
        out["status"] = "not_in_graph"
        return out
    ev = evidence(session, key, flight_id, lead_h, max_earlier_h=max_earlier_h)
    out["evidence"] = ev
    out["glossary"] = {k: v for k, v in GLOSSARY.items() if any(k in e for e in ev)}  # for the evidence table headers
    if not ev:  # nothing to explain: code says so, no LLM call
        out["status"] = "no_evidence"
        return out
    msgs = messages(question, flight_id, lead_h, ev)
    try:
        out["explanation"] = llm(msgs, wait=False)
    except (Paused, httpx.HTTPError):
        out["status"] = "paused"
        return out
    out["checks"] = check(out["explanation"], msgs[-1]["content"], ev)
    out["status"] = "answered"
    return out


# --- Human feedback: the traveler validates or corrects an answer --------------------------

FEEDBACK = kg.ROOT / "data" / "graphrag" / "feedback.jsonl"
MAX_REFINES_PER_DAY = 20  # Groq budget guard: demo answers are cached and never count against it
REFINE = """The traveler says your answer may be wrong. Their note: "{note}"
Check the note against the evidence above. If the evidence supports the traveler, correct the answer.
If it does not, say so politely and keep only what the evidence supports. Follow the same rules."""


def log_feedback(record: dict) -> None:
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK.open("a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S")} | record) + "\n")


def refines_today() -> int:
    if not FEEDBACK.exists():
        return 0
    today = time.strftime("%Y-%m-%d")
    return sum(1 for line in FEEDBACK.open() if (r := json.loads(line))["at"].startswith(today) and r.get("refined"))


def refine(session, key: str, flight_id: str, lead_h: int, max_earlier_h: float, note: str) -> dict:
    """One corrected answer: the first answer (from the cache, so free when it was shown) and the
    traveler's note go back to the LLM with the same evidence. The number check runs against the
    evidence only, so a number that appears only in the note is still flagged."""
    first = answer(session, key, flight_id, lead_h, max_earlier_h)
    out = {"question": first["question"], "note": note, "first": first.get("explanation"), "evidence": first["evidence"],
           "explanation": None, "checks": None,
           "provenance": first["provenance"]}
    if first.get("status") != "answered":
        return out | {"status": "no_first_answer"}
    if refines_today() >= MAX_REFINES_PER_DAY:
        return out | {"status": "limit"}
    msgs = messages(first["question"], flight_id, lead_h, first["evidence"])
    msgs += [{"role": "assistant", "content": first["explanation"]}, {"role": "user", "content": REFINE.format(note=note)}]
    try:
        out["explanation"] = llm(msgs, wait=False)
    except (Paused, httpx.HTTPError):
        return out | {"status": "paused"}
    out["checks"] = check(out["explanation"], msgs[1]["content"], first["evidence"])
    return out | {"status": "refined"}


NUM = re.compile(r"\d+(?:\.\d+)?")


def check(answer: str, context: str, ev: list[dict]) -> dict:
    """Mechanical checks. `context` is everything the LLM was given (question, flight id, evidence).
    invented: numbers in the answer that appear nowhere in the context.
    bad_tags: citations to evidence tags that do not exist.
    First normalized: tags written [ E1 ], 【E1】 or with a zero-width space; "5 000" read as 5000."""
    answer = re.sub(r"[【\[][\s\u200b]*(E\d+)[\s\u200b]*[】\]]", r"[\1]", answer)
    answer = re.sub(r"(\d)[\s\u202f\u00a0,](?=\d{3}\b)", r"\1", answer)
    have = {float(n) for n in NUM.findall(context)}
    invented = sorted({n for n in NUM.findall(re.sub(r"\[E\d+\]", "", answer)) if float(n) not in have})
    tags = set(re.findall(r"\[(E\d+)\]", answer))
    return {"invented": invented, "bad_tags": sorted(tags - {e["tag"] for e in ev}), "cited": len(tags)}


# --- BM25 baseline: the same graph facts as sentences, retrieved by keywords --------------------

CORPUS = {
    "flight": """MATCH (f:Flight {date: $date})-[:MARKETED_BY]->(c:Carrier)
        RETURN f.id AS id, 'Flight ' + f.id + ' from ' + f.origin + ' to ' + f.dest + ', marketed by ' + c.code
             + ', departs ' + f.crs_dep_local + ' local.' AS text, f, null AS known_at""",
    "alternative": """MATCH (a:Flight)-[r:ALTERNATIVE_FOR]->(t:Flight {date: $date})
        RETURN a.id + '>' + t.id + '@' + toString(r.lead_h) AS id, properties(r) AS rel, a.id AS alt, t.id AS trip, null AS known_at""",
    "metar": """MATCH (m:Metar)-[:REPORTED_AT]->(a:Airport) WHERE m.known_at >= datetime($date) - duration('P1D')
          AND m.known_at < datetime($date) + duration('P1D')
        RETURN m.id AS id, 'METAR ' + m.id + ' at ' + a.iata + ': ' + m.metar AS text, m.known_at AS known_at""",
    "taf": """MATCH (t:Taf)-[:REPORTED_AT]->(a:Airport) WHERE t.known_at >= datetime($date) - duration('P1D')
          AND t.known_at < datetime($date) + duration('P1D')
        RETURN t.id AS id, 'TAF ' + t.id + ' for ' + a.iata + ': ' + t.raw AS text, t.known_at AS known_at""",
    "community": """MATCH (a:Airport)-[:IN_COMMUNITY]->(c:Community)
        RETURN a.iata AS id, 'Airport ' + a.iata + ' is in community ' + toString(c.id)
             + ' (busiest: ' + reduce(s = '', x IN c.busiest[0..4] | s + x + ' ') + ').' AS text, null AS known_at""",
    "co_disrupts": """MATCH (a:Airport)-[k:CO_DISRUPTS]->(b:Airport)
        RETURN a.iata + '~' + b.iata AS id, 'Airports ' + a.iata + ' and ' + b.iata
             + ' usually go bad together (weight ' + toString(round(k.weight, 3)) + ').' AS text, null AS known_at""",
}


def flight_text(f) -> str:
    """A flight's forecasts and its hindsight outcome, as the graph stores them."""
    p = "; ".join(f"chance of missing at {l} h: {fmt(f.get(f'p_miss_{l}h'), 'p_miss')}" for l in kg.LEADS)
    out = (f" Outcome, known at {fmt(f.get('outcome_known_at'))}: cancelled {f.get('outcome_cancelled')}, "
           f"arrival delay {fmt(f.get('outcome_arr_delay_min'))} min.")
    return f" {p}." + out


def corpus(session, date: str) -> list[dict]:
    """Every fact near the flight's date, one sentence each, with when it became known."""
    docs = []
    for kind, cy in CORPUS.items():
        for r in session.run(cy, date=date).data():
            if kind == "flight":
                f = r["f"]
                text = r["text"] + flight_text(f)
                known = fmt(f.get("outcome_known_at"))
            elif kind == "alternative":
                x = r["rel"]
                text = (f"Flight {r['alt']} is an alternative for {r['trip']} at {x['lead_h']} h: arrives "
                        f"{fmt(x['arr_ahead_min'])} min earlier, chance of missing {fmt(x['p_miss'], 'p_miss')}.")
                known = None
            else:
                text, known = r["text"], fmt(r["known_at"])
            docs.append({"kind": kind, "id": r["id"], "text": text, "known_at": known})
    return docs


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def bm25(docs: list[dict], query: str, k=10) -> list[dict]:
    index = BM25Okapi([tokens(d["text"]) for d in docs])
    scores = index.get_scores(tokens(query))
    top = sorted(range(len(docs)), key=lambda i: -scores[i])[:k]
    return [{"tag": f"E{j + 1}", "kind": docs[i]["kind"], "text": docs[i]["text"],
             "known_at": docs[i]["known_at"]} for j, i in enumerate(top)]


def hindsight(ev: list[dict], as_of: str | None) -> int:
    """How many evidence items became known after the forecast's as_of, or are outcomes."""
    if as_of is None:
        return 0
    return sum(1 for e in ev if (e.get("known_at") and e["known_at"] > as_of) or "Outcome" in e.get("text", ""))
