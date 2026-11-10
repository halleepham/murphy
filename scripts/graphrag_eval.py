"""Evaluate GraphRAG against its ablation and two baselines.

  PYTHONPATH=. python scripts/graphrag_eval.py            the 5 hand-picked cases
  PYTHONPATH=. python scripts/graphrag_eval.py --sample   plus 27 flights picked by code

Every case asks the suggested questions four ways: graph, graph_no_comm, bm25, none.
Code checks each answer (murphy.graphrag.check); the author reads the answers.
Writes data/graphrag/eval.jsonl. LLM answers are cached, so a rerun costs no calls.
"""
import json
import sys
from murphy import graph as kg, graphrag as gr

CASES = [  # (flight, lead_h, why it is here)
    ("2025-01-10:AA3104:ATL-DFW:1410", 24, "example: TAF forecast freezing rain; cancelled"),
    ("2025-01-06:WN1385:SDF-MDW:1835", 24, "thin evidence: no weather reports at SDF"),
    ("2025-01-10:DL1725:ATL-GSP:1935", 2, "thin evidence: no alternative at 2 h"),
    ("2025-01-12:WN1383:MDW-RSW:1900", 24, "calm day, low risk"),
    ("2025-01-10:XX9999:ATL-DFW:1410", 24, "failure: flight not in the graph"),
]
OUT = kg.ROOT / "data" / "graphrag" / "eval.jsonl"
BANDS = ((0, 0.2, "low"), (0.2, 0.5, "middle"), (0.5, 1.01, "high"))  # chance of missing at the lead


def sample(session) -> list[tuple]:
    """3 replay flights per day, one per risk band. Each flight's lead (24, 6 or 2 h) and its pick
    within a band come from md5 of its ID: fixed, and unrelated to anything about the flight."""
    import hashlib
    rows = session.run("MATCH (f:Flight {in_replay: true}) RETURN f.id AS id, f.date AS date, "
                       "f.p_miss_24h AS p24, f.p_miss_6h AS p6, f.p_miss_2h AS p2").data()
    picked = {}
    for r in rows:
        h = int(hashlib.md5(r["id"].encode()).hexdigest(), 16)
        lead = kg.LEADS[h % 3]
        p = r[f"p{lead}"]
        if p is None:
            continue
        band = next(b for lo, hi, b in BANDS if lo <= p < hi)
        k = (r["date"], band)
        if k not in picked or h < picked[k][0]:
            picked[k] = (h, r["id"], lead, f"sample: {band} risk")
    return [v[1:] for k, v in sorted(picked.items())]


def squash(s):
    """Ignore spacing: the model writes 42 % and DL 414."""
    return "".join(s.split())


def gold(key, ev, lead):
    """The fact a correct answer must contain, taken from the graph evidence by code."""
    if key == "chance" and ev:
        return next(f["p_miss"] for f in ev[0]["forecasts"] if f["lead_h"] == lead)
    if key == "better" and ev:
        return ev[0]["alternative"].split(":")[1]
    return None


def gather():
    """Read all evidence from Neo4j first. The LLM calls can wait hours on Groq's daily limit,
    and Aura drops an idle connection, so nothing waits while the session is open."""
    driver, db = kg.connect()
    jobs, rows = [], []
    with driver, driver.session(database=db) as s:
        cases = CASES + (sample(s) if "--sample" in sys.argv else [])
        print(f"{len(cases)} cases", flush=True)
        for fid, lead, note in cases:
            base = kg.ask(s, "flight", fid, lead)
            if not base:  # code stops here: no LLM call for a flight it cannot find
                rows.append({"flight": fid, "lead_h": lead, "note": note, "system": "graph",
                             "answer": "No record of this flight in the graph.", "llm_called": False})
                continue
            as_of = gr.fmt(next(f["as_of"] for f in base[0]["forecasts"] if f["lead_h"] == lead))
            docs = gr.corpus(s, fid[:10])
            for key, (question, _) in gr.QUESTIONS.items():
                ev_graph = gr.evidence(s, key, fid, lead)
                systems = {"graph": ev_graph, "bm25": gr.bm25(docs, f"{question} {fid}"), "none": None}
                if key in ("why", "region"):  # the only questions with community information
                    systems["graph_no_comm"] = gr.evidence(s, key, fid, lead, communities=False)
                neighbors = sorted({e["airport"] for e in ev_graph if e.get("co_disruption_weight") is not None})
                for name, ev in systems.items():
                    jobs.append({"flight": fid, "lead_h": lead, "as_of": as_of, "note": note, "question": key,
                                 "system": name, "text": question, "evidence": ev,
                                 "gold": gold(key, ev_graph, lead), "neighbors": neighbors})
    print(f"{len(jobs)} answers to get", flush=True)
    return jobs, rows


def main():
    jobs, rows = gather()
    for j in jobs:
        fid, lead, key, name, ev, g = j["flight"], j["lead_h"], j["question"], j["system"], j["evidence"], j["gold"]
        msgs = gr.messages(j["text"], fid, lead, ev)
        if name == "graph_no_comm" and not ev:
            answer = "(no evidence without communities: not asked)"
        else:
            answer = gr.llm(msgs)
        c = gr.check(answer, msgs[-1]["content"], ev or [])
        r = {k: j[k] for k in ("flight", "lead_h", "as_of", "note", "question", "system", "gold")}
        r.update({"evidence_items": len(ev or []),
                  "gold_hit": None if g is None else squash(g) in squash(answer),
                  "hindsight_items": gr.hindsight(ev or [], j["as_of"]),
                  "neighbors_named": [a for a in j["neighbors"] if a in answer],
                  "answer": answer, "evidence": ev, **c})
        rows.append(r)
        print(f"-- {fid} {key} / {name}: items {r['evidence_items']}, hindsight {r['hindsight_items']}, "
              f"invented {c['invented']}, bad tags {c['bad_tags']}, cited {c['cited']}, "
              f"gold {g} hit {r['gold_hit']}, neighbors named {r['neighbors_named']}", flush=True)
        print("   " + answer.replace("\n", "\n   "), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nwrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
