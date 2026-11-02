# 0003 — Forecast target, replay mode, scope, and the base stack

**Date:** 2 Nov 2026
**Status:** accepted

Options for each decision were proposed by the AI with a recommendation; the
owner chose. Alternatives are kept here because the final report must justify
each choice against them.

---

## Decision 1 — Forecast a delay distribution per leg, plus cancellation

**Chosen (B):** for each flight leg, a probability distribution over arrival
delay, plus a probability of cancellation. Itinerary-level risks (missed
connection, arriving after the user's deadline) are **computed in code** from
the per-leg distributions. Numbers never come from the LLM.

**Rejected alternatives:**
- A — the chance of a 15+ minute delay per flight. Simple, but it is the
  generic per-flight forecast Murphy exists to improve on. It cannot answer
  "will I make a 45-minute connection?"
- C — an end-to-end "will this trip succeed" classifier. Hard to explain and
  not reusable across itineraries.

**Why B:** it makes the itinerary focus real, and a probability forecast can be
scored. That is the Allan Murphy point (see `0002`). Planned evaluation:
- Brier score and reliability (calibration) diagrams.
- Baseline: **climatology**, meaning the historical rate for the same route,
  carrier and hour. This is the course's required simpler baseline, and it is
  the standard baseline in probability forecasting.

## Decision 2 — Historical replay first; a live adapter later

**Chosen:** replay mode. Pick a past moment as "now" and use only data
available before it. A live adapter comes later, as a future feature.

**Rejected:** live-only. Live APIs are paid or rate-limited, can't be checked
on the spot, and make demos fragile. Replay is free and reproducible, and every
forecast can be checked against what actually happened.

**Owner's note: intent, not yet an implementation decision.** Murphy should be
*dynamic*: it takes in information as it becomes available, so the forecast a
day out can differ from the one 2 hours out. The aim is to warn about risk
before it happens.

**Consequence the AI flagged, to be built in from the start:** every data
access takes an `as_of` timestamp and returns only what was known by then.
That is cheap now and costly to retrofit. It also gives the report a natural
evaluation: forecast skill against lead time.

## Decision 3 — Scope: US domestic, major carriers only

Exact carrier list and how regional partners are handled: decided in `0004`,
after the data-source research.

## Decision 4 — Python 3.12, from a prebuilt standalone build

**Chosen:** CPython 3.12.16 from `astral-sh/python-build-standalone` release
`20261026`, x86_64-apple-darwin, at `~/.local/opt/cpython-3.12.16`. SHA-256
verified against the release's `SHA256SUMS`
(`a962e47d…c769f3`, match). The venv was rebuilt with `--only-binary=:all:`.

**Why not stay on 3.10:** measured from PyPI metadata on 2 Nov 2026, the
newest x86_64-macOS wheels usable on cp310 were scikit-learn 1.7.2 and pandas
2.3.3, behind the current 1.9.1 and 3.1.0.

**Why 3.12 and not newer:** PyTorch's last macOS x86_64 wheel is **2.2.2**
(uploaded 27 Mar 2024), which supports CPython up to 3.12. Any graph neural
network work depends on it.

**Why not the python.org installer:** it needs admin rights. The standalone
build installs in the home folder and comes from a checksum-verified GitHub
release, which matches `0000` Decision 5.

The old 3.10 venv was moved out of the repo, not deleted, until 3.12 passed
the tests. Test result after the switch: `5 passed`.

## Decision 5 — Free models only, from any provider

**Rejected:** the Claude API. It is billed separately from Claude Pro, and the
owner will not buy API credits. The LLM, vision and embedding models can come
from any free provider or Hugging Face. Embeddings use **sentence-transformers**
(owner's choice). The specific models are decided in a later record, after
research.

## Decision 6 — No Streamlit

**The owner's call:** the app must look good, with interactive, user-facing
features, so it will not use Streamlit. Streamlit was removed from
`requirements.txt` and from the environment test.

This **deviates from the Challenge 2 assignment text**, which names Streamlit.
The report should state the deviation and the reason. Frontend framework: not
decided yet.

## Decision 7 — Every decision must be justifiable in the final report

The owner will write a report, not produced here. So every decision record
lists the alternatives and the evidence. Every stage builds in an evaluation:
a baseline comparison, an ablation where relevant, and failure cases.

## Feature noted for later

Parsing a **screenshot of an itinerary or booking confirmation** into the
same validated schema as free-text input. Privacy: screenshots contain names
and confirmation codes. They must never be committed, and the free-tier
provider's data-use terms matter.

## AI-recommendation instances

| AI recommended | Owner | Why |
|---|---|---|
| Forecast B (per-leg distribution) | accepted | itinerary-specific, scoreable |
| Replay first | modified: replay first, live adapter later | warn-before-it-happens goal |
| Python 3.12 | accepted | newer wheels, torch-compatible |
| Claude API for the LLM | rejected | API costs money; free models only |
