# murphy

Course project for **CS 5542 — Big Data Analytics & Applications**, Fall 2026.
Solo project by Hallee Pham.

> **Status: scaffolding only.** Initialised 1 Nov 2026. There is no application
> yet, and the problem domain has not been chosen. What exists is the
> repository, the toolchain, the verification setup, and the requirements traced
> from the course assignments. See [`docs/HANDOFF.md`](docs/HANDOFF.md).

A previous version of this project is archived, read-only, at
[`halleepham/murphy-v1`](https://github.com/halleepham/murphy-v1). It is not a
reference for this rebuild.

## What the course requires

Each challenge repeats one structure, graded in three stages:

> Human Design → AI Design → Human–AI Co-Design

Accumulated hard requirements, traced to the assignment text:

| Stage | Required |
|---|---|
| Challenge 2 | A working Streamlit application over real data; structured generation; retrieval with visible evidence; a human control that changes the output; 5–10 test cases including a failure case; a comparison against a simpler baseline; deployment |
| Challenge 3 | A Neo4j knowledge graph (4–8 node types, 4–10 relationship types); Cypher including a multi-hop query; a GraphRAG pipeline answering natural-language questions with retrieved evidence; a GNN message-passing exploration |
| Challenge 4 | Community detection (e.g. Louvain/Leiden) with interpretation; one or two multimodal generation outputs; 3–5 prioritised user-facing features with a wireframe; an ablation study |

Every stage also needs a documented *AI recommendation → accept / modify /
reject → why*, and every output needs **Answer + Evidence + Provenance +
Uncertainty**.

## Running it

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env     # then fill it in
.venv/bin/python -m pytest -q
```

There is no application to launch yet.

## Repository layout

```
AGENTS.md            agent instructions — the single source of truth
CLAUDE.md            imports AGENTS.md so both tools read one file
docs/HANDOFF.md      full context: decisions, measurements, constraints
docs/decisions/      one file per design decision, with its evidence
requirements.txt     assignment-mandated dependencies only
tests/               pytest; a Stop hook runs these on every agent turn
private/             gitignored — anything not for publication
data/                gitignored until a size and licensing decision is recorded
```

## Environment notes

- **Intel Mac, x86_64.** Homebrew dropped macOS Intel support in September
  2026 and builds no bottles for this machine, so `brew install` silently
  compiles from source. Install CLI tools from upstream GitHub release assets
  instead, verified against their published checksums.
- Python 3.10.6. Every dependency must have an `x86_64` + cp310 wheel.

## Licence

Coursework. No licence granted.
