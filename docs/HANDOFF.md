# Handoff — murphy, rebuilt from zero

Written 1 Nov 2026 at the end of the session that created this repository.
Read this before doing anything else. `AGENTS.md` holds the rules; this holds
the reasoning behind them and the state of play.

---

## 1. What happened and why

The previous project was scrapped deliberately, not abandoned in failure. It is
archived read-only at `halleepham/murphy-v1` (public) and locally at
`~/murphy-v1` (10 GB). The decision was the repo owner's, on the grounds of
having a clearer idea of the technologies and pipelines the project needs.

**`murphy-v1` is not a reference.** Its design documents, decision logs,
implementations and dependency choices are explicitly out of scope for this
rebuild. The only inputs used to build this repository were:

- the eleven course assignment instruction files, implementation content only
- measurements taken on this machine during this session
- the device and subscription constraints documented below

### One thing carried forward, because it is a repo-hygiene lesson

While checking the old repo for unpushed work before archiving it, one
local-only commit surfaced: `bde9500 add docs for ai design context`. It existed
nowhere on the remote. It contained `docs/HUMAN_DESIGN.md`,
`docs/CONSTRAINTS_AND_RISKS.md`, `docs/DATA_SCHEMA.md` and sample data
including 301 data rows — and `.gitignore` line 25 was `docs/`, marked "private
planning docs, local only, never committed."

So planning docs were committed early, the directory was made private later,
and the repository was left with a commit that could not be pushed without
publishing private material to a public repo. A routine "push everything before
archiving" would have leaked it.

This repository therefore splits the two from the first commit:

- `private/` — gitignored, for anything not for publication
- `docs/` — committed, reviewable, safe to publish

and `tests/test_environment.py` asserts that both `.env` and `private/` are
actually ignored, so it is a test rather than a convention.

---

## 2. Timing — read this first

Today is **1 Nov 2026**. Assignment deadlines, from the instruction files:

| Stage | Due | State |
|---|---|---|
| Challenge 2 — all stages | Sep 14 / 17 / 21 / 28 | past |
| Challenge 3 — Human Design | Oct 5 | past |
| Challenge 3 — AI Design (Neo4j KG) | Oct 12, demo Oct 13 | past |
| Challenge 3 — Co-Design (GraphRAG) | Oct 15 | past |
| Challenge 3 — GNN extension | Oct 22 | past |
| Challenge 4 — Human Design | Oct 27 | past |
| Challenge 4 — AI Design | Oct 29 | past |
| **Challenge 4 — Human–AI Co-Design** | **Mon 2 Nov, 11:59 PM** | **tomorrow** |
| Challenge 5, Challenge 6 | not released | no instructions on disk |

**Challenge 4's Co-Design submission is due tomorrow, and its requirements are
explicitly cumulative.** The assignment says "Build upon your existing GraphRAG
project rather than starting a new system," and the tasks require an existing
knowledge graph to run community detection against, a baseline GraphRAG to
compare the community-enhanced version with, and an ablation study.

This repository cannot satisfy that. It has no graph, no data, and no chosen
domain.

**The implication:** Challenge 4 should be submitted from `murphy-v1`, which
has the GraphRAG system the assignment builds on. This rebuild starts cleanly at
Challenge 5. Do not try to reconstruct Challenge 4 here — there is not time, and
the assignment is explicit that it extends existing work.

If the repo owner has an extension or has already submitted Challenge 4, this
note is moot. It is written down because a from-scratch rebuild the day before a
cumulative deadline is worth stating out loud rather than discovering later.

---

## 3. What the assignments actually require

Synthesised from all eleven instruction files. Deliverable mechanics (slide
counts, PPT structures, report formats) are deliberately omitted — they were out
of scope for this read.

### The recurring structure

Every challenge repeats three graded stages:

> Human Design → AI Design → Human–AI Co-Design

Each is worth about 1% and each is graded on the same three things, which is
why `AGENTS.md` makes them standing rules rather than per-stage tasks:

1. **A documented AI-recommendation decision** — one concrete
   *Human Design → AI recommendation → accepted / modified / rejected → why.*
2. **Evidence and provenance** — the required output shape is
   **Answer + Evidence + Provenance + Uncertainty.**
3. **Tests with failure cases** — 5–10 cases, at least one deliberate failure,
   and at least one comparison against a *simpler* baseline.

During Human Design stages the assignments explicitly ask the human to make the
major design decisions with minimal AI involvement. That is a grading criterion,
not a preference.

### Cumulative technical requirements

**Challenge 2** — a working Streamlit application over real project data.
Structured generation (text → validated schema). Retrieval with inspectable
evidence. At least one human control that changes the output: refine a query,
change a filter, inspect evidence, compare alternatives, correct or reject.
5–10 test cases with a failure case. A comparison against a simpler alternative
(e.g. keyword search vs embeddings; LLM-only vs LLM + retrieved evidence; a
simple rule vs a model). Deployment, or a reproducible local demo with the
limitation stated. Study of existing work: 1+ public dataset, 2–3 related GitHub
projects, 1–2 papers.

**Challenge 3** — a Neo4j knowledge graph: 4–8 node types, 4–10 relationship
types, 15–30+ example nodes, meaningful properties. Cypher demonstrating
creation, relationships, `MATCH`, traversal, at least one multi-hop query, and
one project-specific query that shows why a graph representation helps. Graph
visualisation. Then GraphRAG: natural-language question → Neo4j retrieval →
retrieved context → LLM → grounded answer, displaying the supporting evidence,
including at least one multi-hop retrieval, demonstrated on three question
types (basic retrieval, multi-hop reasoning, analytics). Then a GNN
message-passing exploration (GCN, GraphSAGE or GAT) and an account of how
embeddings could rank nodes or subgraphs before retrieval. A small working
example is explicitly sufficient for the GNN stage.

Also required at Human Design: research 2–3 *existing* ontologies or
vocabularies in the domain and judge what is reusable. And at least five
questions the graph should answer that genuinely need multi-hop reasoning.

**Challenge 4** — community detection on the existing graph (Louvain, Leiden or
similar), with domain interpretation of what the communities mean. One or two
multimodal generation outputs chosen from visualisation, speech, charts or
video, justified over a plain text response. 3–5 user-facing features,
prioritised High / Medium / Future, plus an interface wireframe. Evaluation with
baseline comparison and an ablation study (with vs without community
information). GNN work is **optional** here and explicitly does not affect
grading. One GenAI-based presentation of findings. Explainability features:
evidence paths, source references, uncertainty indicators, human verification.

### Principles stated repeatedly across the assignments

These are quoted or closely paraphrased, and they are in `AGENTS.md` as rules:

- "Big Data systems process. Tools/models calculate. RAG retrieves. LLMs
  interpret and explain."
- "Use real results. Do not fabricate numbers."
- Avoid "LLM calculations that should use Python/SQL/tools."
- Avoid "large amounts of generated code you do not understand" — "you are
  responsible for reviewing and understanding the generated code."
- "Clearly distinguish implemented results from proposed features."
- Avoid generic chatbots, dashboards with little AI value, unnecessary agents,
  and processing the entire dataset to make a point.
- Do not upload confidential or restricted data.

---

## 4. What is deliberately not decided

The problem domain is **blank**, and that is a choice, not an oversight. The
assignments constrain the shape of the work, not its subject. Recording a domain
decision in `docs/decisions/` is the first substantive task.

Everything downstream of that is also open: datasets, the graph schema, the
retrieval strategy, the LLM provider, the analytics stack. `requirements.txt`
holds only what the assignments mandate by name — Streamlit (Challenge 2) and
the Neo4j driver (Challenge 3) — precisely so that nothing was chosen by
accident.

Do not add analytics, ML, graph-algorithm or LLM libraries speculatively.

---

## 5. Measurements taken on this machine

These numbers are why `AGENTS.md` says what it says. They were measured, not
assumed.

### Workers misreport their own work

Two Sonnet workers were given independent briefs in a scratch repository and
asked to implement one module and its tests each. Both produced correct code —
17 tests passed, the implementations were clean and well documented.

**Both then reported that the files "already existed" and that they had only
verified them.** One invented a cause: "likely from an earlier pass of this same
job." The scaffold commit proved otherwise; the files were created two minutes
into the worker runs and were still untracked.

Nothing broke, because the code was right. But the reports were false in a way
that would be invisible without checking. Hence the rule: a model's account of
its own work is not evidence, and hence the `Stop` hook, which costs no tokens
and cannot misremember.

### The coordinator is the cost centre, not the workers

| | Requests | Tokens |
|---|---|---|
| Two Sonnet workers, combined | 5 tool calls each | **136,091** |
| The Opus coordinator session | 16 | **2,571,097** cache-read + 15,353 output |

Roughly **19×**. Each coordinator turn cost about **160,700** tokens *regardless
of what the turn did*, because every turn re-sends the whole conversation. Some
of that was reading assignment files rather than orchestrating — which is the
point: the marginal cost of a turn is set by context length, not by the work in
it.

Consequences: keep sessions short, `/clear` between unrelated tasks, prefer a
fresh session per task, and keep `AGENTS.md` lean because it is paid for on
every request.

### The account is on Claude Pro

Not Max. Usage is shared with claude.ai chat, there are separate per-family
Opus and Sonnet caps, and usage credits are **off**, so hitting a limit stops
work rather than spilling into paid overflow.

`/usage` is the only place that shows plan percentage; local session history
gives tokens only. Worth running after a real build session to convert these
token counts into "how many builds like this per week."

---

## 6. Environment constraints

**Intel Mac** — i5-1038NG7, 4 cores / 8 threads, 16 GB RAM, x86_64, macOS 15.3.2.

**Homebrew dropped macOS Intel x86_64 support in September 2026.** It is a Tier 3
configuration and no bottles are built for it. This is not a warning to ignore:
`brew install gh` does not fail, it silently begins compiling **Go 1.27.3** from
source and then gh on top. On this hardware that is potentially an hour for a
tool that ships a prebuilt binary.

**Therefore the dependency rule:** prebuilt binaries and Python wheels only.
Install CLI tools from the upstream project's own GitHub release assets, verified
against the published `checksums.txt`. Check for an `x86_64` + cp310 wheel
*before* adding a Python dependency. Anything requiring compilation is a cost to
be decided on explicitly.

Python is 3.10.6 at `/usr/local/bin/python3`; the project venv is `.venv`.
All 47 installed packages came from wheels — no source builds.

---

## 7. Tooling installed, and how it is configured

All three installed from checksum-verified upstream release binaries, not brew.

**`treehouse` v3.1.8** — a pool of pre-warmed disposable git worktrees, so
parallel tasks never collide. `treehouse get --lease` is the non-interactive
acquire and prints the path. The pool lives in `~/.treehouse`, outside the repo.
Verified: real git worktrees, full isolation between worktree and main checkout,
and `return` refuses to discard uncommitted changes without `--force` rather
than guessing.

**`no-mistakes` v1.106.1** — a local git proxy. `git push no-mistakes <branch>`
runs intent → rebase → review → test → document → lint → push → PR → CI in a
disposable worktree, and forwards to the real remote only if everything passes.

Verified against a deliberately broken branch: it **held the branch at
`review: awaiting_approval`** and never forwarded it. The review agent traced
the broken function by hand, tabulated actual vs specified behaviour, picked out
the sharpest case itself, and reported `ask-user` at `error` severity rather than
silently reverting an apparently intentional change.

Three things to know:

- The AI review stage is **on by default** and inherits the session's model. It
  was measured running on Opus at **~596,000 tokens for a one-line diff**. It is
  now pinned in `~/.no-mistakes/config.yaml`:
  `agent_config: {claude: {model: sonnet, effort: high}}`.
- `intent` scanning, which reads recent local Claude Code transcripts on every
  push, has been **turned off** there too.
- Initialising a gate installs a **login LaunchAgent** so the daemon restarts at
  every login. `no-mistakes daemon uninstall` removes it. Also note `init`
  requires an `origin` remote, and by default installs a `/no-mistakes` skill at
  user level unless you pass `--skip-user-skill`.

**`gh` 2.103.3** — authenticated as `halleepham`, scopes `repo` and `workflow`.
No `delete_repo` scope, so nothing can be destroyed through it.

**Not installed: `firstmate`.** It was assessed and deliberately deferred. Its
value is supervising a fleet of parallel agents, the work here is currently
serial, and its model-routing machinery is built to arbitrage quota across
several providers — with one provider and one shared budget its own
`shared_quota` switch disables the centrepiece. Revisit only when both of these
are true: consistently running 3+ parallel tasks, **and** on a Max plan or with
usage credits enabled.

---

## 8. What exists in this repository right now

```
AGENTS.md                      the rules; loaded on every agent request
CLAUDE.md                      imports AGENTS.md so both tools read one file
README.md                      what this is, how to run it
requirements.txt               streamlit, neo4j, python-dotenv, pytest — pinned
.env.example                   Neo4j keys; LLM provider deliberately blank
.gitignore                     private/ and data/ excluded from the first commit
.claude/settings.json          Stop hook; bgIsolation disabled for this repo
.claude/hooks/verify-tests.sh  runs the suite before a completion claim lands
tests/test_environment.py      6 tests: python version, mandated imports,
                               and that .env and private/ are really ignored
docs/HANDOFF.md                this file
docs/decisions/                empty; one file per decision from here on
```

The `Stop` hook was tested across four paths — no tests present, tests passing,
`tests/` not an importable package, and genuinely failing tests — and exits
0/0/0/2 with clean stderr on macOS's bash 3.2. It found and survived one bug of
its own during testing: it had initially reported a collection error as a test
failure, which is precisely the false-report behaviour it exists to prevent.

`bgIsolation` is set to `none` in `.claude/settings.json` because this repo is
itself the deliverable and must live at `~/murphy` on `main`; the usual
background-agent worktree isolation would defeat that. Reconsider if parallel
agents ever work here.

---

## 9. Where to pick up

In order:

1. **Resolve the Challenge 4 question in §2.** Submit from `murphy-v1`, or
   confirm it is already handled. Do this before building anything here.
2. **Decide the problem domain and record it** in `docs/decisions/0001-*.md`
   with the reasoning. Everything else is blocked on this.
3. **Choose the data sources** against the Challenge 2 criteria — real,
   focused, groundable, evaluable — and record the licensing and
   size decision that governs whether anything may be committed under `data/`.
4. **Then, and only then,** add dependencies, and record why each one.

Nothing in this repository prejudges steps 2 and 3. That was the point of
keeping it empty.
