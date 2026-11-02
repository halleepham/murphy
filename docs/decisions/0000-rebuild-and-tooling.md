# 0000 — Rebuild from zero, and the agent tooling adopted

**Date:** 1 Nov 2026
**Status:** accepted

This is the first record and it sets the format: what was decided, what the
alternatives were, what evidence decided it. One file per decision. Number them
in sequence. Record them when the decision is made, not at report time.

---

## Decision 1 — Rebuild rather than refactor

The previous project is archived at `halleepham/murphy-v1` and is **not** a
reference for this one. Its design documents, logs, implementations and
dependency choices are out of scope.

The repo owner's call, on the grounds of having a clearer view of the
technologies and pipelines the project needs. The GitHub repo was renamed
`murphy` → `murphy-v1`; GitHub's permanent redirect was verified (HTTP 301), so
links already submitted still resolve.

## Decision 2 — Adopt treehouse and no-mistakes; defer firstmate

**Considered:** the firstmate / treehouse / no-mistakes workflow as a set.

**Adopted:** `treehouse` (disposable worktree pool) and `no-mistakes` (local git
proxy that validates before forwarding to the remote).

**Deferred:** `firstmate`, the multi-agent coordinator.

**Why.** firstmate's value is supervising a fleet of parallel workers; the work
here is currently serial, so it would mean paying for two live contexts to do
what one session does. Its model-routing machinery is also built to arbitrage
quota across multiple providers — with one provider and one shared budget, its
own `shared_quota` switch turns the centrepiece off. Revisit only when both
hold: consistently 3+ parallel tasks, **and** a Max plan or usage credits on.

**Also rejected:** native Claude Code agent teams as the default working mode,
for the cost reason in Decision 4. Per-worker model pinning works and was
verified; it is the coordinator that is expensive, not the workers.

## Decision 3 — A mechanical verification gate is mandatory

**Evidence.** Two Sonnet workers were each given an independent brief in a
scratch repo. Both produced correct code — 17 tests passed. **Both then reported
that the files "already existed"** and that they had merely verified them; one
invented a cause ("likely from an earlier pass of this same job"). The scaffold
commit disproved it: the files were written two minutes into the worker runs.

**Therefore.** A `Stop` hook in `.claude/settings.json` runs the test suite
before a completion claim reaches the reader. It costs no tokens and cannot
misremember. `AGENTS.md` additionally requires that any claim about repository
state be checked against git rather than recollection.

`no-mistakes` is the second layer, gating the push. Verified against a
deliberately broken branch: it held the branch and never forwarded it.

## Decision 4 — Opus for design, Sonnet for implementation; short sessions

**Evidence.** Measured in this session:

| | Requests | Tokens |
|---|---|---|
| Two Sonnet workers combined | — | 136,091 |
| Opus coordinator session | 16 | 2,571,097 cache-read |

About 19×, at ~160,700 tokens per coordinator turn *regardless of the turn's
content*, because each turn re-sends the conversation. The account is on Claude
**Pro** with usage credits off, so a limit stops work rather than spilling into
paid overflow.

**Therefore.** `/clear` between unrelated tasks; a fresh session per task;
`AGENTS.md` kept under ~200 lines because it is paid for on every request.

`no-mistakes`' review agent is pinned to Sonnet in `~/.no-mistakes/config.yaml`
after it was measured consuming **~596,000 Opus tokens to review a one-line
diff**, having inherited the session default. Its transcript-scanning `intent`
feature is off.

## Decision 5 — Prebuilt binaries and wheels only

**Evidence.** `brew install --dry-run gh` reported: "You are using macOS on
Intel x86_64. We do not provide support for this platform (as-of September
2026)... Homebrew no longer builds bottles for this configuration." It would
have compiled Go 1.27.3 from source before gh.

**Therefore.** CLI tools are installed from the upstream project's own GitHub
release assets, verified against the published `checksums.txt`. Python
dependencies must have an `x86_64` + cp310 wheel, checked before adding.
Anything requiring compilation is an explicit cost decision.

All three installed tools and all 47 Python packages came from wheels or
prebuilt binaries. No source builds.

## Decision 6 — `private/` from the first commit; `docs/` committed

**Evidence.** The old repo had one local-only commit, `bde9500`, carrying
`docs/HUMAN_DESIGN.md`, `docs/DATA_SCHEMA.md`, `docs/CONSTRAINTS_AND_RISKS.md`
and 301 rows of sample data — while `.gitignore` line 25 read `docs/`, "private
planning docs, local only, never committed." Planning docs had been committed
before the directory was made private, leaving a commit that could not be pushed
to a public repo without leaking them.

**Therefore.** `private/` is gitignored from commit one and `docs/` is committed
and reviewable. `tests/test_environment.py` asserts that `.env` and `private/`
are genuinely ignored, making it a test rather than a convention.

## Decision 7 — The problem domain is left undecided

Not an oversight. The assignments constrain the shape of the work, not its
subject, and `requirements.txt` therefore holds only what they mandate by name:
Streamlit (Challenge 2) and the Neo4j driver (Challenge 3).

The domain decision is `0001`, and it blocks everything downstream.
