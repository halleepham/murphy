# 0001 — The 5.5 model generation, and stale model references as a standing hazard

**Date:** 1 Nov 2026
**Status:** accepted

Recorded because it changes which models this project runs on, and because the
way the error surfaced is itself the hazard worth documenting.

---

## What happened

The repo owner asked for Claude Code to be updated in order to use "the 5.5
family." The agent asserted that no such family existed, citing two sources:
its own training knowledge (cutoff May 2026) and the bundled `claude-api`
skill's model table (cached 24 Jun 2026). The owner rejected that answer and
said to look it up.

Looked up against the official documentation, the owner was right. Both cited
sources predate the releases by months. The assertion was confidently wrong.

## Decision 1 — Adopt the 5.5 generation

The current lineup, from `platform.claude.com/docs/en/models/overview`:

| Model | API ID | $/MTok in · out | Cache read | Default effort |
|---|---|---|---|---|
| Claude Fable 5.1 | `claude-fable-5-1` | $10 / $50 | $0.25 | `high` |
| Claude Opus 5.5 | `claude-opus-5-5` | $4 / $20 | $0.20 | `medium` |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | $2 / $10 | $0.10 | `high` |
| Claude Haiku 5.5 | `claude-haiku-5-5` | from $0.10 / $0.50 | n/r | `medium` |

Released: Opus 5.5 on 22 Sep 2026, Sonnet 5.5 on 28 Sep, Haiku 5.5 on 7 Oct.
All four are 1M context, 128K max output, reliable knowledge cutoff Jun 2026.
Cache-read rates are 10% of base input, except 2.5% on Fable 5.1 and 5% on
Opus 5.5 and Sonnet 5.5; `n/r` = not recorded, Haiku 5.5's input price is
tiered so no single figure applies.
**Claude Opus 5 and Claude Sonnet 5 are now documented as legacy.**

**Decision 4 of `0000` is unchanged and needs no edit.** It names tiers — "Opus
for design conversations; Sonnet for implementation workers" — not model IDs.
It now resolves to `claude-opus-5-5` and `claude-sonnet-5-5`.

**This generation moves the cost picture in the right direction** for the
problem measured in `HANDOFF.md` §5, where the dominant line was 2,571,097
cache-read tokens on an Opus 5 coordinator session. Opus 5.5 prices cache reads
at 5% of base input rather than 10% — $0.20/MTok against Opus 5's $0.50/MTok,
2.5× cheaper per token — and defaults to `medium` effort rather than `high`.

Those are API list rates. **This account is on Claude Pro, not API billing**, so
they indicate direction, not spend. The binding constraint remains the plan's
usage cap. Re-run `/usage` after the first real build session on 5.5 to
re-baseline, per `HANDOFF.md` §5.

## Decision 2 — Model references are stale until verified against live docs

The failure was not a wrong fact. It was two stale caches agreeing with each
other, which read as corroboration.

- The bundled `claude-api` skill's table is dated in its own heading
  (`cached: 2026-06-24`) and lists Opus 5 as current. It will keep doing so.
- Agent training knowledge is older still.
- The two agreeing is not evidence. Neither can know what shipped after it.

**Therefore:** when this project reaches the LLM stage of Challenge 2 or 3, the
model ID goes into code only after checking `platform.claude.com/docs`. A model
ID is never constructed by pattern from a remembered one.

This is the same class of error as the one in `0000` Decision 3 — a model's
confident account of something it cannot actually check — and the same remedy
applies: verify against a live source, not against recollection.

## The AI-recommendation instance

The course grades one *Human Design → AI recommendation → accepted / modified /
rejected → why* per stage. This is a clean instance of **rejected**, and it is
recorded here as it happened rather than reconstructed later:

- **Human design input:** use the 5.5 family for planning and implementation.
- **AI recommendation:** no 5.5 family exists; stay on Opus 5 and Sonnet 5,
  with a cost argument against Fable 5.1 attached.
- **Human decision:** rejected. "Opus 5.5 launched months ago, look it up."
- **Why the rejection was right:** the recommendation rested on two sources
  both older than the releases. Verification against the vendor's own
  documentation confirmed the human and refuted the agent.

No challenge stage is underway, so this is not submitted as a stage
deliverable. It is kept because it is real and dated.

## Evidence

Verified on this machine, 1 Nov 2026:

- Claude Code updated 2.1.263 → 2.1.316 via nvm's npm at
  `~/.nvm/versions/node/v22.14.0`. Confirmed on disk, not from installer
  output. Nothing was written to `/usr/local` (a first attempt there failed
  `EACCES` and installed nothing).
- Model IDs, prices, cache-read rates, default effort levels, release dates and
  legacy status: `platform.claude.com/docs/en/models/overview` and
  `/docs/en/models/opus-5-5/overview`.

## Not verified

- **Whether Claude Pro exposes Opus 5.5 in Claude Code.** Not checkable from a
  tool call. `/model` after a CLI restart is the authoritative answer for this
  account, and the owner runs it.
- The §5 token figures are quoted from `HANDOFF.md`, not re-measured here.

## Consequence for numbering

`0000` Decision 7 states that the domain decision is `0001`. It is now `0002`.
`0000` carries a correction line to that effect; the decision itself stands.
The domain remains undecided and still blocks everything downstream.
