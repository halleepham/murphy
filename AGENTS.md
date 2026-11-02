# murphy — agent instructions

CS 5542 Big Data Analytics & Applications. Solo project, Hallee Pham.
Rebuilt from zero on 1 Nov 2026. The previous repository is archived at
`halleepham/murphy-v1`; **do not consult it.** Nothing in it is a precedent.

Full context: `docs/HANDOFF.md`. Decisions: `docs/decisions/`.

## The problem domain

Decided in `docs/decisions/0002-problem-domain.md`: itinerary-specific flight
disruption risk, with trustworthy probability forecasts and recommended
alternatives. Do not assume a dataset, model, or architecture until a decision
record exists for it. If a task seems to need one, propose options and ask.

## Every stage follows the same graded loop

The course repeats one structure at each stage, each worth ~1%:

> Human Design → AI Design → Human–AI Co-Design

Three things are graded at every stage, so build them as you go rather than
reconstructing them at submission time:

1. **A documented AI-recommendation decision.** One concrete instance of
   *Human Design → AI recommendation → accepted / modified / rejected → why.*
   Record these in `docs/decisions/` as they happen.
2. **Evidence and provenance on every output.** The required shape is
   **Answer + Evidence + Provenance + Uncertainty.** A record ID, file name,
   timestamp, graph node, or model output — something the user can trace.
3. **Tests including failure cases.** 5–10 cases per stage, at least one
   deliberate failure or thin-evidence case, and at least one comparison
   against a *simpler* baseline.

**Human Design stages are co-planned** (owner's decision, `0002`). Propose
options with a recommendation; the owner decides. Deadlines have passed:
build only the end product, no slides or reports, but use every required
method.

## Rules from the assignments — not negotiable

These are quoted or paraphrased course requirements, not preferences.

- **Separate the responsibilities.** "Big Data systems process. Tools/models
  calculate. RAG retrieves. LLMs interpret and explain."
- **The LLM never computes a value.** An LLM calculation that should have been
  Python or SQL is named as a thing to avoid. Numbers come from code.
- **Never fabricate a number.** "Use real results. Do not fabricate numbers."
  No illustrative figures, no placeholder metrics, no invented sample output.
  If a value isn't computed yet, say so.
- **Distinguish built from proposed.** Clearly separate implemented results
  from proposed features and expected benefits.
- **Never commit confidential or restricted data.** Private material goes in
  `private/` (gitignored). `docs/` is committed and public.
- **The human must stay in the loop.** Every stage needs at least one real
  control: refine a query, change a filter, inspect evidence, compare
  alternatives, correct or reject a result.
- **Avoid generated code the human doesn't understand.** Explain what you
  wrote. Prefer less code. The repo owner is accountable for all of it.
- **Avoid:** generic chatbots, dashboards with little AI value, unnecessary
  agents, processing the whole dataset to prove a point.

## Verification: worker reports are not evidence

Measured in this repo's setup session, and the reason this section exists: two
Sonnet workers each wrote files, lost track of having done so, and reported
them as pre-existing. One invented a cause. The code was correct; the reports
were false.

So:

- **Never report work as done on the strength of your own recollection.** Run
  the check and paste the output. `git status`, the test command, the actual
  file contents.
- **Verify a claim about repository state against git**, not against memory of
  what you did earlier in the session.
- A `Stop` hook in `.claude/settings.json` runs the test suite so a completion
  claim is checked mechanically before it reaches the reader. Do not disable it.
- When reporting a result, state what you actually ran. "Tests pass" without a
  command and its output is not a report.

## Model and cost discipline

This account is on **Claude Pro**, not Max. Usage is shared with claude.ai
chat, and there is no paid overflow. Measured here: a long-running coordinator
session cost ~160k tokens *per turn* regardless of what the turn did, roughly
19× two Sonnet workers combined. Context length is the cost, not the work.

- **Keep sessions short.** `/clear` between unrelated tasks. A fresh session
  per task beats one long one.
- **Opus for design conversations; Sonnet for implementation workers.** Pin
  worker models explicitly rather than letting them inherit.
- This file is loaded on every request. Keep it under ~200 lines. Detail goes
  in `docs/`, which loads only when read.

## Environment — this machine has real constraints

- **Intel Mac** (x86_64, 16 GB). **Homebrew dropped macOS Intel support in
  September 2026** — it builds no bottles for this machine, so `brew install`
  silently falls back to compiling from source. `brew install gh` would have
  built Go 1.27.3 first.
- **Therefore: prebuilt binaries and Python wheels only.** Install CLI tools
  from the project's own GitHub release assets, verified against their
  published `checksums.txt`. Anything that needs compiling is a cost to be
  decided on, not absorbed silently.
- Python is 3.12.16 at `~/.local/opt/cpython-3.12.16` (`0003`). Venv at `.venv`.
- Check wheel availability for `x86_64` + cp312 *before* adding a dependency.
  PyTorch is capped at 2.2.2 on this machine (needs `numpy<2`).
- Free models only (no paid APIs). No Streamlit (`0003`).
- Every decision record lists alternatives and evidence: the owner's final
  report must justify each choice.
- Never add a dependency without asking.

## Tooling installed

- `treehouse` — pooled disposable git worktrees. `treehouse get --lease` for a
  non-interactive acquire. Pool lives in `~/.treehouse`, outside the repo.
- `no-mistakes` — local git proxy. `git push no-mistakes <branch>` runs
  review → test → docs → lint before forwarding. Globally pinned to Sonnet in
  `~/.no-mistakes/config.yaml`; transcript-scanning (`intent`) is off. Note it
  installs a login LaunchAgent when a gate is initialised.
- `gh` — authenticated as `halleepham`, scopes `repo` + `workflow`. No
  `delete_repo` scope, so nothing can be destroyed through it.

## Working style

- Propose before building anything structural. Small phases, one at a time.
- If the docs contradict the data, stop and say so. Never quietly work around.
- Record decisions in `docs/decisions/` **as they are made**, with the
  reasoning and the evidence. Reconstructing the "why" later loses it.
- US spelling in slides, diagrams, and reports.
- Be concise. Short sentences.
