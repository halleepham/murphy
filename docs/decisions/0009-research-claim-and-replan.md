# 0009 — Research claim, users, and the replan

**Date:** 6 Nov 2026
**Status:** accepted by the owner. Details of the trust signal and the decision
test are not decided yet. This revises the plan of 2 Nov; `docs/PLAN.md` holds
the result.

## Decision 1 — The claim is the project's main question

> A forecasting system that knows when to distrust itself makes better
> decisions than one that is merely accurate.

Stated by the owner on 5 Nov. No earlier record had it. The closest was `0002`:
"how much the user can trust the forecast". The 2 Nov plan would have finished
without testing it, since it had no way to distrust a single forecast, no action
tied to that, and no test based on decisions.

## Decision 2 — Users and what makes murphy different

**Owner's words, summarized.** Users are travelers with a deadline or time
constraint: a tight connection, a wedding, a meeting. A user enters a flight and
sees the risk. Other tools give a number that does not relate to a person's
itinerary and time constraints; murphy does. Trusting the score is a mix of
having evidence and not giving just a single number about delay.

## Decision 3 — Replan, do not restart

The owner asked whether to replan from scratch. **AI recommended** keeping the
code and data and rewriting the plan: the data layer, `as_of` rules and inputs
are what any version needs, and everything missing is unbuilt. **Owner: accepted.**

## Decision 4 — Retrain walk-forward; 2026 stays sealed

**Options:** score the current model on 2026 now; or retrain, build the trust
layer on earlier years, and test everything on 2026 together. **AI recommended**
the second: 2026 is the only clean year left for the claim, and a look now would
compromise it. **Owner: accepted**, and said retraining is fine if it follows
proper data science and the claim. The current model becomes the baseline.
This supersedes the plan in `0008` Decision 8 to score that model next. Its
choice of no calibration change stands for that model.

## Decision 5 — Lead times stay at 24, 6 and 2 hours

**Options:** only the day before; up to about a week; from booking. **AI
recommended** up to about a week. **Owner: rejected**, keeping 24, 6 and 2 hours.
More than 24 hours out, the app says it is too early for a reliable forecast.
That message is an AI proposal, included in the plan.

## Decision 6 — Required methods need not serve the claim

**AI proposed** judging every method by whether it improves trust or decisions.
**Owner: modified.** Methods can enhance the claim without all going into it. The
owner's example: flights are scored independently, but community detection and
graphs can show that risk is not isolated. The professor suggested knowledge graphs
as a way for users to understand things, though that need not be their only use.
Assignment instructions still guide how each method is used and evaluated.

## Decision 7 — Course deadlines do not drive the plan

Challenge 5 (released 5 Nov, poster, demo video, GitHub) has dates of 6, 9 and
13 Nov. **AI recommended** a reduced version for 9 Nov. **Owner: rejected**: do not
worry about the deadline or the assignment due 6 Nov, aim for a good project, and
use **all required methods**. So the GNN is no longer first to cut. Slides and
posters are still not built, as in `0002`.

## AI-recommendation instances

| AI recommended | Owner | Why |
|---|---|---|
| Replan, keep code and data | accepted | what exists is needed by any version |
| Walk-forward retrain; seal 2026 for the claim | accepted | proper evaluation; one clean year left |
| Checking up to a week ahead | rejected | owner keeps 24, 6 and 2 hours |
| Judge every method by the claim | modified | methods may enhance it without serving it |
| Reduced scope for 9 Nov | rejected | quality over deadline; all methods required |
