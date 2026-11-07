# 0012 — What the app does

Date: 6 Nov 2026. Nothing here is built.

## Decision 1 — The features

**Owner, 6 Nov 2026:**

- **Accepted as listed in `docs/PLAN.md` layer 7:** enter a flight and a time constraint;
  see the chance of making it, keep or consider changing, alternatives with the estimated
  cost, the evidence, the track record, and how the forecast changes by lead time.
- **Required by the owner:** generative AI, and a visualization that is unique rather
  than a standard chart, with at least an interactive map. The map may show "zones", such
  as weather, and how they affect the forecast and risk of different alternatives. This
  may come with communities and graphs (layer 4).
- **A thought, not required:** an LLM the user can ask about policies, what to do, and
  why one flight was recommended over another. It answers in its own way, like a chatbot,
  and is separate from the LLM that writes the explanations.

## Open — AI proposals, not yet reviewed by the owner

**Map.** Weather exists only at 30 airports, with no gridded data, so weather zones
cannot cover the whole country. Communities can cover every airport, but do not exist yet.
Proposed order: routes for the current flight and its alternatives, colored by risk, with
a control for the forecast time; then weather at the 30 airports, with the others marked as
having no weather data; then community zones if layer 4 supports them.

**"How the zone affects risk."** Proposed: rerun a flight's forecast with its weather
inputs blanked and show the difference. That reports what the model did with the inputs. It
does not show that the weather caused the risk, and the app must say so.

**Question answering.** The course warns against generic chatbots. Proposed: treat this as
the GraphRAG of layer 5, so it answers from retrieved evidence and computed results, shows
its sources, and declines when evidence is thin. Numbers still come from code.
