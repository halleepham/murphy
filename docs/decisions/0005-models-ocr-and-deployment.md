# 0005 — Models, OCR, and deployment direction

**Date:** 2 Nov 2026
**Status:** accepted (direction). Hosting details get verified at deploy time.

Research notes with URLs: `private/research/research-models.md`.

## Decision 1 — Verified stack for embeddings and graph neural networks

Test-installed from wheels in a scratch Python 3.12 venv on this machine,
2 Nov 2026:
- torch 2.2.2, numpy 1.26.4, transformers 4.57.6, sentence-transformers 5.7.0,
  torch-geometric 2.8.1.
- A `GCNConv` forward pass ran.
- `BAAI/bge-small-en-v1.5` loaded and encoded 3 sentences (384 dimensions) in
  10.9 s, including the model download.

sentence-transformers 6.x is **not** usable: it needs transformers 5, which
needs a newer torch than this machine's ceiling of 2.2.2.

## Decision 2 — LLM: free hosted, with a local fallback

- **Primary: Groq's free tier.** It supports strict JSON schemas and has a
  vision model, and per its docs it does not retain inference data by default.
  Free-plan limits are UNVERIFIED.
- **Fallback:** a local model through Ollama, which ships a universal binary.
  Its speed on this Mac has not been measured.
- **Rejected: Gemini's free tier for user data.** Its terms say free-tier
  inputs are used to improve Google's products and may be read by human
  reviewers.

## Decision 3 — Screenshot OCR runs in the user's browser

The owner asked for a non-local option, because a deployed server cannot use
Apple's macOS OCR.

**Chosen: Tesseract.js**, which does OCR in the browser. The image never leaves
the user's device, it costs nothing, and it works the same locally and
deployed. Only the extracted text goes to the LLM for schema extraction.
Flight numbers are then checked against BTS records.

**Alternatives considered:**
- OCR.space free API: about 25,000 requests a month per third-party listings.
  Its retention terms were not confirmed on its own site.
- A vision LLM (Groq, Cloudflare): the screenshot leaves the device.
- Cloud Vision and Azure: their free tiers need a billing account.
- Apple Vision: works only on macOS.

**Kept as an optional fallback:** a vision LLM, for screenshots Tesseract reads
badly. This is to be compared in the tests: OCR + LLM against vision LLM.

## Decision 4 — Deployment: simple and only if cheap

**Direction:**
- A Python API (FastAPI) and a custom-designed frontend.
- Heavy computation (training, forecasts for the replay scenarios, the
  community graph) runs **offline**. The deployed app serves the results and
  does only light work.
- Neo4j on **Aura Free**. A deployed app cannot reach a database on a laptop.

**Aura Free caveats (from search; confirm on neo4j.com):**
- It may pause after 72 hours without writes and be deleted 30 days after
  that.
- It has no Graph Data Science plugin, so community detection runs in Python.
- Size limit: 200k nodes / 400k relationships per the more recent sources;
  older sources say 50k / 175k.

**Hosting candidate:** Hugging Face Spaces CPU Basic (2 vCPU, 16 GB, sleeps
after 48 hours idle, per the HF docs). Sources conflict on whether a free
account can create Docker Spaces. Verify when deploying. If not, choose
another free host.
