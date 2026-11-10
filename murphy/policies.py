"""Policy passages. Code parses and retrieves; no LLM decides an entitlement.

Sources, saved in private/policies/raw/ (contract text is not committed):
  - 14 CFR parts 250, 259, 260 from the eCFR API (federal: applies to every carrier);
  - Delta's contract of carriage (HTML) and Southwest's (PDF).

`passages()` splits each source at its own section headings, then cuts any long section
into windows of at most MAX_WORDS at paragraph breaks. Each passage keeps its carrier,
section label, source URL, collection date and the raw file's SHA-256, so it can be traced.

Search (Decision 3): code picks a situation; its fixed question is searched over the
flight's carrier plus the federal rules. Carriers without a collected contract get the
federal rules only, and `search` says so. Methods: "embed" (bge-small, cosine), "bm25".

Build: python -m murphy.policies  -> private/policies/{passages.parquet, embed.npy}
"""

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "private" / "policies" / "raw"
OUT = ROOT / "private" / "policies"
COLLECTED = "2026-11-07"
MAX_WORDS = 220  # bge-small reads 512 tokens; 220 words stays well inside
MODEL = "BAAI/bge-small-en-v1.5"
K = 3
CONTRACTS = {"DL": "Delta", "WN": "Southwest"}  # carriers whose contract was collected

SOURCES = {
    # file: (carrier, URL)
    "ecfr_250.xml": ("federal", "https://www.ecfr.gov/current/title-14/part-250"),
    "ecfr_259.xml": ("federal", "https://www.ecfr.gov/current/title-14/part-259"),
    "ecfr_260.xml": ("federal", "https://www.ecfr.gov/current/title-14/part-260"),
    "delta_coc.html": ("DL", "https://www.delta.com/us/en/legal/contract-of-carriage-dgr"),
    "southwest_coc.pdf": ("WN", "https://www.southwest.com/swa-resources/pdfs/corporate-commitments/contract-of-carriage.pdf"),
}


def _clean(s: str) -> str:
    return re.sub(r"[ \t ]+", " ", s).strip()


def ecfr_sections(path: Path) -> list[tuple[str, list[str]]]:
    """One section per eCFR section (DIV8): ('14 CFR 260.6 Refunding fare ...', paragraphs)."""
    root = ET.parse(path).getroot()
    out = []
    for div in root.iter("DIV8"):
        head = _clean("".join(div.find("HEAD").itertext())).replace("§ ", "14 CFR ")
        paras = [_clean("".join(p.itertext())) for p in div.findall("P")]
        out.append((head, [p for p in paras if p]))
    return out


DELTA_RULE = re.compile(r"^RULE \d+:")
DELTA_SUB = re.compile(r"^[A-Z][.)]? ")  # "A. Involuntary Refunds", "A) ...", "A General Provisions"


def delta_sections(path: Path) -> list[tuple[str, list[str]]]:
    """Split at RULE headings and their lettered subsections (bold text in the page)."""
    s = path.read_text(errors="replace")
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", s)
    s = re.sub(r"(?is)<b\b[^>]*>(.*?)</b>", lambda m: "\n\x00" + m.group(1) + "\n", s)
    s = re.sub(r"(?i)</p>|<br\s*/?>|</li>|</h\d>|</div>", "\n", s)
    lines = [_clean(html.unescape(re.sub(r"<[^>]+>", "", l))) for l in s.split("\n")]
    out, rule, sub, paras = [], None, None, []

    def flush():
        if rule and paras:
            out.append((f"Delta {rule}" + (f" — {sub}" if sub else ""), paras.copy()))
        paras.clear()

    for line in lines:
        bold = line.startswith("\x00")
        line = line.lstrip("\x00").strip()
        if not line:
            continue
        if bold and DELTA_RULE.match(line):
            flush(); rule, sub = line, None
        elif bold and rule and DELTA_SUB.match(line) and len(line) < 140:
            flush(); sub = line
        elif rule:
            paras.append(line)
    flush()
    return out


SW_FURNITURE = re.compile(r"^(Southwest Airlines Co\.|\d+|59th Revised|\d\d/\d\d/\d{4}|Contract of Carriage)$")


def southwest_sections(path: Path) -> list[tuple[str, list[str]]]:
    """Split at the headings listed in the PDF's own table of contents (pages 2-4)."""
    import pypdf
    pages = [p.extract_text() or "" for p in pypdf.PdfReader(path).pages]
    toc = [re.sub(r"\s*\.{3,}.*$", "", l).strip() for p in pages[1:4] for l in p.splitlines() if "...." in l]
    toc = [t for t in toc if re.match(r"^(\d+|[a-z])\. ", t)]
    body = [_clean(l) for p in pages[1:] for l in p.splitlines() if "...." not in l]
    out, top, sub, paras, i = [], None, None, [], 0

    def flush():
        if top and paras:
            out.append((f"Southwest {top}" + (f" — {sub}" if sub else ""), paras.copy()))
        paras.clear()

    for line in body:
        if not line or SW_FURNITURE.match(line):
            continue
        # headings appear in TOC order; match the next one only, so body text can't fake one
        if i < len(toc) and line.rstrip(". ") == toc[i].rstrip(". "):
            flush()
            if toc[i][0].isdigit():
                top, sub = toc[i], None
            else:
                sub = toc[i]
            i += 1
        elif paras and not re.search(r"[.:;]$", paras[-1]):
            paras[-1] += " " + line  # PDF wraps lines mid-sentence: rejoin
        else:
            paras.append(line)
    flush()
    if i < len(toc):
        raise ValueError(f"Southwest: only {i} of {len(toc)} TOC headings found in the body")
    return out


def windows(paras: list[str], max_words: int = MAX_WORDS) -> list[str]:
    """Group paragraphs into chunks of at most max_words; a longer paragraph is cut by words."""
    chunks, cur, n = [], [], 0
    for p in paras:
        words = p.split()
        while len(words) > max_words:  # one huge paragraph
            if cur:
                chunks.append(" ".join(cur)); cur, n = [], 0
            chunks.append(" ".join(words[:max_words])); words = words[max_words:]
        if n + len(words) > max_words and cur:
            chunks.append(" ".join(cur)); cur, n = [], 0
        cur += words; n += len(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


PARSERS = {".xml": ecfr_sections, ".html": delta_sections, ".pdf": southwest_sections}


def passages(raw: Path = RAW) -> pd.DataFrame:
    rows = []
    for name, (carrier, url) in SOURCES.items():
        path = raw / name
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        for section, paras in PARSERS[path.suffix](path):
            parts = windows(paras)
            for j, text in enumerate(parts):
                rows.append({
                    "id": f"{path.stem}:{len(rows):04d}",
                    "carrier": carrier, "section": section,
                    "part": f"{j + 1}/{len(parts)}", "text": text,
                    "source_file": name, "url": url, "collected": COLLECTED, "sha256": sha,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- search

# situation -> (label shown to the traveler, the question that is searched)
SITUATIONS = {
    "cancelled": ("My flight is cancelled",
                  "The airline cancelled my flight. Can I get my money back or a seat on another flight?"),
    "delay_refund": ("Long delay: I no longer want to go",
                     "My flight is delayed by several hours and I no longer want to travel. Can I get a refund?"),
    "delay_care": ("Long delay: hotel or meals",
                   "My flight is delayed overnight. Will the airline pay for a hotel or meals?"),
    "schedule_change": ("The airline changed my flight time",
                        "The airline changed my departure time by several hours. What are my options?"),
    "change_myself": ("I want to switch flights myself",
                      "I want to switch to an earlier flight myself. Is there a fee, and what happens to my fare?"),
    "missed_connection": ("I missed my connection",
                          "My first flight was late and I missed my connecting flight. Will the airline rebook me?"),
    "tarmac": ("Stuck on the plane on the ground",
               "We have been sitting on the plane on the runway for three hours. What are the rules?"),
    "bumped": ("Bumped from an oversold flight",
               "The flight is oversold and I was denied boarding. What compensation do I get?"),
}


# the situation shown first, from the trip's advice; the traveler can pick another
DEFAULT_SITUATION = {"cancelled": "cancelled", "consider_changing": "change_myself",
                     "at_risk_no_better_flight": "delay_refund", "too_uncertain": "delay_refund",
                     "keep": "delay_refund"}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Index:
    """Both methods over the passages. Embeddings are passed in or computed."""

    def __init__(self, passages: pd.DataFrame, embed: np.ndarray | None = None):
        from rank_bm25 import BM25Okapi
        self.p = passages.reset_index(drop=True)
        self.embed = embed
        self._bm25 = BM25Okapi([tokens(f"{s} {t}") for s, t in zip(self.p.section, self.p.text)])
        self._model = None

    def _encode(self, texts):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL)
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32)

    def scores(self, method: str, question: str) -> np.ndarray:
        if method == "embed":
            if self.embed is None:
                self.embed = self._encode(embed_texts(self.p))
            return self.embed @ self._encode([question])[0]
        if method == "bm25":
            return self._bm25.get_scores(tokens(question))
        raise ValueError(f"unknown method {method!r}")

    def search(self, question: str, carrier: str, method: str = "embed", k: int = K) -> dict:
        """Top k passages from the carrier's contract and the federal rules. Ties: source order."""
        carrier = carrier.strip().upper()
        pool = {"federal", carrier} if carrier in CONTRACTS else {"federal"}
        cand = np.flatnonzero(self.p.carrier.isin(pool).to_numpy())
        every = self.scores(method, question)
        top = cand[np.argsort(-every[cand], kind="stable")[:k]]
        note = None if carrier in CONTRACTS else (
            f"{carrier}'s contract of carriage was not collected; federal rules only.")
        return {"carrier": carrier, "method": method, "question": question, "note": note,
                "passages": self.p.iloc[top].assign(score=every[top].astype(float).round(4))
                            .to_dict("records")}


def embed_texts(p: pd.DataFrame) -> list[str]:
    """The section label goes in front, so "RULE 22: REFUNDS" helps a refund question."""
    return [f"{s}. {t}" for s, t in zip(p.section, p.text)]


if __name__ == "__main__":
    df = passages()
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "passages.parquet", index=False)
    np.save(OUT / "embed.npy", Index(df)._encode(embed_texts(df)))
    print(df.groupby("carrier").agg(sections=("section", "nunique"), passages=("id", "size"),
                                     words=("text", lambda t: t.str.split().str.len().sum())))
