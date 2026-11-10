"""Tests for policy passages: parsing and splitting.

Unit tests use tiny strings written here; they are fixtures, not data. Real-source
tests skip when private/policies/raw is absent (contract text is not committed).

Cases: windows keep paragraphs whole, cut one huge paragraph, and handle nothing
(thin evidence); a tiny eCFR file; and the real sources, where every passage
must trace to a file whose hash matches, and the sections a traveler needs exist.
"""

import hashlib

import pandas as pd
import pytest

from murphy import policies as P


def test_windows_keep_paragraphs_whole():
    paras = ["a " * 100, "b " * 100, "c " * 100]
    chunks = P.windows([p.strip() for p in paras], max_words=220)
    assert len(chunks) == 2
    assert chunks[0].split().count("a") == 100 and chunks[0].split().count("b") == 100


def test_windows_cut_one_huge_paragraph():
    chunks = P.windows(["w " * 500], max_words=220)
    assert [len(c.split()) for c in chunks] == [220, 220, 60]


def test_windows_nothing_gives_nothing():  # thin evidence: a section with no text
    assert P.windows([]) == []


def test_ecfr_sections(tmp_path):
    x = tmp_path / "e.xml"
    x.write_text('<DIV5><DIV8 N="1.1"><HEAD>§ 260.6 Refunds.</HEAD>'
                 '<P>(a) Pay <I>promptly</I>.</P><P> </P></DIV8></DIV5>')
    assert P.ecfr_sections(x) == [("14 CFR 260.6 Refunds.", ["(a) Pay promptly."])]


REAL = P.RAW / "southwest_coc.pdf"


@pytest.fixture(scope="module")
def real():
    if not REAL.exists():
        pytest.skip("private/policies/raw not present")
    return P.passages()


def test_real_passages_trace_to_their_files(real):
    for name, g in real.groupby("source_file"):
        assert g.sha256.nunique() == 1
        assert g.sha256.iloc[0] == hashlib.sha256((P.RAW / name).read_bytes()).hexdigest()
    assert real.id.is_unique
    assert real.text.str.split().str.len().max() <= P.MAX_WORDS


def test_real_sections_a_traveler_needs(real):
    need = {
        "federal": ["14 CFR 260.6", "14 CFR 259.4"],  # refunds; tarmac delay plan
        "DL": ["RULE 19", "RULE 21", "RULE 22"],      # delays, rerouting, refunds
        "WN": ["9. Flight Changes", "4. Tickets — c. Refunds"],
    }
    for carrier, keys in need.items():
        secs = real[real.carrier == carrier].section
        for k in keys:
            assert secs.str.contains(k, regex=False).any(), (carrier, k)


@pytest.fixture(scope="module")
def index(real):
    import numpy as np
    if not (P.OUT / "embed.npy").exists():
        pytest.skip("policy embeddings not built")
    return P.Index(pd.read_parquet(P.OUT / "passages.parquet"), np.load(P.OUT / "embed.npy"))


def test_search_stays_in_the_airlines_pool(index):
    for m in ("embed", "bm25"):
        r = index.search(P.SITUATIONS["cancelled"][1], "dl", m, k=10)
        assert r["carrier"] == "DL" and r["note"] is None
        assert {x["carrier"] for x in r["passages"]} <= {"DL", "federal"}


def test_search_for_an_airline_not_collected_gives_federal_rules_and_says_so(index):  # thin evidence
    r = index.search(P.SITUATIONS["tarmac"][1], "F9")
    assert {x["carrier"] for x in r["passages"]} == {"federal"} and "not collected" in r["note"]


def test_unknown_method_fails(index):  # deliberate failure
    with pytest.raises(ValueError):
        index.search("refund", "DL", "llm")


def test_every_default_situation_exists():
    assert set(P.DEFAULT_SITUATION.values()) <= set(P.SITUATIONS)
