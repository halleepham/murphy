"""Similar past airport-days. No LLM: retrieval and every number are code.

One document per airport and local date, 2023-2025, the 30 airports with weather:
  - the TAF known just before local midnight (latest known_at, as asof.taf_as_of),
    read over the day's busy hours, 06:00-22:00 local;
  - a plain-English sentence written from its fields (`describe`);
  - the day's outcome: departures, and how many left 15+ min late or were cancelled.

`search` finds the k most similar earlier days at the same airport. A day counts
only if it ended before the forecast was made (date < as_of's local date).
Methods: "embed" (bge-small, cosine), "bm25", "numeric" (nearest neighbors on
standardized weather and calendar numbers). Ties go to the most recent day.

Build: python -m murphy.pastdays  -> data/pastdays/{corpus.parquet, embed.npy}
"""

import ast
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

from murphy.datalayer.weather import AIRPORTS
from murphy.forecast.features import TAF_MAX_HORIZON_H

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pastdays"
TAF = ROOT / "data" / "weather" / "taf.parquet"
FLIGHTS = ROOT / "data" / "flights"
YEARS = (2023, 2024, 2025)  # 2026 is sealed (PLAN step 8)
WINDOW = (6, 22)            # local hours the summary covers
LATE_MIN = 15
K = 10
MODEL = "BAAI/bge-small-en-v1.5"
NO_CEILING = 12000          # ft, used when no BKN/OVC/VV layer is forecast
PREVAILING = ("initial", "FM")

INTENSITY = {"-": "light", "+": "heavy"}
DESCRIPTOR = {"VC": "nearby", "MI": "shallow", "BC": "patchy", "DR": "drifting",
              "BL": "blowing", "SH": "showers of", "TS": "thunderstorms with", "FZ": "freezing"}
PHENOMENON = {"RA": "rain", "SN": "snow", "DZ": "drizzle", "PL": "ice pellets", "GR": "hail",
              "GS": "small hail", "SG": "snow grains", "IC": "ice crystals", "UP": "unknown precipitation",
              "BR": "mist", "FG": "fog", "HZ": "haze", "FU": "smoke", "DU": "dust", "SA": "sand",
              "SQ": "squalls", "FC": "funnel cloud", "SS": "sandstorm", "DS": "duststorm", "PO": "dust whirls",
              "VA": "volcanic ash"}
FLAGS = ("TS", "SN", "RA", "FZ", "FG", "BR", "PL", "DZ")


def wx_words(code: str) -> str:
    """'-SHRA' -> 'light showers of rain'; 'VCTS' -> 'nearby thunderstorms'."""
    words, c = [], code
    if c[:1] in INTENSITY:
        words.append(INTENSITY[c[0]]); c = c[1:]
    while c[:2] in DESCRIPTOR:
        words.append(DESCRIPTOR[c[:2]]); c = c[2:]
    while c[:2] in PHENOMENON:
        words.append(PHENOMENON[c[:2]]); c = c[2:]
    if c:                      # unknown remainder: keep the code, never drop it
        words.append(c)
    text = " ".join(words)
    return text.removesuffix(" with").removesuffix(" of")


def _list(v) -> list:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return list(ast.literal_eval(v)) if isinstance(v, str) else list(v)


def day_weather(periods: pd.DataFrame) -> dict:
    """Summarize TAF periods (already cut to the window) into a few fields.

    Prevailing periods (initial, FM) give the numbers and the main weather.
    TEMPO and PROB30 weather goes to `at_times` only.
    """
    prev = periods[periods.period_type.isin(PREVAILING)]
    temp = periods[~periods.period_type.isin(PREVAILING)]
    ceil = NO_CEILING
    for skyc, skyl in zip(prev.skyc, prev.skyl):
        for c, h in zip(_list(skyc), _list(skyl)):
            if c in ("BKN", "OVC", "VV") and h is not None:
                ceil = min(ceil, int(h))
    wx = sorted({w for v in prev.presentwx for w in _list(v)})
    at_times = sorted({w for v in temp.presentwx for w in _list(v)} - set(wx))
    gust = prev.gust.max()
    return {
        "vis_min": float(prev.visibility.min()) if prev.visibility.notna().any() else np.nan,
        "ceil_min": ceil,
        "wind_max": float(prev.sknt.max()) if prev.sknt.notna().any() else np.nan,
        "gust_max": 0.0 if pd.isna(gust) else float(gust),
        "wx": wx,
        "at_times": at_times,
    }


def describe(airport: str, day: date, w: dict) -> str:
    """The sentence a day is searched by. Every word comes from a field."""
    parts = [f"{airport}, a {day:%A} in {day:%B}."]
    main = [wx_words(c) for c in w["wx"]]
    parts.append((", ".join(main).capitalize() + ".") if main else "No significant weather.")
    if w["at_times"]:
        parts.append("At times " + ", ".join(wx_words(c) for c in w["at_times"]) + ".")
    if w["ceil_min"] >= NO_CEILING:
        parts.append("No low clouds.")
    else:
        parts.append(f"{'Low clouds' if w['ceil_min'] < 1000 else 'Clouds'} at {w['ceil_min']} ft.")
    v = w["vis_min"]
    if pd.notna(v):
        parts.append("Visibility over 6 miles." if v > 6 else
                     f"{'Poor visibility' if v < 3 else 'Visibility'} {v:g} mile{'' if v == 1 else 's'}.")
    if pd.notna(w["wind_max"]):
        windy = w["wind_max"] >= 20 or w["gust_max"] >= 30
        mph = 1.15078  # the TAF gives knots; the sentence says mph, which travelers read. Thresholds stay in knots.
        s = f"{'Strong wind' if windy else 'Wind'} up to {w['wind_max'] * mph:.0f} mph"
        parts.append(s + (f", gusts {w['gust_max'] * mph:.0f} mph." if w["gust_max"] else "."))
    return " ".join(parts)


def numeric(row) -> list:
    """Numbers for the nearest-neighbor comparison: weather, flags, calendar."""
    d = row["date"]
    codes = " ".join(list(row["wx"]) + list(row["at_times"]))
    return [row["vis_min"], row["ceil_min"], row["wind_max"], row["gust_max"],
            *[float(f in codes) for f in FLAGS],
            np.sin(2 * np.pi * d.month / 12), np.cos(2 * np.pi * d.month / 12),
            np.sin(2 * np.pi * d.weekday() / 7), np.cos(2 * np.pi * d.weekday() / 7)]


def with_ends(product: pd.DataFrame) -> pd.DataFrame:
    """Give prevailing periods an end, as forecast/features.py reads them: each
    runs until the next prevailing period starts; the last one until issue +
    TAF_MAX_HORIZON_H. TEMPO and PROB30 periods keep their own end."""
    p = product.copy()
    prev = p.period_type.isin(PREVAILING)
    starts = np.sort(p.loc[prev, "period_start_utc"].unique())
    cap = p.issued_utc.max() + pd.Timedelta(hours=TAF_MAX_HORIZON_H)
    nxt = {s: (starts[i + 1] if i + 1 < len(starts) else cap) for i, s in enumerate(starts)}
    p.loc[prev, "period_end_utc"] = p.loc[prev, "period_start_utc"].map(nxt)
    return p


def local_utc(day: date, hour: int, tz: str) -> datetime:
    """`hour`:00 local on `day`, as naive UTC (the data's convention)."""
    return datetime.combine(day, time(hour), ZoneInfo(tz)).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def _cutoffs(airport: str, tz: str) -> pd.DataFrame:
    """Each local date with its as-of (local midnight) and window, in UTC."""
    days = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31").date
    utc = lambda d, h: local_utc(d, h, tz)
    return pd.DataFrame({"airport": airport, "date": days,
                         "as_of": [utc(d, 0) for d in days],
                         "win_start": [utc(d, WINDOW[0]) for d in days],
                         "win_end": [utc(d, WINDOW[1]) for d in days]})


def outcomes() -> pd.DataFrame:
    """Per airport and local date: departures, and how many were late or cancelled."""
    files = [str(FLIGHTS / f"{y}_{m:02d}.parquet") for y in YEARS for m in range(1, 13)]
    sql = f"""
        SELECT origin AS airport, flight_date AS date, count(*) AS n_dep,
               sum(CASE WHEN cancelled OR dep_delay_min >= {LATE_MIN} THEN 1 ELSE 0 END) AS n_late
        FROM read_parquet($f) WHERE duplicate_flag = 'N' AND origin IN {tuple(AIRPORTS)}
        GROUP BY 1, 2
    """
    df = duckdb.sql(sql, params={"f": files}).df()
    df["date"] = pd.to_datetime(df.date).dt.date
    return df


def build_corpus(taf_path: Path = TAF) -> pd.DataFrame:
    tz = pd.read_parquet(ROOT / "data" / "airports.parquet").set_index("airport").tz
    taf = pd.read_parquet(taf_path)
    rows = []
    for a in AIRPORTS:
        t = taf[taf.airport == a]
        products = t[["known_at"]].drop_duplicates().sort_values("known_at")
        cut = _cutoffs(a, tz[a]).sort_values("as_of")
        # latest known_at strictly before as_of, as asof.taf_as_of does
        cut = pd.merge_asof(cut, products, left_on="as_of", right_on="known_at",
                            allow_exact_matches=False)
        by_known = dict(tuple(t.groupby("known_at")))
        for r in cut.itertuples():
            base = {"airport": a, "date": r.date, "as_of": r.as_of, "taf_known_at": r.known_at}
            if pd.isna(r.known_at):
                rows.append(base | {"has_taf": False}); continue
            p = with_ends(by_known[r.known_at])
            p = p[(p.period_start_utc < r.win_end) & (p.period_end_utc > r.win_start)]
            if p.empty:
                rows.append(base | {"has_taf": False}); continue
            w = day_weather(p)
            rows.append(base | w | {"has_taf": True, "text": describe(a, r.date, w),
                                    "product_id": p.product_id.iloc[0]})
    corpus = pd.DataFrame(rows).merge(outcomes(), on=["airport", "date"], how="left")
    corpus["n_dep"] = corpus.n_dep.fillna(0).astype(int)
    corpus["n_late"] = corpus.n_late.fillna(0).astype(int)
    corpus["late_share"] = corpus.n_late / corpus.n_dep.where(corpus.n_dep > 0)
    return corpus.reset_index(drop=True)


# ---------------------------------------------------------------- search

def tokens(text: str) -> list[str]:
    return text.lower().replace(",", " ").replace(".", " ").split()


class Index:
    """All three methods over one corpus. Embeddings are passed in or computed."""

    def __init__(self, corpus: pd.DataFrame, embed: np.ndarray | None = None):
        from rank_bm25 import BM25Okapi
        keep = (corpus.has_taf & corpus.late_share.notna()).to_numpy()
        self.c = corpus[keep].reset_index(drop=True)
        self.embed = embed[keep] if embed is not None else None
        self._bm25 = {a: (BM25Okapi([tokens(t) for t in g.text]), g.index.to_numpy())
                      for a, g in self.c.groupby("airport")}
        X = np.array([numeric(r) for _, r in self.c.iterrows()], dtype=float)
        self._fill = np.nanmedian(X, axis=0)          # a missing field gets the median
        X = np.where(np.isnan(X), self._fill, X)
        self._mu, sd = X.mean(0), X.std(0)
        self._sd = np.where(sd > 0, sd, 1)
        self.X = (X - self._mu) / self._sd
        self._ord = np.array([d.toordinal() for d in self.c.date])
        self._model = None

    def _encode(self, texts):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL)
        return self._model.encode(texts, normalize_embeddings=True, batch_size=64)

    def scores(self, method: str, airport: str, text: str, nums: list, cand: np.ndarray,
               qvec: np.ndarray | None = None) -> np.ndarray:
        if method == "embed":
            q = qvec if qvec is not None else self._encode([text])[0]
            return self.embed[cand] @ q
        if method == "bm25":
            bm, idx = self._bm25[airport]
            s = pd.Series(bm.get_scores(tokens(text)), index=idx)
            return s.loc[cand].to_numpy()
        if method == "numeric":
            q = np.asarray(nums, dtype=float)
            q = (np.where(np.isnan(q), self._fill, q) - self._mu) / self._sd
            return -np.linalg.norm(self.X[cand] - q, axis=1)
        raise ValueError(f"unknown method {method!r}")

    def search(self, airport: str, before: date, text: str, nums: list, method: str = "embed",
               k: int = K, qvec: np.ndarray | None = None) -> pd.DataFrame:
        """k most similar days at `airport` dated before `before`. Ties: most recent first."""
        c = self.c
        cand = np.flatnonzero((c.airport == airport).to_numpy() & (self._ord < before.toordinal()))
        if len(cand) == 0:
            return c.iloc[[]].assign(score=[])
        s = self.scores(method, airport, text, nums, cand, qvec)
        order = np.lexsort((-self._ord[cand], -s))[:k]
        return c.iloc[cand[order]].assign(score=s[order])


# ---------------------------------------------------------------- one trip (the app)

# The app searches with plain numbers (8 Nov 2026): closest in weather and calendar, measured
# as distance. Thin above the 2025 95th percentile of the mean distance of the k days.
METHOD = "numeric"
THIN_ABOVE = 3.16  # 2025 95th percentile, scripts/pastdays_eval.py (re-derived for distance)


def similar(ix: Index, airport: str, day: date, as_of: datetime, tz: str, taf_as_of=None) -> dict:
    """Similar earlier days for a flight leaving `airport` on local date `day`, forecast at
    naive-UTC `as_of`. The TAF is the one known at as_of, read over the day's window.
    Only days that ended before as_of count. Answer + evidence + provenance + uncertainty."""
    from datetime import timezone
    if taf_as_of is None:
        from murphy.datalayer.asof import taf_as_of
    out = {"status": "ok", "answer": None, "query": None, "days": [],
           "provenance": {"method": f"nearest neighbors on standardized forecast numbers (visibility, cloud base, wind, "
                                    f"gusts, weather types) and calendar (month, weekday), same airport, top {K}",
                          "corpus": f"{airport} airport-days {YEARS[0]}-{YEARS[-1]}, before the forecast's local date",
                          "outcome": f"share of departures {LATE_MIN}+ min late or cancelled (BTS)",
                          "weather": "IEM TAF as issued"},
           "uncertainty": {"mean_distance": None, "thin_above": THIN_ABOVE, "reason": None}}
    if airport not in AIRPORTS:
        return out | {"status": "no_weather"}
    before = pd.Timestamp(as_of).tz_localize("UTC").tz_convert(tz).date()
    taf = taf_as_of(as_of.replace(tzinfo=timezone.utc), airport)
    if not taf.empty:
        taf = with_ends(taf)
        taf = taf[(taf.period_start_utc < local_utc(day, WINDOW[1], tz)) &
                  (taf.period_end_utc > local_utc(day, WINDOW[0], tz))]
    if taf.empty:
        return out | {"status": "thin", "uncertainty": out["uncertainty"] | {"reason": "No forecast (TAF) covers this day yet."}}
    w = day_weather(taf)
    text = describe(airport, day, w)
    out["query"] = {"text": text, "taf_product_id": taf.product_id.iloc[0], "taf_known_at": str(taf.known_at.iloc[0])}
    hit = ix.search(airport, before, text, numeric({"date": day, **w}), METHOD)
    out["days"] = [{"date": str(r.date), "text": r.text, "late_share": float(r.late_share), "n_dep": int(r.n_dep),
                    "n_late": int(r.n_late), "distance": -float(r.score), "taf_product_id": r.product_id}
                   for r in hit.itertuples()]
    dist = -float(hit.score.mean()) if len(hit) else None
    out["uncertainty"]["mean_distance"] = dist
    if len(hit) < K:
        out["status"], out["uncertainty"]["reason"] = "thin", f"Only {len(hit)} earlier days at {airport}."
    elif dist > THIN_ABOVE:
        out["status"], out["uncertainty"]["reason"] = "thin", "No close past days: the most similar ones differ from this forecast."
    else:
        out["answer"] = {"late_share": float(hit.late_share.mean()),
                         "range": [float(hit.late_share.min()), float(hit.late_share.max())], "days": len(hit)}
    return out


def build(out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus()
    corpus.to_parquet(out / "corpus.parquet")
    from sentence_transformers import SentenceTransformer
    ok = corpus[corpus.has_taf]
    vec = SentenceTransformer(MODEL).encode(ok.text.tolist(), normalize_embeddings=True,
                                            batch_size=64, show_progress_bar=True)
    full = np.zeros((len(corpus), vec.shape[1]), dtype=np.float32)
    full[ok.index] = vec
    np.save(out / "embed.npy", full)
    print(f"{len(corpus)} airport-days, {len(ok)} with a TAF -> {out}")


if __name__ == "__main__":
    build()
