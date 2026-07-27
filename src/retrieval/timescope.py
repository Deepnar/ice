"""TimeScope — deterministic temporal-query understanding (Track T, T2).

Parses temporal intent out of a prompt into a ``TimeScope`` — mode
(``current`` | ``as_of`` | ``range`` | ``evolution``) plus a UTC window — that
travels through retrieval inside the existing ``scope`` dict
(``scope["timescope"]``, D1). Detection is pure, regex/lexicon-based and
LLM-free (<1 ms, A6 philosophy): deterministic parsing is strictly more
debuggable, and the lexicon is the extension point.

Joint gate (A4's joint-signal lesson): a resolvable time expression alone
never flips the mode — it must co-occur with a recall-shaped prompt
(question mark / interrogative opener / reference signal / p_ltm ≥ 0.5).
A bare "yesterday" in a content clause stays ``current``; a false positive
would inject stale context into a normal answer.

Vague pasts ("a while back", "long ago") are deliberately NOT resolved into
windows — an invented window silently hides the memories outside it.

B1 seam: ``detect_timescope`` accepts ``intent_tags`` and returns
``evidence`` so a future temporal-recall classifier label can join the same
gate; window *resolution* stays deterministic regardless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence, Tuple

from src.api.config import settings

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeScope:
    mode: str = "current"           # "current" | "as_of" | "range" | "evolution"
    t0: Optional[datetime] = None   # UTC, inclusive
    t1: Optional[datetime] = None   # UTC, exclusive
    matched_text: str = ""          # expression(s) that fired (telemetry)
    evidence: tuple = ()            # signals that passed the gate (telemetry; B1 seam)


CURRENT = TimeScope(mode="current")


def to_scope_dict(ts: Optional[TimeScope]) -> Optional[dict]:
    """Serialize for scope["timescope"]; None for current (nothing to carry)."""
    if ts is None or ts.mode == "current":
        return None
    return {
        "mode": ts.mode,
        "t0": ts.t0.isoformat() if ts.t0 else None,
        "t1": ts.t1.isoformat() if ts.t1 else None,
    }


def from_scope(scope: Optional[dict]) -> TimeScope:
    """Parse scope.get("timescope") back into a TimeScope. Tolerant: anything
    missing or malformed degrades to CURRENT (never crash the read path)."""
    if not scope:
        return CURRENT
    d = scope.get("timescope")
    if not isinstance(d, dict):
        return CURRENT
    mode = d.get("mode")
    if mode not in ("as_of", "range", "evolution"):
        return CURRENT
    try:
        t0 = datetime.fromisoformat(d["t0"]) if d.get("t0") else None
        t1 = datetime.fromisoformat(d["t1"]) if d.get("t1") else None
    except (ValueError, TypeError, KeyError):
        return CURRENT
    if mode in ("as_of", "range") and not (t0 and t1):
        return CURRENT  # windowed modes are meaningless without a window
    return TimeScope(mode=mode, t0=t0, t1=t1)


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = (
    r"(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
)
_YEAR_RE = r"(199\d|20\d\d)"  # 1990–2099 guard: anything else is a number, not a date

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a": 1, "an": 1, "couple": 2, "few": 3,
}

# Words that legitimise a *bare* month name, year, or season as a time
# expression ("in march", "since 2024"). Guards against "you may want",
# "march forward", "spring cleaning", and prose numbers. vs/compare anchor the
# "2024 vs 2025" comparison case (§2.1 step 5).
_ANCHOR_WORDS = {"in", "during", "last", "since", "before", "until", "from",
                 "by", "around", "of", "between", "and", "after",
                 "vs", "versus", "compare", "compared"}

_INTERROGATIVES = {"what", "when", "how", "show", "list", "tell", "remind",
                   "which", "where", "did", "was", "were"}

# Season spans (northern-hemisphere convention; start month, end month).
# Winter belongs to the year it *starts* in (winter 2025 = Dec 2025 – Mar 2026).
_SEASONS = {
    "spring": (3, 6), "summer": (6, 9), "autumn": (9, 12), "fall": (9, 12),
    "winter": (12, 3),
}

_EVOLUTION_CUES = [
    re.compile(r"\bhow\s+(?:did|has|have|does)\b.{0,80}?\b(?:evolv\w*|chang\w*|develop\w*|progress\w*)", re.I | re.S),
    re.compile(r"\b(?:evolution|history|progression|timeline)\s+of\b", re.I),
    re.compile(r"\bover\s+time\b", re.I),
    re.compile(r"\b(?:originally|at\s+first|initially|back\s+when)\b", re.I),
    re.compile(r"\bat\s+the\s+(?:start|beginning)\b", re.I),
    re.compile(r"\bwhat\s+did\s+(?:i|we)\s+(?:start|begin)\s+with\b", re.I),
]
# compare-cue fires only when ≥2 expressions resolved (both sides temporal)
_COMPARE_CUE = re.compile(r"\b(?:compare|compared|vs\.?|versus)\b", re.I)

_FENCED_CODE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`]*`")


# ---------------------------------------------------------------------------
# Window helpers (all windows are [t0, t1), UTC)
# ---------------------------------------------------------------------------


def _utc(y, m, d) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _month_start(y: int, m: int) -> datetime:
    return _utc(y, m, 1)


def _next_month(y: int, m: int) -> Tuple[int, int]:
    return (y + 1, 1) if m == 12 else (y, m + 1)


def _days(n: float) -> timedelta:
    return timedelta(days=n)


def _day_window(d: datetime) -> Tuple[datetime, datetime]:
    pad = _days(settings.timescope_pad_min_days)
    start = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return start - pad, start + _days(1) + pad


def _month_window(y: int, m: int) -> Tuple[datetime, datetime]:
    pad = _days(settings.timescope_pad_month_days)
    ny, nm = _next_month(y, m)
    return _month_start(y, m) - pad, _month_start(ny, nm) + pad


def _span_window(t0: datetime, t1: datetime) -> Tuple[datetime, datetime]:
    pad = _days(settings.timescope_pad_span_days)
    return t0 - pad, t1 + pad


def _year_window(y: int) -> Tuple[datetime, datetime]:
    pad = _days(settings.timescope_pad_year_days)
    return _utc(y, 1, 1) - pad, _utc(y + 1, 1, 1) + pad


def _relative_window(delta_days: float, unit: str, now: datetime) -> Tuple[datetime, datetime]:
    """N days/weeks/months/years ago → center ± granularity-scaled halfwidth."""
    center = now - _days(delta_days)
    if unit in ("day", "week"):
        half = min(max(settings.timescope_pad_min_days,
                       settings.timescope_rel_short_frac * delta_days),
                   settings.timescope_rel_short_cap_days)
    else:
        half = min(max(settings.timescope_pad_month_days,
                       settings.timescope_rel_long_frac * delta_days),
                   settings.timescope_rel_long_cap_days)
    return center - _days(half), center + _days(half)


def _recent_past_month(m: int, now: datetime) -> int:
    """Year of the most recent past occurrence of month *m* ("in march" when
    March hasn't happened yet this year → last year's March)."""
    return now.year if _month_start(now.year, m) <= now else now.year - 1


def _season_span(name: str, year: int) -> Tuple[datetime, datetime]:
    sm, em = _SEASONS[name]
    if name == "winter":
        return _utc(year, 12, 1), _utc(year + 1, 3, 1)
    return _utc(year, sm, 1), _utc(year, em, 1)


# ---------------------------------------------------------------------------
# Expression scanning
# ---------------------------------------------------------------------------


@dataclass
class _Expr:
    start: int
    end: int
    text: str
    t0: datetime
    t1: datetime
    kind: str  # "point" → as_of; "range" → range


def _prev_word(text: str, pos: int) -> str:
    m = re.search(r"([\w']+)[\s:,]*$", text[:pos])
    return m.group(1).lower() if m else ""


def _scan_expressions(text: str, now: datetime) -> List[_Expr]:
    """Find every resolvable time expression. Longest match wins on overlap."""
    found: List[_Expr] = []

    def add(m: re.Match, t0: datetime, t1: datetime, kind: str):
        found.append(_Expr(m.start(), m.end(), m.group(0).strip(), t0, t1, kind))

    # explicit ISO date
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1990 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                t0, t1 = _day_window(_utc(y, mo, d))
            except ValueError:
                continue
            add(m, t0, t1, "point")

    # month-name + year ("march 2025", optionally "in march 2025")
    for m in re.finditer(rf"\b{_MONTH_RE}\s+{_YEAR_RE}\b", text, re.I):
        mo, y = _MONTHS[m.group(1).lower()], int(m.group(2))
        t0, t1 = _month_window(y, mo)
        add(m, t0, t1, "point")

    # month-name + day (+ optional year): "march 5", "march 5th, 2025"
    for m in re.finditer(rf"\b{_MONTH_RE}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+{_YEAR_RE})?\b", text, re.I):
        mo, d = _MONTHS[m.group(1).lower()], int(m.group(2))
        if not 1 <= d <= 31:
            continue
        y = int(m.group(3)) if m.group(3) else _recent_past_month(mo, now)
        try:
            t0, t1 = _day_window(_utc(y, mo, d))
        except ValueError:
            continue
        add(m, t0, t1, "point")

    # "5th of march [2025]"
    for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH_RE}(?:\s+{_YEAR_RE})?\b", text, re.I):
        d, mo = int(m.group(1)), _MONTHS[m.group(2).lower()]
        if not 1 <= d <= 31:
            continue
        y = int(m.group(3)) if m.group(3) else _recent_past_month(mo, now)
        try:
            t0, t1 = _day_window(_utc(y, mo, d))
        except ValueError:
            continue
        add(m, t0, t1, "point")

    # quarters / halves
    for m in re.finditer(rf"\bq([1-4])\s+(?:of\s+)?{_YEAR_RE}\b", text, re.I):
        q, y = int(m.group(1)), int(m.group(2))
        s = _month_start(y, 3 * q - 2)
        ny, nm = (y + 1, 1) if q == 4 else (y, 3 * q + 1)
        t0, t1 = _span_window(s, _month_start(ny, nm))
        add(m, t0, t1, "range")
    for m in re.finditer(rf"\b(?:h([12])|(first|second)\s+half(?:\s+of)?)\s+{_YEAR_RE}\b", text, re.I):
        h = int(m.group(1)) if m.group(1) else (1 if m.group(2).lower() == "first" else 2)
        y = int(m.group(3))
        s, e = (_utc(y, 1, 1), _utc(y, 7, 1)) if h == 1 else (_utc(y, 7, 1), _utc(y + 1, 1, 1))
        t0, t1 = _span_window(s, e)
        add(m, t0, t1, "range")

    # early/mid/late + year → year-third (range); + month → month-third (point)
    for m in re.finditer(rf"\b(early|mid|late)\s+{_YEAR_RE}\b", text, re.I):
        part, y = m.group(1).lower(), int(m.group(2))
        spans = {"early": (_utc(y, 1, 1), _utc(y, 5, 1)),
                 "mid": (_utc(y, 5, 1), _utc(y, 9, 1)),
                 "late": (_utc(y, 9, 1), _utc(y + 1, 1, 1))}
        t0, t1 = _span_window(*spans[part])
        add(m, t0, t1, "range")
    for m in re.finditer(rf"\b(early|mid|late)\s+{_MONTH_RE}(?:\s+{_YEAR_RE})?\b", text, re.I):
        part, mo = m.group(1).lower(), _MONTHS[m.group(2).lower()]
        y = int(m.group(3)) if m.group(3) else _recent_past_month(mo, now)
        ny, nm = _next_month(y, mo)
        month_end = _month_start(ny, nm)
        thirds = {"early": (_month_start(y, mo), _month_start(y, mo) + _days(11)),
                  "mid": (_month_start(y, mo) + _days(10), _month_start(y, mo) + _days(21)),
                  "late": (_month_start(y, mo) + _days(20), month_end)}
        s, e = thirds[part]
        pad = _days(settings.timescope_pad_min_days)
        add(m, s - pad, e + pad, "point")

    # seasons ("last winter", "spring 2024", "in summer") — a bare season with
    # no year/"last"/anchor stays unmatched ("spring cleaning" is not a date).
    for m in re.finditer(rf"\b(last\s+)?(spring|summer|autumn|fall|winter)\b(?:\s+(?:of\s+)?{_YEAR_RE})?", text, re.I):
        last, name = bool(m.group(1)), m.group(2).lower()
        if not last and not m.group(3) and _prev_word(text, m.start()) not in _ANCHOR_WORDS:
            continue
        if m.group(3):
            year = int(m.group(3))
        else:
            year = now.year
            s, e = _season_span(name, year)
            if s > now or (last and e > now):
                year -= 1  # not started yet — or "last <season>" while inside it
        t0, t1 = _span_window(*_season_span(name, year))
        add(m, t0, t1, "range")

    # whole year with anchor word ("in 2024", "since 2023") — bare years in
    # prose ("python 2020 release") stay unmatched without the anchor.
    for m in re.finditer(rf"\b{_YEAR_RE}\b", text):
        if _prev_word(text, m.start()) in _ANCHOR_WORDS:
            t0, t1 = _year_window(int(m.group(1)))
            add(m, t0, t1, "range")

    # bare month with anchor word ("in march", "since january", "last july")
    for m in re.finditer(rf"\b{_MONTH_RE}\b(?!\s+\d)(?!-)", text, re.I):
        if _prev_word(text, m.start()) in _ANCHOR_WORDS:
            mo = _MONTHS[m.group(1).lower()]
            t0, t1 = _month_window(_recent_past_month(mo, now), mo)
            add(m, t0, t1, "point")

    # relative: "<N|word> days/weeks/months/years ago"
    rel = rf"\b(\d{{1,3}}|{'|'.join(_NUMBER_WORDS)})\s+(?:of\s+)?(day|week|month|year)s?\s+ago\b"
    for m in re.finditer(rel, text, re.I):
        raw = m.group(1).lower()
        n = _NUMBER_WORDS.get(raw, None)
        if n is None:
            n = int(raw)
        if n <= 0:
            continue
        unit = m.group(2).lower()
        delta = n * {"day": 1, "week": 7, "month": 30.44, "year": 365.25}[unit]
        t0, t1 = _relative_window(delta, unit, now)
        add(m, t0, t1, "point")

    # fixed relatives
    for m in re.finditer(r"\byesterday\b", text, re.I):
        t0, t1 = _day_window(now - _days(1))
        add(m, t0, t1, "point")
    for m in re.finditer(r"\bthe\s+other\s+day\b", text, re.I):
        add(m, now - _days(8), now, "point")  # "1–7 days ago", loosely padded
    for m in re.finditer(r"\blast\s+(week|month|year)\b", text, re.I):
        unit = m.group(1).lower()
        if unit == "week":
            t0, t1 = _relative_window(7, "week", now)
            kind = "point"
        elif unit == "month":
            py, pm = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
            t0, t1 = _month_window(py, pm)
            kind = "point"
        else:
            t0, t1 = _year_window(now.year - 1)
            kind = "range"
        add(m, t0, t1, kind)

    # longest-match-wins overlap resolution
    found.sort(key=lambda e: (-(e.end - e.start), e.start))
    kept: List[_Expr] = []
    for e in found:
        if all(e.end <= k.start or e.start >= k.end for k in kept):
            kept.append(e)
    kept.sort(key=lambda e: e.start)
    return kept


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_timescope(
    prompt: str,
    *,
    now: Optional[datetime] = None,
    intent_tags: Sequence[str] = (),
    p_ltm: float = 0.0,
) -> TimeScope:
    """Detect and resolve temporal intent. Pure (no DB, no LLM); ``now`` is
    injectable for tests. Returns CURRENT unless a resolvable expression or
    evolution cue co-occurs with a recall-shaped prompt (the joint gate)."""
    if not settings.timescope_enabled or not prompt:
        return CURRENT
    now = now or datetime.now(timezone.utc)

    # 1. Strip code — temporal words inside code are content, not intent.
    text = _INLINE_CODE.sub(" ", _FENCED_CODE.sub(" ", prompt))

    # 2/3. Scan expressions + evolution cues.
    exprs = _scan_expressions(text, now)
    cue_matches = [c.search(text) for c in _EVOLUTION_CUES]
    cue_texts = [m.group(0) for m in cue_matches if m]
    if len(exprs) >= 2 and _COMPARE_CUE.search(text):
        cue_texts.append("compare")
    evolution = bool(cue_texts)

    if not exprs and not evolution:
        return CURRENT

    # 4. Joint gate — expression/cue alone never flips the mode.
    stripped = text.strip().lower()
    first_word = stripped.split()[0] if stripped.split() else ""
    evidence: List[str] = [f"expr:{e.text}" for e in exprs] + [f"cue:{c}" for c in cue_texts]
    # D8 (2026-07-27) removed a fourth arm, `reference_signal`, when DI3 died.
    # It was not a loss: measured over 9,441 held-out rows it was carrying 49
    # non-current TimeScopes that the p_ltm arm below correctly refused, and
    # every one was a long pasted document ("Write a synopsis of the following
    # text…", p_ltm 0.00–0.16) whose *length* had accumulated enough "the"s to
    # clear DI3's reference density. Substituting memory_decision's
    # REFERENTIAL_WORDS was measured too and is worse still (527 non-current vs
    # 416), for the same reason — so the anaphora signal stays out of this gate.
    if "?" in text:
        evidence.append("gate:question_mark")
    elif first_word in _INTERROGATIVES:
        evidence.append("gate:interrogative")
    elif p_ltm >= 0.5:
        evidence.append("gate:p_ltm")
    else:
        return CURRENT  # recall shape missing — statement, not a memory query

    # 5. Wrappers + mode assignment.
    wrapped_range = False
    if len(exprs) == 1:
        e = exprs[0]
        prev = _prev_word(text, e.start)
        if prev == "since":
            exprs = [_Expr(e.start, e.end, f"since {e.text}", e.t0, now, "range")]
            wrapped_range = True
        elif prev == "before":
            exprs = [_Expr(e.start, e.end, f"before {e.text}",
                           _utc(2000, 1, 1), e.t0, "range")]
            wrapped_range = True
    elif len(exprs) == 2:
        a, b = exprs
        between = _prev_word(text, a.start) == "between"
        joiner = text[a.end:b.start].strip().lower()
        if between and joiner == "and":
            exprs = [_Expr(a.start, b.end, f"between {a.text} and {b.text}",
                           a.t0, b.t1, "range")]
            wrapped_range = True

    t0 = min(e.t0 for e in exprs) if exprs else None
    t1 = max(e.t1 for e in exprs) if exprs else None

    if evolution:
        mode = "evolution"
    elif wrapped_range or len(exprs) > 1 or exprs[0].kind == "range":
        mode = "range"
    else:
        mode = "as_of"

    # Guards: future = scheduling, not memory; windows never extend past now.
    if t0 and t1:
        if t0 >= t1:
            t0, t1 = t1, t0
        if t0 > now:
            return CURRENT
        if t1 > now:
            t1 = now

    matched = "; ".join([e.text for e in exprs] + cue_texts)[:120]
    return TimeScope(mode=mode, t0=t0, t1=t1,
                     matched_text=matched, evidence=tuple(evidence))
