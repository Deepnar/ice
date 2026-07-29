"""C16 — "is this enough?", asked of the QUESTION rather than of the other fragments.

Every stopping rule ICE nearly shipped compared candidates to each other: a
score floor, a knee on the score curve, a word-overlap novelty check. All three
answer *"is this fragment as good as the others?"* — a ranking question — when
the one that matters is *"do I now have enough to answer this?"*, which is
about the selected set versus the question. And all three go inert exactly
where the store is worth having: in a mature memory everything reads ~0.8
similar to a question about that memory, so a relative floor keeps all 59
candidates and a knee correctly finds no knee.

The word-overlap check was worse than inert — it was a bet on writing
convention, which CLAUDE.md's standing rule forbids: the same fact written two
ways would have read as novel, and two unrelated fragments sharing vocabulary
would have read as duplicates.

**What this does instead.** The question is already a vector. Every candidate is
a vector. Start with the question as "what still needs answering", repeatedly
take the candidate covering most of what is left, and subtract what it covered:

    r <- q
    loop:
        gain(f) = <f_hat, r>            how much of the remainder this covers
        best    = argmax gain
        stop if gain(best) < min_gain           nothing adds anything
        select best
        r <- r - <r, f_hat> f_hat               subtract what it covered
        stop if the coverage curve has bent     (see _knee)

Three properties fall out, none of which needed a special case:

* **Redundancy is free.** A fragment restating something already selected points
  along the direction just subtracted, so its gain is ~0. No words involved, so
  the same fact phrased differently is still caught.
* **The mature-store case resolves correctly.** 59 restatements: the residual
  goes near-orthogonal after two or three picks and the rest contribute nothing.
  59 turns genuinely covering different aspects: each keeps contributing and all
  are kept. The method separates "59 relevant turns" from "59 ways of saying one
  thing", which is the distinction that actually matters.
* **Legs become comparable.** One geometry for all eight retrieval mechanisms,
  so the fused RRF score goes back to doing only what it is good at — producing
  candidates — instead of being asked to rank across legs it cannot compare.

**What it does NOT do, stated plainly.** Coverage measures *direction*, never
*quality*. If the store holds nothing relevant, the best candidate still points
partly along the question — everything does — and this loop will happily pick
the best of a bad set. That is what `set_floor_passes` is for, and it is the
only arm here that can say "there is nothing worth injecting". Embedding
geometry is also a proxy for meaning, not proof of an answer: a fragment can be
perfectly on-topic and factually stale.
"""

import numpy as np
import structlog

logger = structlog.get_logger("ice.retrieval.coverage")


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def set_floor_passes(scores, floor: float) -> bool:
    """The set-level quality gate: is ANY of this worth injecting?

    Decided once, about the whole candidate set, before selection — because
    coverage cannot answer it. A per-fragment floor was tried and rejected: on
    raw RRF (`weight / (60 + rank)`) the spread from rank 1 to rank 20 is about
    1.3x, so a relative per-fragment floor is inert by construction. Asked of
    the BEST candidate it is a real question with a real answer.
    """
    if not scores:
        return False
    return max(scores) >= floor


def _knee(gains, min_prominence: float, min_keep: int) -> int:
    """Where the cumulative-coverage curve bends, or -1 for "no bend".

    Kneedle's max-distance-from-chord on the cumulative gain curve. This is the
    threshold-free version of "this much is enough": a question needing one
    fact bends at 1, a question needing broad context bends at 12, and neither
    needs a number chosen in advance.

    ⚠ A knee detector on a curve with no knee is a random truncator, so a bend
    must be at least `min_prominence` of the chord's span to count, and it can
    never cut below `min_keep`.
    """
    n = len(gains)
    if n <= min_keep:
        return -1
    cum = np.cumsum(np.asarray(gains, dtype=np.float64))
    total = float(cum[-1])
    if total <= 1e-9:
        return -1
    y = cum / total                       # 0..1
    x = np.arange(n, dtype=np.float64) / max(1, n - 1)
    # Distance from the straight line joining first and last point.
    dist = y - (x * (y[-1] - y[0]) + y[0])
    idx = int(np.argmax(dist))
    if float(dist[idx]) < min_prominence:
        return -1                          # no bend worth calling a knee
    return max(min_keep, idx + 1)


def select_by_coverage(query_vec, candidates, *, alpha: float = 0.7,
                       min_gain: float = 0.02, min_keep: int = 2,
                       max_keep: int = 40, knee_enabled: bool = True,
                       knee_min_prominence: float = 0.08):
    """Greedy residual selection.

    `candidates` is a list of ``(item, vector_or_None, score)``. Items whose
    vector is None are **admitted without competing** — a fragment ICE cannot
    place in the space (a rendered timeline, a codex payload with no stored
    embedding) must not be silently dropped for lacking a coordinate. They are
    returned first and flagged, so the count of them is visible rather than
    inferred.

    Returns ``(selected_items, record)`` where `record` carries, per candidate,
    the gain it offered and the reason it was or was not taken. That record is
    what makes the decision auditable later without re-running retrieval.
    """
    unplaced = [(i, it) for i, (it, v, _s) in enumerate(candidates) if v is None]
    placed = [(i, it, _unit(v), s) for i, (it, v, s) in enumerate(candidates)
              if v is not None]

    record = {"n_candidates": len(candidates), "n_unplaced": len(unplaced),
              "gains": [], "stop_reason": None, "knee_at": None}
    selected = [it for _i, it in unplaced]

    if not placed:
        record["stop_reason"] = "no_vectors"
        return selected, record

    r = _unit(query_vec)
    if float(np.linalg.norm(r)) <= 1e-12:
        # No usable question vector — coverage cannot rank, so do not pretend
        # to. Everything passes through in its incoming order.
        record["stop_reason"] = "no_query_vector"
        return selected + [it for _i, it, _v, _s in placed], record

    # Rank-normalise the retrieval score WITHIN the candidate set so the blend
    # is scale-free: raw RRF values are not comparable across queries, and the
    # bonuses stretch them by up to 5x.
    order = sorted(range(len(placed)), key=lambda k: placed[k][3], reverse=True)
    rank_norm = [0.0] * len(placed)
    for rank, k in enumerate(order):
        rank_norm[k] = 1.0 - (rank / max(1, len(placed) - 1)) if len(placed) > 1 else 1.0

    remaining = list(range(len(placed)))
    chosen, gains = [], []

    while remaining and len(chosen) < max_keep:
        best_k, best_val, best_gain = None, -1.0, 0.0
        for k in remaining:
            gain = abs(float(np.dot(placed[k][2], r)))
            val = alpha * gain + (1.0 - alpha) * rank_norm[k]
            if val > best_val:
                best_k, best_val, best_gain = k, val, gain

        if best_gain < min_gain and len(chosen) >= min_keep:
            record["stop_reason"] = "min_gain"
            break

        chosen.append(best_k)
        gains.append(best_gain)
        remaining.remove(best_k)

        # Subtract the component the chosen fragment covered. What is left is,
        # literally, the part of the question nothing selected has spoken to.
        f = placed[best_k][2]
        r = r - float(np.dot(r, f)) * f
        n = float(np.linalg.norm(r))
        if n <= 1e-9:
            record["stop_reason"] = "residual_exhausted"
            break
        r = r / n

    if record["stop_reason"] is None:
        record["stop_reason"] = "max_keep" if len(chosen) >= max_keep else "exhausted"

    cut = -1
    if knee_enabled and len(chosen) > min_keep:
        cut = _knee(gains, knee_min_prominence, min_keep)
        if cut > 0:
            record["knee_at"] = cut
            record["stop_reason"] = "knee"
            chosen = chosen[:cut]
            gains = gains[:cut]

    record["gains"] = [round(g, 5) for g in gains]
    record["n_selected"] = len(chosen) + len(unplaced)
    selected += [placed[k][1] for k in chosen]
    logger.info("coverage_selected", candidates=len(candidates),
                selected=record["n_selected"], unplaced=len(unplaced),
                stop=record["stop_reason"], knee_at=record["knee_at"])
    return selected, record
