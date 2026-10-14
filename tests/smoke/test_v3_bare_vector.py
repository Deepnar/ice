"""Bare-vector control: deterministic chunks, overlap skip, budget, span coverage."""

from experiments.v3_repair import bare_vector as bv


def test_windows_cover_text_with_overlap_and_keep_offsets():
    text = " ".join(f"w{i}" for i in range(10))
    spans = bv.windows(text, 4)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(text[a:b].split()[0].startswith("w") for a, b in spans)
    assert spans[1][0] < spans[0][1]  # 25% overlap
    assert bv.windows(text, None) == [(0, len(text))] == bv.windows(text, 50)
    assert bv.windows("   ", 4) == []


def test_select_ranks_by_cosine_skips_same_message_overlap_and_respects_budget():
    turns = [dict(turn_number=1, prompt="alpha beta gamma delta", response="port 7319 chosen"),
             dict(turn_number=2, prompt="unrelated words here", response="nothing")]
    chunks = bv.chunk_turns(turns, 2)
    vec = lambda t: [1.0, 0.0] if "7319" in t or "port" in t else [0.0, 1.0]
    index = bv.Index(lambda texts: [vec(t) for t in texts])
    chosen = bv.select(index, "which port", chunks, budget_tokens=1000, k=None, count=len)
    assert chosen[0][0].text in ("port 7319", "7319 chosen") and chosen[0][1] == 1.0
    picked = [c for c, _ in chosen if c.turn == 1 and c.role == "assistant"]
    assert all(not (a.start < b.end and b.start < a.end) for a in picked for b in picked if a is not b)
    assert sum(len(c.render()) for c, _ in bv.select(index, "which port", chunks, 40, None, len)) <= 40
    assert len(bv.select(index, "which port", chunks, 1000, 1, len)) == 1


def test_span_available_from_recent_turns_or_contiguous_chunks():
    a, b = bv.Chunk(3, "assistant", 0, 10, "x" * 10), bv.Chunk(3, "assistant", 8, 20, "y" * 12)
    span = dict(turn=3, role="assistant", start=5, end=18)
    assert bv.span_available(span, [(a, 1.0), (b, 0.9)], recent_from=10)
    assert not bv.span_available(span, [(a, 1.0)], recent_from=10)
    assert not bv.span_available(dict(span, role="user"), [(a, 1.0), (b, 0.9)], recent_from=10)
    assert bv.span_available(dict(span, turn=12), [], recent_from=10)
