"""Standalone behavioral test for B2's memory-retrieval decision.

Run: uv run python tests/test_memory_decision.py
Pure-logic test — no DB, no model load (exercises the decision + the
classifier's confidence finalization directly).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.config import settings
from src.classifier.schemas import ClassificationResult
from src.api.memory_decision import (
    decide_memory_retrieval, memory_pressure, estimate_recent_window_tokens,
    derive_total_budget, _sigmoid, _logit,
)
from src.classifier.schema import (CONTEXT_RELIANCE, finalize_context_scalars,
                                   load_schema, load_v1_schema)

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


def mk(p_ltm=0.1, topics=None, intents=None, max_conf=0.99, ctx_conf=0.8,
       reference=False, ctx="Zero_Shot", prompt="hello there", p_temporal=0.0):
    r = ClassificationResult(
        topic_tags=topics or ["Software_&_Tech"],
        intent_tags=intents or ["Factual_Retrieval"],
        context_reliance=ctx,
        raw_probs=[0.0] * 25,
        max_confidence=max_conf,
        prompt=prompt,
    )
    r.p_ltm = p_ltm
    r.ctx_confidence = ctx_conf
    r.reference_signal = reference
    r.p_temporal = p_temporal
    return r


def decide(r, turns=1, tokens=200.0, timescope_mode=None):
    return decide_memory_retrieval(r, turn_count=turns, total_tokens=tokens,
                                   settings=settings, timescope_mode=timescope_mode)


print("── math sanity ──")
check("sigmoid(0)=0.5", abs(_sigmoid(0.0) - 0.5) < 1e-9)
check("logit/sigmoid roundtrip", abs(_sigmoid(_logit(0.73)) - 0.73) < 1e-6)
check("pressure neutral (0.5) when convo fits window",
      abs(memory_pressure(500, estimate_recent_window_tokens(3), 2000, 4000) - 0.5) < 1e-6)
check("pressure high when history far beyond window",
      memory_pressure(40000, estimate_recent_window_tokens(200), 2000, 4000) > 0.9)

print("── the core behavior: prefers LTM but doesn't force ──")
# 1. Confident Zero_Shot, short convo → survives (no retrieval).
d = decide(mk(p_ltm=0.03, max_conf=0.99), turns=2, tokens=300)
check("confident zero-shot short convo → NO retrieve", d.retrieve is False)

# 2. Confident LTM → retrieve.
d = decide(mk(p_ltm=0.95, max_conf=0.99), turns=2, tokens=300)
check("confident LTM → retrieve", d.retrieve is True)

# 3. Neutral classifier → bias tips toward retrieval (prefers LTM).
d = decide(mk(p_ltm=0.5, max_conf=0.99), turns=2, tokens=300)
check("neutral p_ltm → retrieve (prefers memory)", d.retrieve is True)

# 4. Long convo with history beyond the window flips a lowish p_ltm.
short = decide(mk(p_ltm=0.25, max_conf=0.99), turns=2, tokens=300)
long = decide(mk(p_ltm=0.25, max_conf=0.99), turns=120, tokens=45000)
check("same p_ltm: short convo no-retrieve", short.retrieve is False)
check("same p_ltm: long convo (beyond-window) → retrieve", long.retrieve is True)
check("memory pressure higher in long convo", long.breakdown["p_len"] > short.breakdown["p_len"])

print("── old hard signals are now bumps, not slams ──")
base = mk(p_ltm=0.28, max_conf=0.99)
d_base = decide(base, turns=2, tokens=300)
d_creative = decide(mk(p_ltm=0.28, topics=["Creative_&_Media"], max_conf=0.99), turns=2, tokens=300)
check("borderline non-creative → no retrieve", d_base.retrieve is False)
check("creative bump flips borderline → retrieve", d_creative.retrieve is True)
check("creative flagged in breakdown", d_creative.breakdown["creative"] is True)

d_ref = decide(mk(p_ltm=0.28, reference=True, max_conf=0.99), turns=2, tokens=300)
check("anaphora (reference_signal) bump → retrieve", d_ref.retrieve is True)

d_word = decide(mk(p_ltm=0.30, max_conf=0.99, prompt="can you fix this again like before"), turns=2, tokens=300)
check("referential words detected", d_word.breakdown["referential"] is True)

d_lowconf = decide(mk(p_ltm=0.28, max_conf=0.40), turns=2, tokens=300)
check("low topic/intent confidence safety bump → retrieve", d_lowconf.retrieve is True)
check("low_confidence flagged", d_lowconf.breakdown["low_confidence"] is True)

print("── nothing is *forced*: enough negative signal still says no ──")
d = decide(mk(p_ltm=0.02, topics=["Creative_&_Media"], max_conf=0.99), turns=1, tokens=100)
check("very confident zero-shot survives even a creative bump", d.retrieve is False)

print("── classifier confidence finalization (ML head vs DI3 path) ──")
# B1: the derivation is pure label logic (schema.finalize_context_scalars), so
# it needs a schema but no checkpoint. Both generations are exercised.
V1 = load_v1_schema()
V2 = load_schema()

# v1 ML path: softmax ctx probs at indices 22..24 = [Zero_Shot, LTM, RTS]
r = ClassificationResult([], [], "Long_Term_Memory",
                         [0.0] * 22 + [0.1, 0.7, 0.2], 0.7, "x")
finalize_context_scalars(r, V1)
check("v1 path p_ltm read from ctx[1]", abs(r.p_ltm - 0.7) < 1e-9)
check("v1 path p_rts read from ctx[2]", abs(r.p_rts - 0.2) < 1e-9)
check("v1 path ctx_confidence = top1-top2", abs(r.ctx_confidence - 0.5) < 1e-9)

# DI3 path: raw_probs all zero → derive prior from the label.
r2 = ClassificationResult(["Null_Noise"], ["Casual_Banter"], "Zero_Shot",
                          [0.0] * 25, 0.95, "asdf")
finalize_context_scalars(r2, V1)
check("DI3 zero-shot prior p_ltm low", r2.p_ltm < 0.2)
r3 = ClassificationResult([], [], "Long_Term_Memory", [0.0] * 25, 0.7, "it")
finalize_context_scalars(r3, V1)
check("DI3 LTM prior p_ltm high", r3.p_ltm > 0.8)

print("── B1 D6: v2's four sigmoids derive the same scalars B2 consumes ──")
# ctx head order: [Needs_Memory, Temporal_Recall, Needs_Live_Info, High_Complexity].
# The pad is the context head's OFFSET, read from the schema rather than written
# out: it moved 24 -> 23 when Codebase_Query was dropped, and a literal here just
# silently mis-slices into the intent head.
CTX0 = V2.head(CONTEXT_RELIANCE).offset
r4 = ClassificationResult([], [], "Zero_Shot",
                          [0.0] * CTX0 + [0.9, 0.8, 0.1, 0.2], 0.9, "what did I say in March")
finalize_context_scalars(r4, V2)
check("v2 p_ltm = the Needs_Memory sigmoid", abs(r4.p_ltm - 0.9) < 1e-9)
check("v2 derived context_reliance = Long_Term_Memory", r4.context_reliance == "Long_Term_Memory")
check("v2 p_temporal surfaced", abs(r4.p_temporal - 0.8) < 1e-9)
check("v2 p_complex surfaced", abs(r4.p_complex - 0.2) < 1e-9)

# The orthogonality the 3-way could not express: memory AND live info together.
r5 = ClassificationResult([], [], "Zero_Shot",
                          [0.0] * CTX0 + [0.85, 0.1, 0.80, 0.3], 0.85, "is my usual broker still cheapest")
finalize_context_scalars(r5, V2)
check("v2 memory+live coexist (both high)", r5.p_ltm > 0.8 and r5.p_rts > 0.75)
check("v2 low margin when both fire", r5.ctx_confidence < 0.1)

# All-low reliance is the DERIVED Zero_Shot state (no such label exists in v2).
r6 = ClassificationResult([], [], "Long_Term_Memory",
                          [0.0] * CTX0 + [0.05, 0.02, 0.03, 0.1], 0.9, "write a haiku")
finalize_context_scalars(r6, V2)
check("v2 all-low reliance derives Zero_Shot", r6.context_reliance == "Zero_Shot")

print("── B1 D7: Temporal_Recall label == detector evidence (OR, never twice) ──")
# Straddle the LIVE threshold rather than hardcoding a probability. The value is
# Z1-prep's to tune and was already raised 0.6 → 0.85 once (the head fires as a
# shadow of Needs_Memory); what D7 asserts is the OR mechanism, which must hold at
# whatever threshold is configured.
_t = settings.temporal_label_threshold
_hot, _cold = min(1.0, _t + 0.1), max(0.0, _t - 0.1)
d_label = decide(mk(p_ltm=0.3, p_temporal=_hot), turns=2, tokens=300)
d_detector = decide(mk(p_ltm=0.3), turns=2, tokens=300, timescope_mode="range")
d_both = decide(mk(p_ltm=0.3, p_temporal=_hot), turns=2, tokens=300, timescope_mode="range")
check("temporal label alone fires the bump", d_label.breakdown["temporal_label"] is True)
check("label-only == detector-only bump size",
      abs(d_label.p_need_mem - d_detector.p_need_mem) < 1e-9)
check("both together is not double-counted",
      abs(d_both.p_need_mem - d_detector.p_need_mem) < 1e-9)
d_quiet = decide(mk(p_ltm=0.3, p_temporal=_cold), turns=2, tokens=300)
check("sub-threshold p_temporal does not bump", d_quiet.p_need_mem < d_label.p_need_mem)

print("── C16: model-aware total budget ──")
check("8k model → 6k budget (0.75×)", derive_total_budget(8192, settings) == int(8192 * 0.75))
check("32k model → 24k budget", derive_total_budget(32768, settings) == int(32768 * 0.75))
check("128k model → clamped to max guardrail", derive_total_budget(131072, settings) == settings.context_budget_max)
check("tiny 2k model → clamped to min floor", derive_total_budget(2048, settings) == settings.context_budget_min)
check("unknown model (None) → fallback 23k", derive_total_budget(None, settings) == settings.context_budget_fallback)
check("unknown model (0) → fallback 23k", derive_total_budget(0, settings) == settings.context_budget_fallback)
check("recent window scales with model budget",
      estimate_recent_window_tokens(3, derive_total_budget(8192, settings))
      < estimate_recent_window_tokens(3, derive_total_budget(32768, settings)))
check("recent window default = legacy 23k mirror",
      abs(estimate_recent_window_tokens(3) - 0.3 * (23_000 - 1_800)) < 1e-6)

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
