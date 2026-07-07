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
    _sigmoid, _logit,
)
from src.classifier.classifier import PyTorchClassifier

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
       reference=False, ctx="Zero_Shot", prompt="hello there"):
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
    return r


def decide(r, turns=1, tokens=200.0):
    return decide_memory_retrieval(r, turn_count=turns, total_tokens=tokens, settings=settings)


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
fin = PyTorchClassifier._finalize_confidence
# ML path: ctx probs at indices 22..24 = [Zero_Shot, LTM, RTS]
r = ClassificationResult([], [], "Long_Term_Memory",
                         [0.0] * 22 + [0.1, 0.7, 0.2], 0.7, "x")
fin(None, r)
check("ML path p_ltm read from ctx[1]", abs(r.p_ltm - 0.7) < 1e-9)
check("ML path p_rts read from ctx[2]", abs(r.p_rts - 0.2) < 1e-9)
check("ML path ctx_confidence = top1-top2", abs(r.ctx_confidence - 0.5) < 1e-9)

# DI3 path: raw_probs all zero → derive prior from the label.
r2 = ClassificationResult(["Null_Noise"], ["Casual_Banter"], "Zero_Shot",
                          [0.0] * 25, 0.95, "asdf")
fin(None, r2)
check("DI3 zero-shot prior p_ltm low", r2.p_ltm < 0.2)
r3 = ClassificationResult([], [], "Long_Term_Memory", [0.0] * 25, 0.7, "it")
fin(None, r3)
check("DI3 LTM prior p_ltm high", r3.p_ltm > 0.8)

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
