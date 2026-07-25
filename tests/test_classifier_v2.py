"""Standalone behavioral test for B1's schema-v2 classifier core.

Run: uv run python tests/test_classifier_v2.py

Covers the parts of the B1 spec's §5 checklist that don't need a trained
checkpoint: the schema loader, template discipline (D3), the trunk+3-heads model
and its dual-generation checkpoint dispatch (§4), the derived context-reliance
layer (D6), C15's wide-net trigger on 28-wide probs, and the promotion
round-trip. Needs no DB and no GPU; loads the frozen encoder only for the two
end-to-end classify checks.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.api.config import settings
from src.classifier import templates
from src.classifier.classifier import PyTorchClassifier
from src.classifier.model import (ICEClassifier, LegacyICEClassifierV1,
                                  compute_pos_weights, head_losses,
                                  load_checkpoint)
from src.classifier.promotion import promote_checkpoint
from src.classifier.schema import (CONTEXT_RELIANCE, INTENT, TOPIC, load_schema,
                                   load_v1_schema, resolve_by_width)
from src.classifier.schemas import ClassificationResult
from src.retrieval.orchestrator import _head_confidences

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


V2 = load_schema()
V1 = load_v1_schema()

print("── D1: schema v2 is the single source of head layout ──")
check("schema_version 2", V2.version == 2)
check("topic head still 11", V2.head(TOPIC).width == 11)
check("intent head 11 → 13", V2.head(INTENT).width == 13)
check("context head 3 → 4", V2.head(CONTEXT_RELIANCE).width == 4)
check("total width 28", V2.total_width == 28)
check("v1 indices preserved (Codebase_Query/Code_Change appended last)",
      V2.labels(INTENT)[:11] == V1.labels(INTENT))
check("Zero_Shot is NOT a v2 label", "Zero_Shot" not in V2.labels(CONTEXT_RELIANCE))
check("context head is independent sigmoids",
      V2.head(CONTEXT_RELIANCE).activation == "sigmoid"
      and V2.head(CONTEXT_RELIANCE).decision == "independent")
check("v1 context head stayed softmax", V1.head(CONTEXT_RELIANCE).activation == "softmax")
check("offsets are contiguous",
      [ (h.offset, h.width) for h in V2.heads ] == [(0, 11), (11, 13), (24, 4)])
check("index() is absolute, not head-local",
      V2.head(CONTEXT_RELIANCE).index("Needs_Memory") == 24)
check("every label carries a definition (the labeling rubric reads these)",
      all(all(d.strip() for d in h.definitions) for h in V2.heads))
check("resolve_by_width(28) → v2", resolve_by_width(28).version == 2)
check("resolve_by_width(25) → v1", resolve_by_width(25).version == 1)
check("resolve_by_width(99) → None", resolve_by_width(99) is None)
check("train-time input width is native 1024", V2.input_dim == 1024)

print("── D3: templates are shared and versioned (the mismatch fix) ──")
# The v1 strings are frozen: D5's gate must render the old model's input the way
# that model actually saw it. This literal is the pre-B1 classifier.py text.
V1_EXPECTED = (
    "Given a user prompt, predict:\n"
    "1. TOPIC: what is the subject (Software_&_Tech, Creative_&_Media, etc.)\n"
    "2. INTENT: what is the user trying to do (Factual_Retrieval, Troubleshooting, etc.)\n"
    "3. CONTEXT RELIANCE: does the user need memory (Zero_Shot, Long_Term_Memory, Real_Time_Search)\n\n"
    "User prompt: hi"
)
check("v1 no-context template is verbatim", templates.render("hi", None, version=1) == V1_EXPECTED)
check("v1 context template keeps its prefix",
      templates.render("hi", "prior turn", version=1).startswith(
          "Conversation context (summarized):\nprior turn"))
check("v2 names the four reliance signals",
      "Needs_Memory" in templates.render("hi", None, version=2)
      and "High_Complexity" in templates.render("hi", None, version=2))
check("v2 names the coding intents", "Codebase_Query" in templates.render("hi", None, version=2))
check("context and no-context templates differ",
      templates.render("hi", "ctx") != templates.render("hi", None))
check("empty context falls back to the standalone template",
      templates.render("hi", "") == templates.render("hi", None))
try:
    templates.render("hi", None, version=99)
    check("unknown template version raises", False)
except ValueError:
    check("unknown template version raises", True)
long_ctx = templates.truncate_context(["word " * 400, "second turn"])
check("context truncation respects the 500-word budget",
      len(long_ctx.split()) <= templates.CONTEXT_MAX_WORDS)

print("── D2: trunk + three heads, all-sigmoid, 1024-dim ──")
m = ICEClassifier(schema=V2)
m.eval()   # dropout off: forward() and forward_heads() must agree deterministically
n_params = sum(p.numel() for p in m.parameters())
check(f"~700k params (got {n_params:,})", 600_000 < n_params < 800_000)
x = torch.randn(4, 1024)
check("forward returns concatenated 28 logits", tuple(m(x).shape) == (4, 28))
heads = m.forward_heads(x)
check("forward_heads splits per head",
      {k: tuple(v.shape) for k, v in heads.items()}
      == {TOPIC: (4, 11), INTENT: (4, 13), CONTEXT_RELIANCE: (4, 4)})
check("concat order matches schema head order",
      torch.allclose(m(x)[:, V2.slice(INTENT)], heads[INTENT]))

targets = torch.zeros(4, 28)
targets[0, V2.head(CONTEXT_RELIANCE).index("Needs_Memory")] = 1.0
targets[:, V2.head(TOPIC).index("Software_&_Tech")] = 1.0
pw = compute_pos_weights(targets, V2)
check("pos-weights computed per sigmoid head", set(pw) == {TOPIC, INTENT, CONTEXT_RELIANCE})
check("a 1-in-4 label outweighs a 4-in-4 label",
      pw[CONTEXT_RELIANCE][0] > pw[TOPIC][0])
check("pos-weights are capped (no 2000× rare-label shouting)",
      float(pw[CONTEXT_RELIANCE].max()) <= 20.0)
losses = head_losses(heads, targets, V2, pw)
check("per-head losses returned", set(losses) == {TOPIC, INTENT, CONTEXT_RELIANCE})
check("all losses finite", all(torch.isfinite(v) for v in losses.values()))

print("── §4: checkpoints are self-describing; both generations load ──")
tmp = tempfile.mkdtemp()
v2_path = os.path.join(tmp, "v2.pt")
torch.save(m.checkpoint_dict(notes="test"), v2_path)
m2, meta = load_checkpoint(v2_path)
check("v2 checkpoint records schema_version", meta["schema_version"] == 2)
check("v2 checkpoint records template_version", meta["template_version"] == 2)
check("v2 checkpoint records input_dim", meta["input_dim"] == 1024)
check("v2 checkpoint records label names (not just widths)",
      meta["head_labels"][CONTEXT_RELIANCE] == list(V2.labels(CONTEXT_RELIANCE)))
check("weights survive the round-trip",
      torch.allclose(m.trunk[0].weight, m2.trunk[0].weight))

v1_path = os.path.join(tmp, "v1.pt")
torch.save(LegacyICEClassifierV1().state_dict(), v1_path)   # bare state_dict, as v1 saved it
legacy, legacy_meta = load_checkpoint(v1_path)
check("bare v1 state_dict still loads", isinstance(legacy, LegacyICEClassifierV1))
check("v1 load reports schema_version 1", legacy_meta["schema_version"] == 1)
check("v1 load reports the 384 input", legacy_meta["input_dim"] == 384)
check("v1 model exposes per-head views too",
      tuple(legacy.forward_heads(torch.randn(2, 384))[CONTEXT_RELIANCE].shape) == (2, 3))

try:
    m.verify_against(V1)
    check("verify_against catches a width mismatch", False)
except ValueError:
    check("verify_against catches a width mismatch", True)

print("── C15: wide-net trigger reads per-head confidence at 28 wide ──")

peaked_topic_fuzzy_intent = [0.0] * 28
peaked_topic_fuzzy_intent[V2.head(TOPIC).index("Software_&_Tech")] = 0.93
for i in range(V2.head(INTENT).offset, V2.head(INTENT).offset + 3):
    peaked_topic_fuzzy_intent[i] = 0.31
r = ClassificationResult([], [], "Zero_Shot", peaked_topic_fuzzy_intent, 0.93, "x")
t_conf, i_conf = _head_confidences(r)
check("28-wide probs slice correctly (topic peaked)", abs(t_conf - 0.93) < 1e-9)
check("28-wide probs slice correctly (intent fuzzy)", abs(i_conf - 0.31) < 1e-9)

r.head_confidences = {TOPIC: 0.11, INTENT: 0.12}
check("published head_confidences win over re-slicing",
      _head_confidences(r) == (0.11, 0.12))

r25 = ClassificationResult([], [], "Zero_Shot", [0.0] * 22 + [0.1, 0.7, 0.2], 0.7, "x")
r25.raw_probs[V1.head(TOPIC).index("Meta_AI")] = 0.44
check("25-wide (v1 checkpoint) probs still slice correctly",
      abs(_head_confidences(r25)[0] - 0.44) < 1e-9)

di3 = ClassificationResult(["Null_Noise"], ["Casual_Banter"], "Zero_Shot",
                           V2.empty_probs(), 0.95, "asdf")
check("DI3 all-zero convention is schema-wide (28)", len(di3.raw_probs) == 28)
check("all-zero probs fall back to max_confidence", _head_confidences(di3) == (0.95, 0.95))

print("── promotion: backup + atomic swap, shared by B4 and the pipeline ──")
live = os.path.join(tmp, "live.pt")
torch.save({"schema_version": 1, "state_dict": {}, "marker": "old"}, live)
res = promote_checkpoint({"schema_version": 2, "state_dict": {}, "marker": "new"}, live)
check("outgoing checkpoint was backed up", res["backup"] and os.path.exists(res["backup"]))
check("backup holds the OLD model",
      torch.load(res["backup"], weights_only=False)["marker"] == "old")
check("live path now holds the NEW model",
      torch.load(live, weights_only=False)["marker"] == "new")
check("no temp file left behind", not os.path.exists(live + ".tmp"))
fresh = os.path.join(tmp, "fresh.pt")
check("first-ever promotion needs no backup",
      promote_checkpoint({"state_dict": {}}, fresh)["backup"] is None)

print("── end-to-end: both generations serve through PyTorchClassifier ──")

c2 = PyTorchClassifier(model_path=v2_path)
out = c2.classify("what did we decide about the retrieval weights last week?")
check("v2 classify returns 28 probs", len(out.raw_probs) == 28)
check("v2 classify populates p_temporal/p_complex",
      out.p_temporal > 0.0 and out.p_complex > 0.0)
check("v2 classify derives a legacy 3-way string",
      out.context_reliance in ("Zero_Shot", "Long_Term_Memory", "Real_Time_Search"))
check("v2 classify publishes head confidences",
      set(out.head_confidences) == {TOPIC, INTENT, CONTEXT_RELIANCE})
check("v2 tags come from the v2 label space",
      all(t in V2.labels(TOPIC) for t in out.topic_tags))

if os.path.exists(settings.classifier_model_path):
    c1 = PyTorchClassifier(model_path=settings.classifier_model_path)
    live_out = c1.classify("what did we decide about the retrieval weights last week?")
    check("the LIVE v1 checkpoint still serves unchanged", len(live_out.raw_probs) == 25)
    check("live v1 still routes a memory question to LTM",
          live_out.context_reliance == "Long_Term_Memory")
    check("live v1 still populates B2's p_ltm seam", live_out.p_ltm > 0.5)
else:
    print("  SKIP  live v1 checkpoint not present")

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
