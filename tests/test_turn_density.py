"""C1 behavioral test: density signal, storage matrix, grounded-summary retry,
read-time representation choice, and budget degrade-before-drop.

LLM stubbed (real bg-model summaries pend Z1, like A1/A2/A6/C5); NER stubbed in
the pure sections, real in the DB integration. Inserts its own rows, deletes
them after. Run: uv run python tests/test_turn_density.py
"""
import os
import sys
import uuid
import hashlib
from types import SimpleNamespace
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


# ── stub NER for the pure sections (turn_density imports it lazily) ──
import src.retrieval.ner_utils as ner
_real_extract = ner.extract_entities
_stub_entities = []
ner.extract_entities = lambda text, emb, max_chars=None: list(_stub_entities)

from src.workers.turn_density import (
    extract_key_terms, compute_entropy, decide_representation,
    summary_coverage, must_terms, RAW_KEEP_MAX_WORDS,
)

print("── key-term extraction ──")
_stub_entities = ["Orien", "Obsidian Citadel"]
kt = extract_key_terms("Orien reached the Obsidian Citadel at 42% strength, "
                       "using decay_score logic in main.py v2.5", embedder=None)
check("entities from NER", kt["entities"] == ["Orien", "Obsidian Citadel"])
check("figures captured", any("42%" in f for f in kt["figures"]))
check("identifiers captured (snake_case + file.ext)",
      "decay_score" in kt["identifiers"] and "main.py" in kt["identifiers"])

print("── entropy: dense vs diffuse ──")
dense_text = ("Orien fought Kazama at the Citadel. " * 20)
_stub_entities = ["Orien", "Kazama", "Citadel"]
kt_dense = extract_key_terms(dense_text, None)
e_dense = compute_entropy(dense_text, kt_dense, has_code=False)
banter = ("well i guess that is fine and we can just keep going along " * 20)
_stub_entities = []
kt_banter = extract_key_terms(banter, None)
e_banter = compute_entropy(banter, kt_banter, has_code=False)
check(f"dense ({e_dense}) > diffuse ({e_banter})", e_dense > e_banter)
check("code raises entropy",
      compute_entropy("x " * 100, kt_banter, has_code=True) > e_banter)

print("── storage matrix (want_summary + default hint) ──")
d = decide_representation(5000, 0.5, False, False, is_document=True)
check("document → raw hint, no summary", d["inject_raw"] and not d["want_summary"])
d = decide_representation(100, 0.9, False, False, False)
check("short → raw, no summary (no LLM call)", d["inject_raw"] and not d["want_summary"])
d = decide_representation(800, 0.5, True, False, False)
check("long code → raw hint + summary stored, coverage does NOT set hint",
      d["inject_raw"] and d["want_summary"] and not d["summary_decides"])
d = decide_representation(800, 0.2, False, True, False)
check("long creative → raw hint + summary stored (continuity)",
      d["inject_raw"] and d["want_summary"] and not d["summary_decides"])
d = decide_representation(800, 0.6, False, False, False)
check("long dense → summary stored, coverage sets hint",
      d["want_summary"] and d["summary_decides"] and d["reason"] == "dense_long")
d = decide_representation(800, 0.1, False, False, False)
check("long diffuse → summary stored, coverage sets hint",
      d["want_summary"] and d["summary_decides"] and d["reason"] == "diffuse_long")

print("── coverage measurement ──")
_stub_entities = ["Orien", "Kazama"]
kt = extract_key_terms("Orien beat Kazama with 42% odds", None)
check("full coverage = 1.0",
      summary_coverage("Orien defeated Kazama at 42% odds", kt) == 1.0)
partial = summary_coverage("Somebody won a duel with 42% odds", kt)
check(f"dropped entities lower coverage ({partial})", partial < 0.7)
check("no must-terms → vacuously 1.0", summary_coverage("anything", {"entities": [], "figures": [], "identifiers": []}) == 1.0)

print("── grounded summary retry (LLM stubbed) ──")
import src.workers.post_flight as pf
_calls = {"n": 0}


class _FakeCompletions:
    def create(self, **kw):
        _calls["n"] += 1
        content = ("A vague summary without the names.\nKey terms: none"
                   if _calls["n"] == 1 else
                   "Orien defeated Kazama with 42% odds.\nKey terms: Orien, Kazama, 42%")
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content))])


pf.bg_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
summ, cov, _ab = pf.generate_summary("q", "a", kt, "test-model")
check("retry fired when first summary dropped terms", _calls["n"] == 2)
check(f"retry recovered coverage ({cov})", cov >= 0.7 and "Orien" in summ)

print("── read-time chooser (store both, decide per query) ──")
from src.retrieval.orchestrator import HybridRetrievalOrchestrator, ContextFragment
orch = HybridRetrievalOrchestrator(None, None)


def row(raw="raw text about the zephyr protocol details", summ="summary of protocol",
        cov=0.9, inject_raw=True):
    return SimpleNamespace(raw_text=raw, summary_text=summ, summary_coverage=cov,
                           inject_raw=inject_raw)


clf_exact = SimpleNamespace(intent_tags=["Factual_Retrieval"])
clf_broad = SimpleNamespace(intent_tags=["Analysis_&_Summarization"])
clf_neutral = SimpleNamespace(intent_tags=["Casual_Banter"])

t, dg, ab = orch._choose_representation(row(summ=None), clf_neutral, set())
check("no summary → raw", t.startswith("raw"))
t, dg, ab = orch._choose_representation(row(cov=0.3, inject_raw=False), clf_broad, set())
check("untrusted summary (cov 0.3) → raw even when hint says summary", t.startswith("raw") and dg is None)
t, dg, ab = orch._choose_representation(row(cov=None, inject_raw=False), clf_neutral, set())
check("legacy summary (cov NULL) + summary hint → summary (status quo)", t.startswith("summary"))
t, dg, ab = orch._choose_representation(row(), clf_neutral, {"zephyr"})
check("keyword only in raw → raw, NOT degradable", t.startswith("raw") and dg is None)
t, dg, ab = orch._choose_representation(row(summ="summary of zephyr protocol"), clf_exact, {"zephyr"})
check("exactness intent → raw, degradable (keyword survives in summary)",
      t.startswith("raw") and dg is not None)
t, dg, ab = orch._choose_representation(row(), clf_broad, set())
check("compression-tolerant intent → trusted summary", t.startswith("summary"))
t, dg, ab = orch._choose_representation(row(), clf_neutral, set())
check("neutral intent → default hint (raw), degradable", t.startswith("raw") and dg is not None)
t, dg, ab = orch._choose_representation(row(inject_raw=False), clf_neutral, set())
check("neutral intent + summary hint → summary", t.startswith("summary"))

print("── budget: degrade-before-drop ──")
big_raw = "word " * 600
small_sum = "compact summary of the big turn"
f_deg = ContextFragment(text=big_raw, source_type="episodic", score=2.0,
                        token_count=int(600 * 1.33), degrade_text=small_sum)
f_nodeg = ContextFragment(text=big_raw, source_type="episodic", score=1.9,
                          token_count=int(600 * 1.33))
kept = orch._enforce_token_budget([f_deg, f_nodeg], max_tokens=100)
check("too-big fragment degraded to its summary instead of dropped",
      len(kept) == 1 and kept[0].text == small_sum)
check("degraded token_count recomputed", kept[0].token_count < 50)

print("── live-DB integration through evaluate_turn (clobber + coverage gate) ──")
ner.extract_entities = _real_extract   # real NER from here on
from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory, IdempotencyKey

pf.is_gpu_busy = lambda: False
pf.is_user_active = lambda: False
pf.generate_summary = lambda p, r, kt, mu="": (
    "Grounded summary. Key terms: " + ", ".join(must_terms(kt)[:25]), 0.95, "one line abstract")

db = SessionLocal()
conv = Conversation(memory_scope_type="none")   # private → no queue side-effects
db.add(conv)
db.commit()
_batches = []


def run_case(response_body, prompt="tell me"):
    batch = uuid.uuid4()
    _batches.append(batch)
    turn = EpisodicMemory(
        conversation_id=conv.id, batch_id=batch, timestamp=datetime.now(timezone.utc),
        topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory", is_private=True,
        raw_text=f"User: {prompt}\n\nAssistant: {response_body}",
        idempotency_key=f"test-c1-{batch}",
    )
    db.add(turn)
    db.commit()
    pf.evaluate_turn(batch_id=str(batch), prompt=prompt, response=response_body,
                     conversation_id=str(conv.id))
    db.expire_all()
    return db.query(EpisodicMemory).filter_by(batch_id=batch).first()


t = run_case("The ICE proxy routes Qwen models through FastAPI on port 8000. " * 40)
check("entropy_score finally computed (was always NULL)", t.entropy_score is not None and t.entropy_score > 0)
check("long turn: summary stored + coverage recorded",
      t.summary_text is not None and t.summary_coverage == 0.95)
check("trusted summary sets hint → inject_raw False", t.inject_raw is False)

t = run_case("word " * 2500)   # single-assistant giant paste
check("document detected", t.is_document is True)
check("CLOBBER FIXED: document keeps raw injection", t.inject_raw is True)

t = run_case("short and simple answer")
check("short turn: raw, no summary, no LLM call", t.inject_raw is True and t.summary_text is None)

# cleanup
for b in _batches:
    db.query(IdempotencyKey).filter_by(key=hashlib.sha256(str(b).encode()).hexdigest()).delete()
db.query(EpisodicMemory).filter_by(conversation_id=conv.id).delete()
db.query(Conversation).filter_by(id=conv.id).delete()
db.commit()
db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
