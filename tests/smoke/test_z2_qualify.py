"""Z2 judge qualification (experiments/z2/qualify.py): the module compiles and every judge prompt it
records still exists — Z3 judging run 1's panel failed on both (2026-11-03): a `global` after its
first use, and a renamed claim prompt that would have crashed after every judge call was spent."""

import importlib
import re
from pathlib import Path


def test_qualify_compiles_and_names_only_existing_judge_prompts():
    importlib.import_module("experiments.z2.qualify")
    source = Path("experiments/z2/qualify.py").read_text()
    modules = {"grade": "experiments.z2.grade", "bg_audit": "experiments.z2.bg_audit",
               "walk_audit": "experiments.z2.walk_audit", "L": "experiments.z2.ledger"}
    names = set(re.findall(r"\b(grade|bg_audit|walk_audit|L)\.([A-Za-z_]+)", source))
    missing = [f"{m}.{a}" for m, a in names if not hasattr(importlib.import_module(modules[m]), a)]
    assert names and not missing


def test_planted_claims_go_through_the_batched_claim_check():
    import json
    from experiments.z2 import grade
    from experiments.z2 import qualify as Q
    seen = []

    def ask(system, prompt):
        seen.append((system, json.loads(prompt)))
        return {"claims": [{"id": "C1", "stated_by": [], "found": True, "passage": 0}]}
    assert Q.claim_found(ask, "the cat is Tofu", [dict(index=0, turn=1, role="user", text="my cat Tofu")]) is True
    assert seen[0][0] is grade.CLAIMS_SYSTEM and seen[0][1]["CLAIMS"] == [{"id": "C1", "text": "the cat is Tofu"}]
    assert Q.claim_found(lambda s, p: None, "x", []) is None                  # no reply: not judged, never False
