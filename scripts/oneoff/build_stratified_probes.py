"""A 120-probe subset, stratified by TYPE and by CONVERSATION.

⚑ EQUAL ALLOCATION ACROSS TYPES, NOT PROPORTIONAL. The five types are five
different metrics that are never averaged, and they are compared per type
between arms — so power should be spread evenly, not concentrated in
`episodic_lookup` (232 of 444) which is the type the NER swap barely touches.
24 per type puts the weight on `codex_multihop`, `summary_synthesis` and
`temporal`, which are where the graph actually shows up.

⚑ AND STRATIFIED BY CONVERSATION INSIDE EACH TYPE. Sampling failed twice in
this project by drawing unevenly across the three conversations, which are three
different registers. Round-robin, deterministic seed, so BOTH ARMS ANSWER THE
SAME PROBES — the comparison is meaningless otherwise.
"""
from __future__ import annotations
import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("experiments/curation_files/typed_probes.json")
OUT = Path("experiments/curation_files/typed_probes.stratified120.json")
PER_TYPE = 24
SEED = 20260820

probes = json.loads(SRC.read_text())["probes"]
by_type = defaultdict(lambda: defaultdict(list))
for p in probes:
    by_type[p["probe_type"]][p["conversation"]].append(p)

rng = random.Random(SEED)
picked = []
for ptype in sorted(by_type):
    convs = sorted(by_type[ptype])
    pools = {c: rng.sample(by_type[ptype][c], len(by_type[ptype][c])) for c in convs}
    # ⚠ Target computed ONCE. Evaluating it inside the loop condition against
    # the DRAINING pools made it shrink as items were taken, so a type with 40
    # probes stopped at 20 — the loop raced its own denominator.
    target = min(PER_TYPE, sum(len(v) for v in pools.values()))
    take, i = [], 0
    while len(take) < target:
        c = convs[i % len(convs)]
        if pools[c]:
            take.append(pools[c].pop())
        i += 1
        if i > 10000:
            break
    picked.extend(take)
    print(f"  {ptype:<20} {len(take):>3}  by conv: "
          f"{dict(Counter(x['conversation'] for x in take))}")

OUT.write_text(json.dumps({"probes": picked,
                           "source": SRC.name, "per_type": PER_TYPE,
                           "seed": SEED,
                           "note": "equal allocation by type; round-robin by "
                                   "conversation; identical for both arms"},
                          indent=1))
print(f"\nwrote {OUT}  ({len(picked)} probes)")
