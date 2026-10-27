"""FINAL's BEAM portion (FINAL_experiments.md §2.1, decided 2026-10-26): pinned files and
the exact conversations, chosen by a fixed seed before any result is seen.

128K tier ("100K" split): all 20. 500K and 1M tiers: 10 each = 2 Coding + 2 Math +
6 general conversations from 6 different categories. Writes MANIFEST.json beside this
file; refuses to overwrite a different selection.

Trimmed 2026-10-28 (maintainer: FINAL already has many evaluations, and B and other histories
reach ~1M tokens): the 1M tier is dropped, and the 500K tier keeps 5 of its 10 — 1 Coding +
1 Math + 3 general, drawn (seed TRIM_SEED) from the 10 already chosen, so the trim stays inside
the pre-registered draw. `--trim` re-derives the original draw, checks it equals the manifest,
then writes the trimmed one.

    uv run --no-sync python experiments/beam/select.py           # write / verify
    uv run --no-sync python experiments/beam/select.py --trim    # the 2026-10-28 trim
"""

import ast
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPO, REVISION = "Mohammadta/BEAM", "3205395e897e7318c7b094ef4e6047b9b82dbb03"
SEED = 20261026
TIERS = {"100K": None, "500K": 10, "1M": 10}       # None = all
PER_TIER = {"Coding": 2, "Math": 2}                 # the rest: one per general category
TRIM = {"500K": dict(keep=5, per_category={"Coding": 1, "Math": 1}), "1M": None}     # None = dropped
TRIM_SEED = 20261028


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(row):
    msgs = [m for s in row["chat"] for m in s]
    questions = ast.literal_eval(row["probing_questions"])
    return dict(conversation_id=row["conversation_id"], category=row["conversation_seed"]["category"],
                title=row["conversation_seed"]["title"], sessions=len(row["chat"]),
                user_messages=sum(m["role"] == "user" for m in msgs),
                approx_tokens=sum(len(m["content"] or "") for m in msgs) // 4,
                questions=sum(len(v) for v in questions.values()))


def choose(rows, k, rng):
    if k is None:
        return rows
    by_cat = {}
    for r in sorted(rows, key=lambda r: r["conversation_id"]):
        by_cat.setdefault(r["category"], []).append(r)
    picked = []
    for cat, n in PER_TIER.items():
        picked += rng.sample(by_cat.pop(cat), n)
    general = sorted(by_cat)
    for cat in rng.sample(general, k - len(picked)):
        picked.append(rng.choice(by_cat[cat]))
    return sorted(picked, key=lambda r: (r["category"], r["conversation_id"]))


def build():
    import pyarrow.parquet as pq
    rng = random.Random(SEED)
    out = dict(source=f"https://huggingface.co/datasets/{REPO}", revision=REVISION, license="cc-by-sa-4.0",
               seed=SEED, rule="128K all; 500K/1M: 2 Coding + 2 Math + 6 general from distinct categories",
               files={}, tiers={})
    for tier, k in TIERS.items():
        path = DATA / f"{tier}.parquet"
        out["files"][tier] = dict(file=f"data/{tier}.parquet", sha256=sha256(path))
        rows = [summary(r) for r in pq.read_table(path).to_pylist()]
        chosen = choose(rows, k, rng)
        out["tiers"][tier] = dict(available=len(rows), chosen=len(chosen),
                                  user_messages=sum(r["user_messages"] for r in chosen), conversations=chosen)
    return out


def trim(manifest, rng=None):
    """The 2026-10-28 trim of a manifest's draw (see the module note)."""
    rng = rng or random.Random(TRIM_SEED)
    out = json.loads(json.dumps(manifest))
    for tier, rule in TRIM.items():
        if tier not in out["tiers"]:
            continue
        if rule is None:
            del out["tiers"][tier]
            continue
        rows = out["tiers"][tier]["conversations"]
        by_cat = {}
        for r in sorted(rows, key=lambda r: r["conversation_id"]):
            by_cat.setdefault(r["category"], []).append(r)
        picked = []
        for cat, n in rule["per_category"].items():
            picked += rng.sample(by_cat.pop(cat), n)
        for cat in rng.sample(sorted(by_cat), rule["keep"] - len(picked)):
            picked.append(rng.choice(by_cat[cat]))
        picked = sorted(picked, key=lambda r: (r["category"], r["conversation_id"]))
        out["tiers"][tier].update(chosen=len(picked), user_messages=sum(r["user_messages"] for r in picked),
                                  conversations=picked, trimmed_from=len(rows))
    out["trim"] = dict(date="2026-10-28", seed=TRIM_SEED,
                       rule="1M tier dropped; 500K: 1 Coding + 1 Math + 3 general from the 10 drawn")
    return out


def main() -> int:
    manifest = build()
    path = HERE / "MANIFEST.json"
    if "--trim" in sys.argv:
        current = json.loads(path.read_text())
        if current.get("trim"):
            print("MANIFEST.json is already trimmed")
            return 0
        if current != manifest:
            print("MANIFEST.json differs from a fresh draw: refusing to trim it", file=sys.stderr)
            return 1
        path.write_text(json.dumps(trim(manifest), indent=1, ensure_ascii=False) + "\n")
        manifest = json.loads(path.read_text())
    elif path.exists() and json.loads(path.read_text()).get("trim"):
        manifest = trim(manifest)
    if path.exists() and json.loads(path.read_text()) != manifest:
        print("MANIFEST.json differs from a fresh selection: refusing to overwrite", file=sys.stderr)
        return 1
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    for tier, t in manifest["tiers"].items():
        print(f"{tier}: {t['chosen']}/{t['available']} conversations, {t['user_messages']} user messages")
        for c in t["conversations"]:
            print(f"   {c['category'][:34]:34s} {c['user_messages']:5d} msgs {c['approx_tokens']//1000:5d}K  {c['title'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
