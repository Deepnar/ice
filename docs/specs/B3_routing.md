# B3 — Learned MoE routing (gated build)

Assumes decided specs: `FINAL_experiments.md` (the MoE pilot gate + the judged
per-(topic, model) scores that seed the reward table), `B1_classifier_retrain.md`
(`p_complex`, per-head confidences), `C7_scheduling.md` (no redis — state in
postgres), `E0_E7_services_mcp.md` (registry edits via service). Grounded at
commit `063b4f6`: `find_best_model(topic_tags, intent_tags)` hardcoded overlap
scorer; file-backed registry with `_ollama_name_to_hf_id` HF-guessing (garbage-
prone, roadmap-condemned); session stickiness = 3-shift hysteresis in main.py;
C16's `get_model_context_window`.

## 1. Decisions

- **D1: three gates, in order, before ANY of this is built:** (a) F15's advisor
  has produced a specialist pool that actually fits the hardware; (b) FINAL's
  50-probe pilot shows specialists ≥ +0.15 over the generalist; (c) B1 promoted
  (`p_complex` + head confidences exist). **If (b) fails, B3 closes as
  "generalist-only"** — the roadmap entry gets that verdict and this spec's
  build section is void. Everything below is the build branch.
- **D2: two-stage policy behind the stable signature**
  `find_best_model(topic_tags, intent_tags, required_tokens=None,
  classification=None)` (new optional kwarg; old callers unaffected):
  1. **Class selection (deterministic):** `p_complex ≥ 0.6` OR
     `required_tokens > small-class window` → **big** class (cloud model if the
     F11 toggle is on and consented, else biggest local); derived-Zero_Shot with
     high ctx_confidence and short prompt → **small** class; else **specialist**
     class.
  2. **Within-class (learned):** epsilon-greedy (ε=0.1) over
     `routing_rewards(topic_key, model_name) → {mean_score, n}` (postgres
     table): topic_key = the dominant topic tag; unseen pairs inherit the
     generalist's prior. **Seeded offline** from FINAL's judged artifacts
     (`scripts/seed_routing_rewards.py` reads the per-probe scores + the
     manifest's model ids); **updated online ONLY by explicit F9 feedback**
     (thumbs → ±1 scaled into the mean). No implicit signals — regenerations
     and edits are too noisy to learn from.
- **D3: exploration is suppressed where it hurts:** never explore (ε=0) on
  big-class turns (the user's hardest asks are not the experiment budget) or on
  `confirmed: false` models.
- **D4: registry cleanup lands with this (roadmap fold):** delete
  `_ollama_name_to_hf_id` and the HF-tag maps; tags come from a one-time
  bg-model tagging proposal per model + user confirmation (F4 UI or
  `ice_control registry_edit`); unconfirmed models route as generalists only.
  Registry gains `keep_warm: bool`; selection passes Ollama `keep_alive` for
  the chosen model and keeps the generalist warm always (the 5–15 s swap-spike
  mitigation; hysteresis already caps switch frequency).
- **USER-REQUIRED:** confirm proposed tags per registry model (~1 min each);
  consciously enable the cloud toggle (F11) if wanted.
- **Empirical deferral:** ε, the 0.6 complexity threshold, and reward-mean
  decay (whether old scores age out) — Z1-prep sweeps the first two; the third
  waits for months of F9 data (rule: revisit when any (topic, model) n > 200).

## 2. Files & integration points

`model_registry/registry.py` (engine swap behind the signature; cleanup;
keep_alive) · migration (`routing_rewards`) · `scripts/seed_routing_rewards.py`
· main.py routing call passes `classification` + `required_tokens=total_budget`
· F9 hook writes rewards (when built) · `tests/test_routing.py` (policy math:
class thresholds, ε behavior incl. suppression, unseen-pair prior, hysteresis
interaction; seeding from a fixture FINAL artifact; live smoke: sticky model
survives, keep_alive param present).

## 3. Edge cases

Empty reward table (pre-seed) → pure class selection + generalist (today's
behavior, minus the overlap scorer's noise). Registry model removed while
sticky → stickiness resets (exists). Cloud toggle on but key missing → fall to
biggest local with one log line. Two topics tied → topic_key = alphabetical
first of the tied set (deterministic).

## 4. Look-ahead constraints

F12 (multi-model responses) stays parked until this proves single-specialist
wins. F15 writes `context_window`/sizes the pool this reads. FINAL re-measures
routing with the same probes that seeded it — noted there as a
train/test-contamination caveat: seed from Exp-N, measure on Exp-N+1's fresh
probes.

## 5. Traps

- Don't build before the gates — the whole item was deferred because neutral
  routing + no pool + no signal; the gates ARE the item.
- Don't learn from implicit signals — a regeneration means many things; a
  poisoned reward table is worse than none.
- Don't let the bandit override stickiness mid-conversation — hysteresis first,
  policy second (topic whiplash was the original sin the stickiness fixed).
- Don't keep the HF-guessing "as fallback" — wrong tags are worse than no tags
  (unconfirmed = generalist is the fallback).
