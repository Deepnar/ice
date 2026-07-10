# C13 + C14 — Caching strategy · KV-cache persistence (decision framework)

Assumes decided specs: `C7_scheduling.md` (redis is GONE — "Redis or in-process"
is answered: in-process), `Z1` expansion (profiling happens there),
`B1/C4/C9/C10/C11/G23_C17` (the write-paths whose churn made caching premature
are now all specced — invalidation can be designed against their decided
shapes). Grounded: the roadmap's own verdict (premature, not blocked; no
measured bottleneck; C14 wants a settled prompt shape and backend cache APIs).

**This is deliberately a framework spec, not a build spec** — building a caching
layer before Z1's profile exists would violate the item's own analysis. What IS
decided now:

## 1. Decisions

- **D1: the two stable pieces ship early (with Z1-prep's tuning pass, since it
  hammers retrieval):** (a) **prompt-embedding LRU** — process-local
  `functools.lru_cache(512)` keyed on the exact encode input string, wrapping
  the embedder call in `create_core()`'s shared instance (repeat prompts and
  the T3/G13 re-encode paths hit it); (b) the **nvidia-smi 10s cache** (already
  C7/G4's). Both are invalidation-free (content-keyed / TTL) — that's why they
  ship early and nothing else does.
- **D2: everything else waits for Z1's profile**, which times: leg SQL, encode
  calls, cluster scoring, classifier forward, assembly. Decision rules (rule
  2b, the whole spec): a stage >15% of p50 pre-flight latency → it gets a cache
  design; cluster scores only if C5's rework still shows >30ms; classifier
  outputs are cached ONLY keyed on (prompt, context-window contents) — i.e.
  effectively never for context-aware turns; fragment-level caching is
  presumptively REJECTED (every T/C6/C8 mode-parameter multiplies key-space;
  invalidation couples to every write path — the cost the roadmap named).
- **D3 (C14): investigation protocol, no build.** At Z1, measure actual prefix
  stability (log the assembled message list hash by segment across 50 real
  turns): if the stable prefix (system + slots + conversation summary) exceeds
  ~30% of prompt tokens, test Ollama's native prompt caching behavior
  (keep_alive + identical prefixes) and record findings; persistent KV export
  (vLLM/SGLang APIs) is only revisited if a dedicated-mode power-user story
  materializes. Fragment ordering already stable-prefix-first — no code change.
- **USER-REQUIRED:** none.

## 2. Files / validation / traps

D1a lands in `create_core()` (one decorator + a `cache_info` log line at
shutdown; test: second identical encode is a hit). The rest is a Z1 checklist
item: produce `docs/tuning_report.md` §caching with the measured table + which
decision rules fired. **Traps:** don't cache fragments "just for the repeated-
question case" (T/C6 mode keys explode it); don't add a cache without a
measured stage cost (this spec's rules are the gate); don't resurrect redis for
caching (C7 killed it; in-process or nothing).
