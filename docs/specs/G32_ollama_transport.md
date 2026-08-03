# G32 — ICE↔Ollama control surface: the audit, and what it implies
Assumes decided specs: none

> **Status: §0 (the audit) is COMPLETE and factual — measured 2026-08-03 against
> Ollama 0.30.7 on the maintainer's machine. §1 carries one open user decision.
> No production code has been written.**
>
> The roadmap made the audit G32's *first deliverable* on the grounds that the
> known list of dropped parameters was "what was TESTED, not the control
> surface". That was right, and the audit's main result is that **the entry's
> central premise is wrong**: constrained decoding is *not* native-only.

---

## 0. The audit

### 0.1 Method

Every probe is **two-sided** and keyed to an **observable effect** — a number in
`/api/ps`, a field in the response envelope, or output that either conforms to a
schema or does not. "The request didn't error" is never the test: the `/v1` shim
accepts unknown parameters silently, which is the whole problem. Probe model
`qwen3:4b-instruct-bg` unless noted; the thinking probes use the resident
`gemma4:26b-a4b-it-q4_K_M`, which genuinely reasons.

⚠ **One invalid comparison was caught and redone**, and it is worth recording
because it is the shape of mistake this audit exists to avoid: the first
truncation probe sent `num_ctx=2048` on the native arm and nothing on the `/v1`
arm. But `/v1` *drops* `num_ctx` — so that arm ran at the 32,768 default and
never truncated at all. The "native is loud about truncation" reading was an
artifact of the two arms having different windows. See §0.4.

### 0.2 Endpoint enumeration (Ollama 0.30.7)

Every route below is present on the live server. "ICE today" is grep-verified
against `src/`.

| Endpoint | What it does | ICE today | What it would buy |
|---|---|---|---|
| `POST /api/chat` | native chat; `options`, `format`, `keep_alive`, `think`, `tools` | **unused** | the whole native control surface (§0.3) |
| `POST /api/generate` | native completion; also the **unload** verb (`keep_alive:0` → `done_reason:"unload"`) | **unused** | G4(b)'s VRAM release mechanism |
| `GET /api/ps` | resident runners + **allocated `context_length`** | `runtime_probe.observed_context_window` | — (already used) |
| `POST /api/show` | `model_info` (arch ceiling), template, params, **`capabilities`** | `runtime_probe` reads `*.context_length` only | **`capabilities` is the capability-detection primitive G32 asks for** — see §0.5 |
| `GET /api/tags` | installed models | `registry.populate_from_ollama` | — (already used) |
| `POST /api/embed` | embeddings | unused | **nothing.** Returned `501 "does not support embeddings"`; ICE embeds locally via `SentenceTransformer` and G23 pinned embedding identity. A non-lever. |
| `POST /api/embeddings` | legacy embeddings | unused | nothing (as above) |
| `/api/pull` `/api/push` `/api/create` `/api/copy` `/api/delete` `/api/blobs/:digest` | model management | unused | out of scope; F15's hardware advisor is the only plausible future consumer (`pull`) |
| `GET /api/version` | server version | unused | capability gating by version, if ever needed |
| `POST /api/signout` | cloud-account signout | unused | nothing |
| `POST /v1/chat/completions` | OpenAI shim | **chat path** (raw httpx, `main.py:432/435`) **and background path** (OpenAI SDK, `bg_client_factory.py:98`) | — |
| `GET /v1/models`, `POST /v1/completions`, `POST /v1/embeddings`, `POST /v1/responses` | OpenAI shim | unused | nothing ICE needs |

### 0.3 The parameter diff — what survives `/v1`

| Control | Native | Through `/v1` | Evidence |
|---|---|---|---|
| `options.num_ctx` | **honoured** | **DROPPED** | runner allocated 8,192 native vs 32,768 (the default) through the shim |
| `options.top_k` (and `min_p`, `repeat_penalty`, `num_gpu`, …) | **honoured** | **DROPPED** | `top_k=1` at temp 1.6 collapsed native output 3→1 distinct; through `/v1`, 6→6 |
| `keep_alive` | **honoured** | **DROPPED** | `keep_alive:0` unloaded the model natively; through `/v1` it stayed resident. `keep_alive:"60s"` moved `expires_at` off the year-2318 value the `OLLAMA_KEEP_ALIVE=-1` env pins |
| `think` | **honoured**, and returns thinking as its **own** `message.thinking` field | DROPPED (no `think` key; reasoning arrives inside `message.reasoning`) | 26B: `think:true` → 625 ch thinking; `think:false` → 0 ch; both answered "No" |
| `format` = JSON schema | **honoured** | — | 8/8 conform |
| **`response_format` = `json_schema`** | — | **HONOURED** ⚠ | **8/8 conform**, including forcing every relation to an enum value the model would never choose (`RELATES_TO`) |
| `response_format` = `json_object` | — | **IGNORED** | **0/8** — indistinguishable from sending no constraint at all (control arm: 0/8) |
| `reasoning_effort` | — | honoured | `"none"` → 0 ch reasoning + real content; unset → 525 ch reasoning + **empty content**. This is the 2026-08-03 outage, reproduced |
| `temperature`, `seed`, `stop`, `max_tokens` | honoured | honoured | `seed=7` at temp 1.6 gave 1 distinct on both arms |
| final-token stats | **unconditional** on the last stream chunk (`prompt_eval_count`, `eval_count`, `done_reason`) | requires `stream_options.include_usage` | — |

**⚠ The headline correction.** The roadmap entry states: *"JSON-schema
constrained decoding is native-only too (Ollama's `format`; the OpenAI
`response_format` is ignored upstream)."* **That is false.** `response_format:
{"type": "json_schema", …}` binds exactly as hard as native `format` — measured
on the schema shape codex extraction actually needs (array of triplets with an
enum-constrained relation), 8/8 on both, and both obeyed an enum that excluded
every verb the text implied. What *is* ignored is `json_object`, which is the
weaker of the two and the one ICE happens to send.

### 0.4 Truncation — the corrected finding

Ollama 0.30.7 does **not** truncate by default: when a prompt exceeds the
resident runner's window it **reloads the runner at a larger window**. Measured:
a runner pinned at 2,048 received a ~6,012-token prompt and came back resident
at 32,768, answering normally, on both transports.

Silent truncation happens only when the model **cannot** grow — i.e. the prompt
exceeds the model's *architectural* ceiling. On `tinyllama` (GGUF max 2,048), a
6,012-token prompt returned **HTTP 200, `done_reason:"stop"`, and
`prompt_eval_count: 2047` on BOTH transports.** ~4,000 tokens vanished with no
error and no flag, natively too. This is C16's `predicted=2909 / actual=2047`,
and it explains it: tinyllama was the model that could not grow.

**So the native endpoint is not inherently louder about truncation.** What makes
it loud is the *combination*: send `options.num_ctx` explicitly and an oversized
prompt becomes a typed refusal —

```
HTTP 400 {"error":{"code":400,"type":"exceed_context_size_error",
  "message":"request (6012 tokens) exceeds the available context size (2048 tokens)…",
  "n_prompt_tokens":6012,"n_ctx":2048}}
```

`n_prompt_tokens` and `n_ctx` are exactly the two numbers ICE's budget
arithmetic wants to check itself against. And `num_ctx` is the parameter `/v1`
drops — so this capability is genuinely unreachable through the shim, but as a
*pair*, not as a property of the endpoint.

### 0.5 `/api/show` → `capabilities` is the detection primitive

The entry asks for "capability detection, not a hardcoded branch". It exists:

```
gemma4:26b-a4b-it-q4_K_M  ['completion', 'vision', 'tools', 'thinking']
qwen3:4b-instruct-bg      ['tools', 'thinking', 'completion']
gpt-oss:latest            ['completion', 'tools', 'thinking']
```

⚠ **But it is a template-derived claim, not a behavioural one.**
`qwen3:4b-instruct-bg` advertises `thinking` and emits **0 characters** of it
under both `think:true` and `think:false`. Treat `capabilities` as a
*negative* filter (absent ⇒ definitely unsupported) and never as a positive
guarantee.

### 0.6 Three live defects found by the audit, all transport-independent

1. **`maintenance_agent.py:399` sends `response_format={"type":"json_object"}`**
   — measured **0/8**, identical to sending nothing. ICE believes it is
   constraining the decoder and is not. One-word fix (`json_schema` + a schema),
   no transport change.
2. **`registry.py:162` hardcodes `model="Qwen/Qwen2.5-3B-Instruct-AWQ"`** — the
   *dedicated*-mode default — while `get_bg_client()` in the default *shared*
   mode points at Ollama, which returns `404 model not found`. The caller's
   `except Exception: return {"topic_tags": [], "intent_tags": []}` swallows it.
   Background model auto-tagging has been dead in the default configuration, and
   silently: the CLAUDE.md silent-fallback rule, a third instance.
3. **`ollama_send_num_ctx` defaults `False`, and would be dropped if it were
   `True`.** C16 shipped this believing it landed. Both halves are now measured.

### 0.7 What dropping the OpenAI SDK would cost — confirmed narrow

The entry's grounding holds. The chat path uses raw `httpx` and never touches
the SDK; the background path constructs it at **exactly one site**
(`bg_client_factory.py:98`), consumed by **18 call sites** that all pass the
same five arguments (`model`, `messages`, `temperature`, `max_tokens`,
`timeout`) — plus `response_format` at exactly one (`maintenance_agent.py:399`).
Only `background_model_mode="dedicated"` (vLLM/SGLang on `:8002/v1`, which does
not speak Ollama-native) requires the SDK branch to survive.

---

## 1. Decisions

### 1.1 OPEN — the user's call: does (a) still swap the transport?

The audit changes the cost/benefit the entry was written against. Stated
plainly, because the entry's own justification no longer holds:

- The entry's headline prize for (a) was **constrained decoding** — "what makes
  A12's one-small-model design affordable and kills the regex JSON fallback in
  `extract_triplets`". **That prize does not require the native transport.** It
  is `response_format={"type":"json_schema", …}` on the existing SDK client.
- What *does* require native: `num_ctx`, `keep_alive` + explicit unload
  (**G4(b)'s mechanism**), `options.*` sampling knobs, and the typed
  over-context refusal.
- Of those, the background path has a real use for `keep_alive` (release the bg
  model between drains) and little use for `num_ctx` (background prompts are
  small and bounded). The `num_ctx` need lives on the **chat** path — G32(b).

**Options, with the recommendation first:**

| | Option | Gets | Costs |
|---|---|---|---|
| **A** *(recommended)* | **Split (a) in two.** (a1) constrained decoding + the three §0.6 defects, on the existing SDK. (a2) the native transport, scoped to what only it can do, folded into G4(b)'s VRAM work where `keep_alive` is actually consumed. | A12's benchmark unblocks after a1 — a small, contained change | two commits instead of one; the native client lands later |
| **B** | Build the capability-aware native transport now, as written. | one seam, done once | the larger change lands to buy `keep_alive`, whose consumer (G4(b)) is deferred to Z1 |
| **C** | a1 only; close (a); re-home the native transport entirely into G4(b). | smallest step | G32's audit findings lose their home |

**Recommendation: A.** It puts A12's unblocking on the smallest change that
achieves it, and it stops the native transport being built to serve a consumer
that Z1 owns. The capability-reporting requirement — *unavailable capabilities
LOGGED, never silently dropped* — moves to a2 and still gets built; F11's cloud
models still land on the SDK branch and inherit it.

*(Everything below is written for whichever option is chosen; §2–§5 pend that
decision.)*
