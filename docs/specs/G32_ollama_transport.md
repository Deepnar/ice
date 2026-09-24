# G32 — ICE↔Ollama control surface: the audit, and what it implies
Assumes decided specs: none

> **Status:** §0 is the historical Ollama 0.30.7 audit. The user subsequently
> authorized whole-system v3 repair and asked the implementation session to
> choose routine designs; (a1) shipped on the compatibility SDK, while native
> background residency control remains with G4. §2 below settles G32(b), the
> foreground transport, against the running Ollama 0.32.13 API.
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

## 2. v3 foreground transport decision — 2026-09-23

**Chosen split:** keep the existing OpenAI-compatible path for a registry model
with a nonlocal `base_url` (and the later optional cloud-answering switch). The
default local Ollama foreground path uses `/api/chat`, whose native stream is
newline-delimited JSON. A focused live 0.32.13 request confirmed content chunks
and a final `done=true` object with `prompt_eval_count`, `eval_count` and
`done_reason`. [Ollama's API reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
confirms native `options`, and its [OpenAI compatibility guide](https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx)
says the compatibility endpoint cannot set context size per request.

The transport adapter converts native content chunks into the same OpenAI SSE
delta frames the client and post-flight parser already consume. Its final frame
carries translated usage, followed by `[DONE]`; reasoning/thinking content is
never inserted into the answer text. Unknown/invalid native frames or a stream
ending without `done=true` are failures, not completed turns. Compatibility SSE
passes through unchanged but must also end with `[DONE]`. A failed generation
does not write an empty or partial assistant turn to the memory store.

For local Ollama, `ollama_send_num_ctx` defaults on. `fit` requests the estimated
prompt with the configured tokenizer safety margin plus generation reserve;
`max` requests the configured ceiling. Both are capped by the known serving
window and `ollama_num_ctx_max`; the native endpoint makes over-context errors
typed rather than silently truncating. Keep the opt-out for constrained hosts,
but log when it is used. Never send Ollama-native `options` through `/v1`.
The selected fallback model uses the local transport and base URL, even when
the primary model was configured on an external provider.

**Validation:** parser controls for split NDJSON, content/thinking separation,
terminal usage and missing/invalid terminal frames; mocked HTTP status errors;
two-sided request-body controls for `fit`, `max`, cap and compat; in-process chat
control for SSE and post-flight success/failure; one live native completion on
the running Ollama instance. Preserve the existing G5 truncated-SSE and C16
usage tests. G4 still owns foreground/background model residency policy.

**v3 implementation check, 2026-09-24:** a live Ollama 0.32.13 native request
returned a complete answer, terminal usage and the requested `num_ctx`; the
in-process foreground route exercised successful storage, refusal to store a
failed stream, and external-primary-to-local fallback model attribution.
Disposable-database smoke/source regression passed 339/339 and settings freeze
passed 140/140. These are transport and lifecycle checks, not an answer-quality
measurement or a guarantee against a model's architectural context ceiling.
