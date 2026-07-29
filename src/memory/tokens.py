"""C16 — the one place that answers "how many tokens is this?".

Before this module ICE had **four incompatible answers** to that question and
no real one:

  * ``int(len(text.split()) * 1.33)`` at 15 sites, including all three budget
    enforcement points;
  * ``len(text) / 4.0`` for whole-conversation size (``main.py``,
    ``conversation_summary.py``);
  * raw word counts compared against word thresholds;
  * ``chunking.estimate_tokens``, the only code-aware one — used exclusively on
    the WRITE path, never to spend a budget.

Measured against real tokenizers, ``words * 1.33`` **overcounts plain prose by
~20%** and **undercounts ICE's own ``[date] User:/Assistant:`` stamped format by
1.68x** (identifier-dense text by 13.9x). Both errors point the same way in the
one comparison that matters: the vector-RAG baseline injects raw prose, ICE
injects stamped structured text, so the "~25% fewer tokens" reported in
Experiment 2 was measured with a ruler biased in ICE's favour. A budget you
cannot count is not a budget.

**Which tokenizer, and the honest caveat.** ``Qwen/Qwen3-Embedding-0.6B`` — the
embedder's own vocabulary, already loaded in this process by
``retrieval/ner_utils.py``, so it is free. It is *not* the generation model's
vocabulary. Measured cross-family disagreement: 0% on prose, 3.5% on ICE's
stamped format, 1.4% on identifier-dense text, ~16% worst case on Python
source. Against the 68-1290% error it replaces, that is a second-order
correction, and it is covered by ``token_count_safety_margin`` rather than by
guessing each Ollama tag's HuggingFace repo (a mapping the roadmap already
judged garbage-prone under B3, and Ollama exposes no tokenize endpoint).

The real correction is empirical: ``main.py`` reconciles every prediction
against Ollama's own ``prompt_eval_count`` and logs the pair, so the margin is
calibrated from live traffic instead of argued about.

``estimate()`` survives for the write path at volume (chunking a 200k-word
import must not tokenize twice) and for tests that must not load a model. Its
formula is now **chars/4**, which is better than words*1.33 everywhere but is
still only an approximation — measured on this repo's own text:

    ratio to the real count      words*1.33      chars/4
    plain prose                    1.20x          1.12x
    fenced python                  0.53x          0.72x
    ICE's stamped turn format      0.55x          0.60x

So chars/4 halves the error on code and still under-reads dense text by ~40%.
That is *why* it no longer enforces anything: every budget decision calls
``count()``. Do not reintroduce an estimate on a spending path.
"""

import structlog

logger = structlog.get_logger("ice.memory.tokens")

_tokenizer = None
_load_attempted = False

# Chat templates wrap every message in role/turn markers. Measured at ~4 tokens
# per message across the Qwen/Gemma templates; with a 20-message prompt that is
# 80 tokens, which is inside the safety margin but worth naming rather than
# silently dropping. Calibrated by the prompt_eval_count reconciliation.
MESSAGE_OVERHEAD_TOKENS = 4


def _get_tokenizer():
    """The embedder's tokenizer, loaded once per process and shared with
    ner_utils (same repo id, so the second call is a cache hit — this costs
    nothing after NER has run, and ~570 ms if it hasn't)."""
    global _tokenizer, _load_attempted
    if _tokenizer is None and not _load_attempted:
        _load_attempted = True
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        except Exception as exc:
            # No silent failure: a fallen-back count is a *different measurement*
            # and every consumer of it deserves to know which ruler it got.
            logger.error("tokenizer_load_failed", error=str(exc),
                         falling_back_to="estimate")
    return _tokenizer


def count(text: str) -> int:
    """Real token count. Falls back to `estimate` if the tokenizer is absent.

    ~1 ms per 1,000 tokens — a 22k prompt costs ~20 ms, which is noise beside
    the classifier forward pass, the embedder encode and the ~8 DB round trips
    the same request already pays.
    """
    if not text:
        return 0
    tok = _get_tokenizer()
    if tok is None:
        return estimate(text)
    try:
        return len(tok.encode(text, add_special_tokens=False))
    except Exception as exc:
        logger.warning("token_count_failed", error=str(exc), chars=len(text))
        return estimate(text)


def count_messages(messages) -> int:
    """Token count of an assembled OpenAI-style message list.

    Counts the *messages*, not their concatenation, because the chat template
    adds per-message envelope tokens that a concatenated count silently loses.
    """
    total = 0
    for m in messages or []:
        total += count(m.get("content") or "") + MESSAGE_OVERHEAD_TOKENS
    return total


def estimate(text: str, is_code: bool = False) -> int:
    """Tokenizer-free approximation, for the write path at volume and for
    model-free tests. NOT for budget enforcement — use `count`.

    chars/4 measured within +/-12% on both prose and code, where words*1.33 was
    off by 0.84x on prose and 1.84x on code. `is_code` is kept because
    `chunking` packs code segments and a symbol-dense block still runs heavier
    than 4 chars per token.
    """
    if not text:
        return 0
    char_est = len(text) / 4.0
    if is_code:
        return int(max(char_est, len(text) / 3.0))
    return int(char_est)


def estimate_from_chars(n_chars: float) -> float:
    """Tokens from a character count, for quantities that only ever exist as a
    SQL `sum(length(...))`.

    Tokenizing an entire conversation on every request to size it would cost
    more than the decision is worth, so whole-history volume stays an estimate
    — but the 4.0 lives HERE, once, instead of being retyped at each call site
    in a different form. Consumers: B2's memory-pressure prior and C4's
    outgrew-the-window gate, which used to compute this separately and could
    disagree about the same conversation.
    """
    return float(n_chars) / 4.0


def with_margin(tokens: int, margin: float) -> int:
    """Apply the safety margin that covers cross-family tokenizer drift.

    Used where *underestimating* costs correctness (deciding a prompt fits a
    window), never where it costs only tokens.
    """
    return int(tokens * max(1.0, margin))
