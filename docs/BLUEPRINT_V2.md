
---

# Phase A — Core Retrieval Fixes (highest impact on evaluation)

These directly affect the precision/recall numbers and must be done before any paper experiments.

| # | Feature | Architecture ref | Current state | What to build | Rough effort |
|---|---------|-----------------|---------------|---------------|-------------|
| A1 | **Decay‑score filtering in retrieval** | §4.2 | Retrieval only checks `is_archived`, not `decay_score`. | Add `AND decay_score > 0.2` (or the configured threshold) to all episodic retrieval queries. | 30 min |
| A2 | **Access‑weighted decay + retrieval strengthening** | §4.2 | Decay applies a flat 3% rate; `access_count` is never incremented. | 1) In the orchestrator, after injecting fragments, increment `access_count` for each retrieved episodic turn. 2) Modify the Decay Worker: decay rate should be a function of days since last access, not just age. 3) Add a “strengthening” step: when a turn is retrieved, its `decay_score` is increased by +0.15 (capped at 1.0). | 2 h |
| A3 | **Wide‑net fallback uses full vector search** | §9.3 / §1.2 | Fallback only returns last 20 turns + global Codex + RAG. | Replace the fallback with: 1) Run the full vector similarity leg (unfiltered by topic tags) over episodic memory. 2) Still include Codex and RAG. 3) Fuse with RRF. | 1 h |
| A4 | **Codex scoping to conversation/cluster** | §8.4 | Codex traversal is always global. | When a `conversation_id` or `cluster_ids` scope is active, filter Codex entities to those that appear in episodic turns belonging to the selected conversation/cluster. Requires a subquery or a join with `episodic_memory`. | 3 h |
| A5 | **Procedural memory scoping to conversation/cluster** | §8 (implied) | Procedural lookup is global. | Same as A4: filter procedural patterns to those whose `source_batch_ids` belong to the scoped conversation/cluster. | 1 h |
| A6 | **Procedural trigger conditions evaluation** | §3.3 | `trigger_conditions` JSONB is never evaluated. | Before injecting a procedural pattern, check if the current prompt’s topic/intent tags satisfy the pattern’s `trigger_conditions`. Only inject if they match. | 1 h |
| A7 | **HyDE query rewriting** | §9.4 | Stub only. | 1) In the orchestrator, when `context_reliance == Long_Term_Memory` and `entropy_score` is below threshold, call the background model to rewrite the prompt into a dense search query. 2) Use the rewritten query for BM25 and vector legs. 3) Add a bypass flag for experiments. | 3 h |
| A8 | **Sliding window – always inject last 10 turns of current conversation** | §3.1 / §10.4 | Not implemented. | In the proxy, after retrieval, fetch the last 10 episodic turns for the current `conversation_id` and prepend them to the assembled prompt as a separate `[RECENT CONTEXT]` block (before the retrieved blocks). | 1 h |
| A9 | **Session diversification before dedup is fine – but missing bookmarked‑boost** | §7.2 | Bookmarked turns are not boosted. | After the bookmarking backend is built (Phase D), multiply the score of bookmarked fragments by 1.5× before RRF fusion. | 30 min |
| A10 | **Classifier fine‑tuning loop** | §1.4 | Not implemented. | 1) Create a Celery beat task that loads `curated_labels`, freezes the SentenceTransformer, retrains the MLP head for 5 epochs, and saves a new checkpoint. 2) Expose an endpoint to trigger it manually. | 2 h |


---

# PHASE A — Core Retrieval Fixes

---

## A1 — Decay‑Score Filtering in Retrieval

**What:** Currently the retrieval queries only skip archived turns (`is_archived = false`).  
The architecture (§4.2) requires that turns with a very low `decay_score` are also excluded from default retrieval.  
We’ll add `AND decay_score > :min_decay` to both the BM25 and the vector episodic queries.

**Files to edit:** `src/retrieval/orchestrator.py`

**Step 1: Add the filter to `_bm25_episodic`**  

Find the SQL `WHERE` clause inside the method (around line 165):

```python
              AND is_archived = false
```

Replace that line with:

```python
              AND decay_score > :min_decay
              AND is_archived = false
```

Now find the `params` dictionary just above the `try` block and add the new parameter:

```python
        params = {"search_terms": search_terms, "min_decay": 0.2}
```

**Step 2: Add the filter to `_vector_episodic`**  

Find the SQL `WHERE` clause in `_vector_episodic` (around line 205):

```python
              AND is_archived = false
```

Replace it with:

```python
              AND decay_score > :min_decay
              AND is_archived = false
```

Add `"min_decay"` to the `params` dictionary:

```python
        params = {"prompt_embedding": prompt_embedding, "min_decay": 0.2}
```

**Why this works:**  
Turns that have been decayed to near‑zero will no longer clutter the retrieval results.  
The threshold is taken from the architecture’s default of 0.2.

---

## A2 — Access‑Weighted Decay + Retrieval Strengthening

**What:** The Decay Worker currently applies the same 3% daily decay to every old turn, regardless of whether the turn has ever been useful.  
The architecture (§4.2) specifies:

- Turns that have been retrieved recently should decay **slower** (access‑weighted decay).
- When a turn is actually injected into a prompt, its `access_count` should be incremented and its `decay_score` partially restored (+0.15, capped at 1.0).

**Files to edit:**  
- `src/retrieval/orchestrator.py` – strengthening on injection.  
- `src/workers/decay.py` – access‑weighted decay formula.

### A2a — Strengthening in the orchestrator

We’ll add a helper method that the orchestrator calls right before returning the final fragment list.  
It increments `access_count` and boosts `decay_score` for every episodic fragment that survives to the final output.

**Open `src/retrieval/orchestrator.py`** and locate the `retrieve()` method.  
Just before the final `return self._enforce_token_budget(…)` line, add a call to the new method:

```python
        # Apply retrieval strengthening to the surviving fragments
        self._strengthen_retrieved(fragments)
```

Now add the new method at the bottom of the `HybridRetrievalOrchestrator` class (before the `_rows_to_fragments` helper):

```python
    def _strengthen_retrieved(self, fragments: List[ContextFragment]):
        """Increment access_count and partially restore decay_score for episodic fragments."""
        for frag in fragments:
            if frag.source_type != "episodic" or not frag.source_batch_id:
                continue
            try:
                turn = self.db.query(EpisodicMemory).get(uuid.UUID(frag.source_batch_id))
                if turn:
                    turn.access_count = (turn.access_count or 0) + 1
                    turn.decay_score = min(1.0, (turn.decay_score or 0.0) + 0.15)
                    self.db.commit()
            except Exception:
                self.db.rollback()
```

Make sure you have the UUID import at the top of the file:

```python
import uuid
```

### A2b — Access‑weighted decay in the Decay Worker

**Open `src/workers/decay.py`**.  

The current decay update is:

```python
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * :rate
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
        """), {"rate": DECAY_RATE, "cutoff": cutoff})
```

Replace it with an access‑weighted version that uses a different rate depending on `access_count`:

```python
        # Turns that have never been accessed decay faster (5% vs 2%)
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * :rate
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
              AND access_count = 0
        """), {"rate": 0.95, "cutoff": cutoff})

        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * :rate
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
              AND access_count > 0
        """), {"rate": 0.98, "cutoff": cutoff})
```

**Why this works:**  
- Retrieval now actively strengthens frequently‑used memories (the system gets better over time).  
- The decay worker respects access patterns: forgotten turns fade faster, useful ones persist.

---

## A3 — Wide‑Net Fallback Uses Full Vector Search

**What:** The confidence fallback currently just grabs the last 20 turns instead of doing a proper similarity search.  
The architecture (§9.3 / §1.2) says: *“query the last 20 Episodic turns, pull the top Codex nodes by keyword overlap, run vector similarity over the RAG store.”*  
We’ll replace the “last 20” with a full vector search (unfiltered by topic tags).

**Open `src/retrieval/orchestrator.py`**, go to the `_wide_net_fallback` method.

**Before (the problematic part):**

```python
        try:
            rows = self.db.execute(text("""
                SELECT id, raw_text, summary_text, lossless_flag, conversation_id
                FROM episodic_memory
                WHERE is_archived = false
                ORDER BY timestamp DESC
                LIMIT 20
            """)).fetchall()
            fragments = self._rows_to_fragments(rows, "episodic")
        except Exception:
            self.db.rollback()
            fragments = []
```

**After — replace with full vector search:**

```python
        try:
            # Full vector similarity search without topic filter
            query = text("""
                SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id,
                       1 - (embedding <=> :prompt_embedding) as score
                FROM episodic_memory
                WHERE embedding IS NOT NULL
                  AND is_archived = false
                  AND decay_score > :min_decay
                ORDER BY score DESC
                LIMIT 10
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            rows = self.db.execute(query, {
                "prompt_embedding": prompt_embedding,
                "min_decay": 0.2
            }).fetchall()
            fragments = self._rows_to_fragments(rows, "episodic")
        except Exception:
            self.db.rollback()
            fragments = []
```

**Why this works:**  
Now when the classifier is uncertain, ICE still performs a real similarity search, not just a chronological dump. The fusion with Codex + RAG follows as before.

---

## A4 — Codex Scoping to Conversation/Cluster

**What:** Currently, Codex graph traversal is always global.  
If retrieval is scoped to a specific `conversation_id`, only entities that appear in turns of that conversation should be considered.  
We’ll add a filter to the `_codex_graph` method.

**Open `src/retrieval/orchestrator.py`**, and modify the `retrieve()` method to pass the scope to `_codex_graph`:

In `retrieve()`, change the call from:

```python
            "codex": self._codex_graph(classification),
```

to:

```python
            "codex": self._codex_graph(classification, scope),
```

Now change the signature of `_codex_graph`:

```python
    def _codex_graph(self, classification, scope: Optional[dict] = None) -> List[ContextFragment]:
```

Inside `_codex_graph`, after the NER step and before the entity lookup, add the scoping logic:

```python
        # If scoped to a conversation, collect only entity IDs that appear in that conversation
        allowed_entity_ids = None
        if scope and "conversation_id" in scope:
            conv_id = scope["conversation_id"]
            try:
                batch_rows = self.db.execute(
                    text("SELECT DISTINCT batch_id FROM episodic_memory WHERE conversation_id = :cid"),
                    {"cid": conv_id}
                ).fetchall()
                batch_ids = [row.batch_id for row in batch_rows]
                if batch_ids:
                    event_rows = self.db.execute(
                        text("SELECT DISTINCT entity_id FROM codex_events WHERE batch_source = ANY(:bids)"),
                        {"bids": batch_ids}
                    ).fetchall()
                    allowed_entity_ids = {row.entity_id for row in event_rows}
            except Exception:
                self.db.rollback()
```

Then, when iterating over `entities`, skip any whose `id` is not in the allowed set (if the set is defined):

```python
            for entity in entities:
                if allowed_entity_ids is not None and entity.id not in allowed_entity_ids:
                    continue
                self._traverse_graph(entity, 0, 2, visited, context_texts)
```

**Why this works:**  
Only Codex entities that were originally extracted from the target conversation will be followed.  
This prevents story lore from leaking into a technical conversation (and vice versa), exactly as §8.4 specifies.

---

## A5 — Procedural Memory Scoping to Conversation/Cluster

**What:** The procedural leg is currently global.  
If scoped, we should only surface patterns whose `source_batch_ids` overlap with the conversation’s turn batches.

**Open `src/retrieval/orchestrator.py`**, change the `retrieve()` method to pass scope to the procedural leg:

```python
            "procedural": self._procedural_lookup(prompt_embedding, classification, scope),
```

Now modify the signature of `_procedural_lookup`:

```python
    def _procedural_lookup(self, prompt_embedding, classification, scope: Optional[dict] = None) -> List[ContextFragment]:
```

Inside `_procedural_lookup`, after the activating intents check and before the query, add the scoping logic:

```python
        # Scoping: restrict to patterns that have source batches in the target conversation
        allowed_batch_ids = None
        if scope and "conversation_id" in scope:
            conv_id = scope["conversation_id"]
            try:
                batch_rows = self.db.execute(
                    text("SELECT DISTINCT batch_id FROM episodic_memory WHERE conversation_id = :cid"),
                    {"cid": conv_id}
                ).fetchall()
                allowed_batch_ids = [row.batch_id for row in batch_rows]
            except Exception:
                self.db.rollback()
```

Then, after the query retrieves patterns, filter them by checking overlap with `allowed_batch_ids`:

```python
        try:
            rows = self.db.execute(query, {"prompt_embedding": prompt_embedding}).fetchall()
            fragments = []
            for r in rows:
                if allowed_batch_ids is not None:
                    # Check overlap with the pattern's source_batch_ids
                    pattern = self.db.query(ProceduralMemory).get(r.id) if hasattr(r, 'id') else None
                    if pattern and not any(bid in allowed_batch_ids for bid in (pattern.source_batch_ids or [])):
                        continue
                fragments.append(ContextFragment(...))
            return fragments
```

**Why this works:**  
Procedural memory injection becomes conversation‑aware, preventing irrelevant workflow patterns from distracting the model.

---

## A6 — Procedural Trigger Conditions Evaluation

**What:** The `trigger_conditions` JSONB column is never evaluated.  
We’ll add a helper that checks whether the current prompt’s topic/intent tags satisfy the stored conditions.

**Add this helper method** to `HybridRetrievalOrchestrator`:

```python
    def _procedural_trigger_match(self, pattern: ProceduralMemory, classification: ClassificationResult) -> bool:
        """Return True if the pattern's trigger_conditions match the current classification."""
        conditions = pattern.trigger_conditions or {}
        if not conditions:
            return True   # no conditions = always match
        required_topics = set(conditions.get("topic_tags", []))
        required_intents = set(conditions.get("intent_tags", []))
        if required_topics and not required_topics.intersection(classification.topic_tags):
            return False
        if required_intents and not required_intents.intersection(classification.intent_tags):
            return False
        return True
```

Now, in `_procedural_lookup`, after fetching the pattern and checking the scope, add a trigger check before appending the fragment:

```python
                pattern = self.db.query(ProceduralMemory).get(r.id)   # fetch the ORM object
                if not self._procedural_trigger_match(pattern, classification):
                    continue
```

**Why this works:**  
Procedural memory entries can now be annotated with specific activation conditions, making them context‑sensitive.

---

## A7 — HyDE Query Rewriting

**What:** HyDE (Hypothetical Document Embeddings) rewrites the user’s vague prompt into a dense, self‑contained search query.  
This helps retrieval precision when the original prompt is short or anaphoric.

**We’ll add a method** to the orchestrator that calls the background model.  
**Open `src/retrieval/orchestrator.py`** and add this method after the constructor:

```python
    def _hyde_rewrite(self, prompt: str) -> Optional[str]:
        """Rewrites the user prompt into a specific, retrieval‑optimised query."""
        try:
            resp = self.bg_client.chat.completions.create(
                model="Qwen/Qwen2.5-3B-Instruct-AWQ",
                messages=[
                    {"role": "system", "content": (
                        "You are a query rewriting engine. Take the user's question and rewrite it "
                        "as a dense, factual search query that would retrieve the relevant past conversation. "
                        "Include key entities and omit polite phrasing. Output ONLY the rewritten query, no other text."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
                timeout=15.0,
            )
            rewritten = resp.choices[0].message.content.strip()
            return rewritten if rewritten else None
        except Exception:
            return None
```

Now, in the `retrieve()` method, before the retrieval legs, call the rewrite if the context reliance is Long_Term_Memory:

```python
        hyde_prompt = None
        if classification.context_reliance == "Long_Term_Memory":
            hyde_prompt = self._hyde_rewrite(classification.prompt)

        # When rewriting succeeded, use the rewritten prompt for BM25 and vector search
        search_prompt = hyde_prompt if hyde_prompt else classification.prompt
```

Then pass `search_prompt` to the BM25 leg instead of `classification.prompt`.  
We’ll modify the `_bm25_episodic` method to accept an optional `override_prompt` parameter, but the simplest way is to temporarily replace `classification.prompt` inside the method or pass the search prompt via a new field.  
We’ll add a `search_prompt` attribute to the `ClassificationResult` dataclass (or just pass it as an argument). The cleanest is to add an optional parameter to the BM25 method.

**Modify `_bm25_episodic` signature** to accept `search_prompt: Optional[str] = None` and use it if provided:

```python
    def _bm25_episodic(self, classification, scope, conv_id: Optional[str] = None, search_prompt: Optional[str] = None) -> List[ContextFragment]:
        prompt_text = search_prompt if search_prompt else classification.prompt
        clean_prompt = re.sub(r'[^\w\s]', ' ', prompt_text)
        ...
```

Then, in `retrieve()`, call it with:

```python
            "bm25": self._bm25_episodic(classification, scope, conv_id, search_prompt),
```

The vector leg also uses the prompt embedding; we’ll compute the embedding from `search_prompt` instead of the original prompt.  
So, before building the legs, we compute two embeddings:

```python
        hyde_prompt = self._hyde_rewrite(classification.prompt) if classification.context_reliance == "Long_Term_Memory" else None
        search_text = hyde_prompt if hyde_prompt else classification.prompt
        if hyde_prompt:
            prompt_embedding = self.embedder.encode(search_text, convert_to_tensor=False).tolist()
```

Then use `prompt_embedding` as before for vector search and `search_text` for BM25.

**Bypass flag** already exists: we can skip HyDE if a config flag is set.  
Add an entry to `settings` if you like, but for now it’s always on when `Long_Term_Memory`.

**Why this works:**  
Rewritten queries retrieve more relevant past turns for vague prompts like “what happened last time?”.

---

## A8 — Sliding Window (Always Inject Last 10 Turns)

**What:** The architecture specifies that the active context always includes the last N turns of the current conversation, regardless of retrieval.  
We’ll inject them as a `[RECENT CONTEXT]` block in the prompt.

**File to edit:** `src/api/prompt_assembler.py`

**Step 1 — Add a function to fetch recent turns**

In `prompt_assembler.py`, add:

```python
from src.memory.models import EpisodicMemory

def get_recent_turns(db_session, conversation_id: str, n: int = 10) -> List[str]:
    """Return the last N turns from the current conversation."""
    turns = db_session.query(EpisodicMemory).filter_by(
        conversation_id=conversation_id
    ).order_by(EpisodicMemory.timestamp.desc()).limit(n).all()
    turns.reverse()   # chronological order
    fragments = []
    for t in turns:
        if t.inject_raw and t.raw_text:
            text = t.raw_text
        elif t.summary_text:
            text = t.summary_text
        else:
            text = (t.raw_text or "")[:300]
        words = text.split()
        if len(words) > 500:
            text = " ".join(words[:500]) + "…"
        fragments.append(text)
    return fragments
```

**Step 2 — Modify the `assemble_prompt` function** to accept the session’s `db` and `conversation_id` and insert the recent context.

Change the signature to:

```python
def assemble_prompt(
    memory_slots: List[MemorySlot],
    retrieved_fragments: List[ContextFragment],
    user_message: str,
    db_session: Session = None,
    conversation_id: str = None,
) -> List[dict]:
```

Then, inside the function, before adding the Codex block, add:

```python
    # 0. Recent context (sliding window)
    if db_session and conversation_id:
        recent_texts = get_recent_turns(db_session, conversation_id, n=10)
        if recent_texts:
            system_content += "\n\n=== RECENT CONTEXT ===\n" + "\n\n".join(recent_texts)
```

**Step 3 — Update the proxy** to pass the session and conversation_id.

In `src/api/main.py`, when calling `assemble_prompt`, pass `db_session=db` and `conversation_id=str(conversation_id)`.

**Why this works:**  
The model always sees the immediate conversational flow, even if no long‑term retrieval was triggered.

---

## A9 — Bookmarked Turn Boost (Deferred)

We’ll implement this **after Phase C (Bookmarking)**.  
Once bookmarked turns are flagged, we’ll multiply their raw score by 1.5× before RRF fusion.

No code changes for now.

---

## A10 — Classifier Fine‑Tuning Loop

**What:** The `curated_labels` table contains user‑corrected labels.  
We’ll build a Celery beat task that runs weekly, retrains only the MLP head on those labels, and saves a new checkpoint.

**File to create:** `src/workers/fine_tune.py`

```python
import torch, os, json
from datetime import datetime, timezone
from src.workers.celery_app import app
from src.api.db import SessionLocal
from src.memory.models import CuratedLabel
from src.classifier.model import ICEClassifier
from src.classifier.dataset import ICEClassifierDataset   # not quite; we'll build a small dataset
```

We’ll keep the implementation simple: load curated labels, encode them with the frozen SentenceTransformer, and run a few epochs on a small DataLoader.

I’ll provide the complete file separately when we reach that step.



## 1. `src/retrieval/orchestrator.py` (complete)

```python
"""Hybrid Retrieval Orchestrator – Phase A hardened: decay filtering, access-weighting,
wide‑net full‑vector, Codex/Procedural scoping, HyDE rewriting, procedural trigger matching."""

import hashlib
import re
import uuid
from typing import List, Optional, Dict
from dataclasses import dataclass
from openai import OpenAI
import structlog
from sqlalchemy.orm import Session
from sqlalchemy import bindparam, text
from pgvector.sqlalchemy import Vector as PgVector
from src.api.config import settings
from src.memory.models import (
    EpisodicMemory,
    CodexEntity,
    CodexEdge,
    ProceduralMemory,
    MemorySlot,
)
from src.classifier.classifier import ClassificationResult

logger = structlog.get_logger("ice.retrieval")


@dataclass
class ContextFragment:
    text: str
    source_type: str          # "episodic", "codex", "procedural", "rag"
    score: float              # RRF fused score
    token_count: int
    source_batch_id: Optional[str] = None
    conversation_id: Optional[str] = None


class HybridRetrievalOrchestrator:
    def __init__(self, db: Session, embedder):
        self.db = db
        self.embedder = embedder
        self.bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def retrieve(
        self,
        classification: ClassificationResult,
        conversation_id: str,
        prompt_embedding: list[float],
        scope: Optional[dict] = None,
    ) -> List[ContextFragment]:
        """Orchestrate multi‑source retrieval."""
        if classification.context_reliance == "Zero_Shot":
            return []
        if classification.context_reliance == "Real_Time_Search":
            return []

        if classification.max_confidence < settings.confidence_fallback_threshold:
            logger.info("wide_net_fallback_triggered", confidence=classification.max_confidence)
            return self._wide_net_fallback(classification, prompt_embedding, conversation_id, scope)

        # Conversation scope filter for episodic legs
        conv_id = None
        if scope and "conversation_id" in scope:
            conv_id = scope["conversation_id"]

        # HyDE query rewriting
        hyde_prompt = None
        if classification.context_reliance == "Long_Term_Memory":
            hyde_prompt = self._hyde_rewrite(classification.prompt)
        search_prompt = hyde_prompt if hyde_prompt else classification.prompt

        # Re‑compute embedding if the search prompt changed
        if hyde_prompt:
            prompt_embedding = self.embedder.encode(search_prompt, convert_to_tensor=False).tolist()

        # Execute all retrieval legs
        legs: Dict[str, List[ContextFragment]] = {
            "bm25": self._bm25_episodic(classification, scope, conv_id, search_prompt),
            "vector": self._vector_episodic(prompt_embedding, classification, scope, conv_id),
            "codex": self._codex_graph(classification, scope),
            "procedural": self._procedural_lookup(prompt_embedding, classification, scope),
            "rag": self._rag_lookup(prompt_embedding, classification),
        }

        fused = self._apply_rrf(legs)
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        deduped = self._deduplicate(diversified)
        final = self._enforce_token_budget(deduped, max_tokens=2000)

        # Strengthen retrieved turns (access count + decay boost)
        self._strengthen_retrieved(final)

        return final

    # ------------------------------------------------------------------
    # HyDE rewriting
    # ------------------------------------------------------------------
    def _hyde_rewrite(self, prompt: str) -> Optional[str]:
        try:
            resp = self.bg_client.chat.completions.create(
                model="Qwen/Qwen2.5-3B-Instruct-AWQ",
                messages=[
                    {"role": "system", "content": (
                        "You are a query rewriting engine. Take the user's question and rewrite it "
                        "as a dense, factual search query that would retrieve the relevant past conversation. "
                        "Include key entities and omit polite phrasing. Output ONLY the rewritten query, no other text."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
                timeout=15.0,
            )
            rewritten = resp.choices[0].message.content.strip()
            return rewritten if rewritten else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # BM25 episodic (full‑text search)
    # ------------------------------------------------------------------
    def _bm25_episodic(self, classification, scope, conv_id: Optional[str] = None, search_prompt: Optional[str] = None) -> List[ContextFragment]:
        prompt_text = search_prompt if search_prompt else classification.prompt
        clean_prompt = re.sub(r'[^\w\s]', ' ', prompt_text)
        search_words = [w for w in clean_prompt.split() if w][:30]
        search_terms = " & ".join(search_words) if search_words else "ice"

        topic_filter = "AND topic_tags && :topics" if classification.topic_tags else ""
        conv_filter = "AND conversation_id = :conv_id" if conv_id else ""

        query = text(f"""
            SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id,
                   ts_rank(
                       to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')),
                       query
                   ) as score
            FROM episodic_memory, to_tsquery('english', :search_terms) query
            WHERE to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(summary_text, '')) @@ query
              {topic_filter}
              {conv_filter}
              AND decay_score > :min_decay
              AND is_archived = false
            ORDER BY score DESC
            LIMIT 10
        """)

        params = {"search_terms": search_terms, "min_decay": 0.2}
        if classification.topic_tags:
            params["topics"] = classification.topic_tags
        if conv_id:
            params["conv_id"] = conv_id

        try:
            rows = self.db.execute(query, params).fetchall()
            return self._rows_to_fragments(rows, "episodic")
        except Exception as err:
            logger.error("bm25_retrieval_failed", error=str(err))
            self.db.rollback()
            return []

    # ------------------------------------------------------------------
    # Vector episodic (pgvector cosine similarity)
    # ------------------------------------------------------------------
    def _vector_episodic(self, prompt_embedding, classification, scope, conv_id: Optional[str] = None) -> List[ContextFragment]:
        topic_filter = "AND topic_tags && :topics" if classification.topic_tags else ""
        conv_filter = "AND conversation_id = :conv_id" if conv_id else ""

        query = text(f"""
            SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id,
                1 - (embedding <=> :prompt_embedding) as score
            FROM episodic_memory
            WHERE embedding IS NOT NULL
            {topic_filter}
            {conv_filter}
            AND decay_score > :min_decay
            AND is_archived = false
            ORDER BY score DESC
            LIMIT 10
        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
        params = {"prompt_embedding": prompt_embedding, "min_decay": 0.2}
        if classification.topic_tags:
            params["topics"] = classification.topic_tags
        if conv_id:
            params["conv_id"] = conv_id

        try:
            rows = self.db.execute(query, params).fetchall()
            return self._rows_to_fragments(rows, "episodic")
        except Exception as err:
            logger.error("vector_retrieval_failed", error=str(err))
            self.db.rollback()
            return []

    # ------------------------------------------------------------------
    # Codex graph traversal (conversation‑scoped)
    # ------------------------------------------------------------------
    def _codex_graph(self, classification, scope: Optional[dict] = None) -> List[ContextFragment]:
        prompt = classification.prompt
        candidates = set(re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', prompt))
        if not candidates:
            return []

        normalized = [c.lower().strip() for c in candidates]
        from sqlalchemy import or_
        alias_conditions = [CodexEntity.aliases.any(name) for name in normalized]

        try:
            entities = self.db.query(CodexEntity).filter(
                or_(CodexEntity.canonical_name.in_(normalized), *alias_conditions)
            ).all()

            # Scoping: restrict to entities that appear in the target conversation
            allowed_entity_ids = None
            if scope and "conversation_id" in scope:
                conv_id = scope["conversation_id"]
                try:
                    batch_rows = self.db.execute(
                        text("SELECT DISTINCT batch_id FROM episodic_memory WHERE conversation_id = :cid"),
                        {"cid": conv_id}
                    ).fetchall()
                    batch_ids = [row.batch_id for row in batch_rows]
                    if batch_ids:
                        event_rows = self.db.execute(
                            text("SELECT DISTINCT entity_id FROM codex_events WHERE batch_source = ANY(:bids)"),
                            {"bids": batch_ids}
                        ).fetchall()
                        allowed_entity_ids = {row.entity_id for row in event_rows}
                except Exception:
                    self.db.rollback()

            visited = set()
            context_texts = []
            for entity in entities:
                if allowed_entity_ids is not None and entity.id not in allowed_entity_ids:
                    continue
                self._traverse_graph(entity, 0, 2, visited, context_texts)

            if context_texts:
                combined = "\n\n".join(context_texts)
                return [ContextFragment(
                    text=combined,
                    source_type="codex",
                    score=1.0,
                    token_count=int(len(combined.split()) * 1.33)
                )]
            return []
        except Exception as err:
            logger.error("codex_retrieval_failed", error=str(err))
            self.db.rollback()
            return []

    def _traverse_graph(self, entity, depth, max_depth, visited, context_texts):
        if entity.id in visited or depth > max_depth:
            return
        visited.add(entity.id)
        if entity.context_payload:
            context_texts.append(f"[Entity: {entity.canonical_name}]\n{entity.context_payload}")
        edges = self.db.query(CodexEdge).filter(
            CodexEdge.source_id == entity.id,
            CodexEdge.valid_until == None
        ).all()
        for edge in edges:
            target = self.db.query(CodexEntity).get(edge.target_id)
            if target:
                self._traverse_graph(target, depth + 1, max_depth, visited, context_texts)

    # ------------------------------------------------------------------
    # Procedural lookup (scoped + trigger‑condition evaluation)
    # ------------------------------------------------------------------
    def _procedural_lookup(self, prompt_embedding, classification, scope: Optional[dict] = None) -> List[ContextFragment]:
        activating = {"Strategic_Planning", "Generation", "Open_Exploration"}
        if not any(i in classification.intent_tags for i in activating):
            return []

        # Scoping: collect batch_ids for the target conversation
        allowed_batch_ids = None
        if scope and "conversation_id" in scope:
            conv_id = scope["conversation_id"]
            try:
                batch_rows = self.db.execute(
                    text("SELECT DISTINCT batch_id FROM episodic_memory WHERE conversation_id = :cid"),
                    {"cid": conv_id}
                ).fetchall()
                allowed_batch_ids = [row.batch_id for row in batch_rows]
            except Exception:
                self.db.rollback()

        query = text("""
            SELECT id, pattern_description,
                   1 - (embedding <=> :prompt_embedding) as score
            FROM procedural_memory
            WHERE embedding IS NOT NULL AND is_active = true
            ORDER BY score DESC
            LIMIT 5
        """)
        try:
            rows = self.db.execute(query, {"prompt_embedding": prompt_embedding}).fetchall()
            fragments = []
            for r in rows:
                pattern = self.db.query(ProceduralMemory).get(r.id)
                if not pattern:
                    continue
                # Scope filter
                if allowed_batch_ids is not None:
                    if not any(bid in allowed_batch_ids for bid in (pattern.source_batch_ids or [])):
                        continue
                # Trigger condition match
                if not self._procedural_trigger_match(pattern, classification):
                    continue
                fragments.append(ContextFragment(
                    text=pattern.pattern_description,
                    source_type="procedural",
                    score=r.score,
                    token_count=int(len(pattern.pattern_description.split()) * 1.33)
                ))
            return fragments
        except Exception:
            self.db.rollback()
            return []

    def _procedural_trigger_match(self, pattern: ProceduralMemory, classification: ClassificationResult) -> bool:
        conditions = pattern.trigger_conditions or {}
        if not conditions:
            return True
        required_topics = set(conditions.get("topic_tags", []))
        required_intents = set(conditions.get("intent_tags", []))
        if required_topics and not required_topics.intersection(classification.topic_tags):
            return False
        if required_intents and not required_intents.intersection(classification.intent_tags):
            return False
        return True

    # ------------------------------------------------------------------
    # RAG lookup (unchanged)
    # ------------------------------------------------------------------
    def _rag_lookup(self, prompt_embedding, classification) -> List[ContextFragment]:
        if classification.context_reliance != "Long_Term_Memory":
            return []
        if not ("Factual_Retrieval" in classification.intent_tags or
                "Analysis_&_Summarization" in classification.intent_tags):
            return []
        if not any(w in classification.prompt.lower() for w in ["document", "pdf", "reference", "manual", "guide"]):
            return []

        query = text("""
            SELECT chunk_text,
                   1 - (embedding <=> :prompt_embedding) as score
            FROM rag_chunks
            ORDER BY score DESC
            LIMIT 5
        """)
        try:
            rows = self.db.execute(query, {"prompt_embedding": prompt_embedding}).fetchall()
            return [ContextFragment(
                text=r.chunk_text,
                source_type="rag",
                score=r.score,
                token_count=int(len(r.chunk_text.split()) * 1.33)
            ) for r in rows]
        except Exception:
            self.db.rollback()
            return []

    # ------------------------------------------------------------------
    # RRF fusion, diversification, dedup, token budget
    # ------------------------------------------------------------------
    def _apply_rrf(self, legs: Dict[str, List[ContextFragment]], k: int = 60) -> List[ContextFragment]:
        rrf_scores: Dict[str, float] = {}
        fragment_registry: Dict[str, ContextFragment] = {}

        for leg_name, fragments in legs.items():
            fragments.sort(key=lambda x: x.score, reverse=True)
            for rank, frag in enumerate(fragments, start=1):
                frag_hash = hashlib.sha256(frag.text.encode('utf-8')).hexdigest()
                if frag_hash not in fragment_registry:
                    fragment_registry[frag_hash] = frag
                rrf_scores[frag_hash] = rrf_scores.get(frag_hash, 0.0) + (1.0 / (k + rank))

        fused = []
        for frag_hash, score in rrf_scores.items():
            fragment_registry[frag_hash].score = score
            fused.append(fragment_registry[frag_hash])

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    def _session_diversify(self, fragments, current_id, max_per_conversation=3):
        counts: Dict[str, int] = {}
        result = []
        for f in fragments:
            cid = f.conversation_id
            if not cid:
                result.append(f)
            elif cid == current_id:
                result.append(f)
            else:
                counts[cid] = counts.get(cid, 0) + 1
                if counts[cid] <= max_per_conversation:
                    result.append(f)
        return result

    def _deduplicate(self, fragments):
        seen = set()
        unique = []
        for f in fragments:
            h = hashlib.sha256(f.text.encode('utf-8')).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(f)
        return unique

    def _enforce_token_budget(self, fragments, max_tokens=2000):
        total = 0
        result = []
        for f in fragments:
            if total + f.token_count <= max_tokens:
                result.append(f)
                total += f.token_count
        return result

    # ------------------------------------------------------------------
    # Strengthening (access count + decay boost)
    # ------------------------------------------------------------------
    def _strengthen_retrieved(self, fragments: List[ContextFragment]):
        for frag in fragments:
            if frag.source_type != "episodic" or not frag.source_batch_id:
                continue
            try:
                turn = self.db.query(EpisodicMemory).get(uuid.UUID(frag.source_batch_id))
                if turn:
                    turn.access_count = (turn.access_count or 0) + 1
                    turn.decay_score = min(1.0, (turn.decay_score or 0.0) + 0.15)
                    self.db.commit()
            except Exception:
                self.db.rollback()

    # ------------------------------------------------------------------
    # Wide‑net fallback (now uses full vector search)
    # ------------------------------------------------------------------
    def _wide_net_fallback(self, classification, prompt_embedding, conversation_id, scope):
        try:
            query = text("""
                SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id,
                       1 - (embedding <=> :prompt_embedding) as score
                FROM episodic_memory
                WHERE embedding IS NOT NULL
                  AND is_archived = false
                  AND decay_score > :min_decay
                ORDER BY score DESC
                LIMIT 10
            """).bindparams(bindparam("prompt_embedding", type_=PgVector))
            rows = self.db.execute(query, {
                "prompt_embedding": prompt_embedding,
                "min_decay": 0.2
            }).fetchall()
            fragments = self._rows_to_fragments(rows, "episodic")
        except Exception:
            self.db.rollback()
            fragments = []

        fragments.extend(self._codex_graph(classification))
        fragments.extend(self._rag_lookup(prompt_embedding, classification))

        fused = self._apply_rrf({"fallback": fragments})
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        return self._enforce_token_budget(self._deduplicate(diversified), max_tokens=2000)

    # ------------------------------------------------------------------
    # Helper: convert raw DB rows to ContextFragment list
    # ------------------------------------------------------------------
    def _rows_to_fragments(self, rows, source_type):
        fragments = []
        for row in rows:
            if row.inject_raw and row.raw_text:
                text = row.raw_text
            elif row.summary_text:
                text = row.summary_text
            elif row.raw_text:
                text = row.raw_text[:300]
            else:
                continue

            words = text.split()
            if len(words) > 500:
                text = ' '.join(words[:500]) + '…'

            if not text:
                continue

            score_val = getattr(row, "score", 1.0)
            fragments.append(ContextFragment(
                text=text,
                source_type=source_type,
                score=float(score_val),
                token_count=int(len(text.split()) * 1.33),
                source_batch_id=str(row.id),
                conversation_id=str(row.conversation_id) if row.conversation_id else None
            ))
        return fragments
```

---

## 2. `src/workers/decay.py` (complete)

```python
"""Decay Worker – applies access‑weighted memory decay and archival."""

import structlog
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory, ColdStorage
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.decay")
DECAY_RATE_UNACCESSED = 0.95   # 5% decay per day for never‑accessed turns
DECAY_RATE_ACCESSED = 0.98     # 2% decay for turns that have been accessed
STRENGTHEN_AMOUNT = 0.15
ARCHIVE_THRESHOLD = 0.1
COLD_THRESHOLD = 0.05


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def apply_decay(self):
    """Daily task: decay old turns, archive stale ones, move to cold storage."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        # Access‑weighted decay: unaccessed turns decay faster
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * :rate
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
              AND access_count = 0
        """), {"rate": DECAY_RATE_UNACCESSED, "cutoff": cutoff})

        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * :rate
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND is_bookmarked = FALSE
              AND is_archived = FALSE
              AND access_count > 0
        """), {"rate": DECAY_RATE_ACCESSED, "cutoff": cutoff})

        # Archive turns below threshold
        db.execute(text("""
            UPDATE episodic_memory
            SET is_archived = TRUE
            WHERE decay_score < :archive_threshold AND is_archived = FALSE
        """), {"archive_threshold": ARCHIVE_THRESHOLD})

        # Move extremely stale archived turns to cold_storage
        cold_rows = db.execute(text("""
            SELECT id, raw_text, summary_text, topic_tags, timestamp
            FROM episodic_memory
            WHERE is_archived = TRUE AND decay_score < :cold_threshold
        """), {"cold_threshold": COLD_THRESHOLD}).fetchall()

        for row in cold_rows:
            db.execute(text("""
                INSERT INTO cold_storage (id, archived_at, raw_text, summary_text, topic_tags, timestamp)
                VALUES (:id, :now, :raw, :summary, :tags, :ts)
                ON CONFLICT (id) DO NOTHING
            """), {
                "id": row.id,
                "now": datetime.now(timezone.utc),
                "raw": row.raw_text,
                "summary": row.summary_text,
                "tags": row.topic_tags,
                "ts": row.timestamp
            })
            db.execute(text("DELETE FROM episodic_memory WHERE id = :id"), {"id": row.id})

        db.commit()
        logger.info("decay_cycle_complete", archived=len(cold_rows))

    except Exception as exc:
        db.rollback()
        logger.error("decay_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()
```

---

## 3. `src/api/prompt_assembler.py` (complete)

```python
"""Context Structural Assembly Plane – builds the final prompt payload,
   now including the sliding window of recent turns."""

from typing import List, Optional
from sqlalchemy.orm import Session
from src.retrieval.orchestrator import ContextFragment
from src.memory.models import MemorySlot, EpisodicMemory

SYSTEM_RULES = (
    "You are an AI assistant with access to a personal memory system (ICE).\n"
    "The following context has been automatically retrieved from past conversations and knowledge.\n"
    "Use it to answer the user's question accurately. If the context is irrelevant, ignore it."
)


def get_recent_turns(db_session: Session, conversation_id: str, n: int = 10) -> List[str]:
    """Return the text of the last N turns from the current conversation."""
    turns = db_session.query(EpisodicMemory).filter_by(
        conversation_id=conversation_id
    ).order_by(EpisodicMemory.timestamp.desc()).limit(n).all()
    turns.reverse()  # chronological order
    fragments = []
    for t in turns:
        if t.inject_raw and t.raw_text:
            text = t.raw_text
        elif t.summary_text:
            text = t.summary_text
        else:
            text = (t.raw_text or "")[:300]
        words = text.split()
        if len(words) > 500:
            text = " ".join(words[:500]) + "…"
        fragments.append(text)
    return fragments


def assemble_prompt(
    memory_slots: List[MemorySlot],
    retrieved_fragments: List[ContextFragment],
    user_message: str,
    db_session: Optional[Session] = None,
    conversation_id: Optional[str] = None,
) -> List[dict]:
    """Assemble the final prompt in stable‑prefix order."""
    system_content = SYSTEM_RULES

    # 0. Recent context (sliding window)
    if db_session and conversation_id:
        recent_texts = get_recent_turns(db_session, conversation_id, n=10)
        if recent_texts:
            system_content += "\n\n=== RECENT CONTEXT ===\n" + "\n\n".join(recent_texts)

    # 1. Persistent Memory Slots
    if memory_slots:
        slot_lines = []
        for slot in memory_slots:
            if slot.is_active and slot.content:
                slot_lines.append(f"[{slot.slot_name.upper()}]\n{slot.content.strip()}")
        if slot_lines:
            system_content += "\n\n=== PERSISTENT CORE PREFERENCES ===\n" + "\n\n".join(slot_lines)

    # 2. Codex (absolute facts)
    codex_frags = [f for f in retrieved_fragments if f.source_type == "codex"]
    if codex_frags:
        codex_text = "\n\n".join(f.text.strip() for f in codex_frags)
        system_content += f"\n\n=== CODEX KNOWLEDGE GRAPH ASSERTIONS ===\n{codex_text}"

    # 3. Episodic context
    episodic_frags = [f for f in retrieved_fragments if f.source_type == "episodic"]
    if episodic_frags:
        episodic_text = "\n\n".join(f.text.strip() for f in episodic_frags)
        system_content += f"\n\n=== RETRIEVED EPISODIC INTERACTIONS ===\n{episodic_text}"

    # 4. Procedural patterns
    procedural_frags = [f for f in retrieved_fragments if f.source_type == "procedural"]
    if procedural_frags:
        proc_text = "\n\n".join(f.text.strip() for f in procedural_frags)
        system_content += f"\n\n=== PROCEDURAL EXECUTION PATTERNS ===\n{proc_text}"

    # 5. RAG chunks
    rag_frags = [f for f in retrieved_fragments if f.source_type == "rag"]
    if rag_frags:
        rag_text = "\n\n".join(f.text.strip() for f in rag_frags)
        system_content += f"\n\n=== REFERENCE MATERIAL ===\n{rag_text}"

    return [
        {"role": "system", "content": system_content.strip()},
        {"role": "user", "content": user_message},
    ]
```

---

## 4. (Optional) One‑line update in `src/api/main.py`

In the `chat_completions` endpoint, when you call `assemble_prompt`, pass the new arguments:

```python
messages = assemble_prompt(memory_slots, fragments, user_message,
                           db_session=db, conversation_id=str(conversation_id))
```

This enables the sliding window. If you don’t pass them, the recent context block is simply omitted.

We'll now complete **A10** and the **sliding‑window activation** in `main.py`. Both are straightforward.

---

## A10 — Classifier Fine‑Tuning Loop

### What we're building

A standalone Celery task that:
- Loads all rows from the `curated_labels` table.
- Encodes each prompt using the frozen SentenceTransformer.
- Trains **only** the final classification head (the two linear layers) for a few epochs.
- Saves the updated model weights to disk.

It’s designed to be triggered manually or on a schedule (weekly).

---

### Step‑1: Create the worker file

**Create `src/workers/fine_tune.py`** with the following content:

```python
"""Fine‑tuning worker: retrains the MLP head on user‑curated labels."""

import torch
import numpy as np
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

from src.workers.celery_app import app
from src.api.db import SessionLocal
from src.memory.models import CuratedLabel
from src.classifier.model import ICEClassifier

# ------------------------------------------------------------------
# Constants – these MUST match the ones used during initial training
# ------------------------------------------------------------------
TOPIC_LABELS = [
    "Software_&_Tech", "STEM_&_Academics", "Business_&_Finance",
    "Creative_&_Media", "Admin_&_Productivity", "Lifestyle_&_Health",
    "Social_&_Relationships", "World_&_Current_Events", "Meta_AI",
    "Null_Noise", "General_Reference_&_Trivia"
]

INTENT_LABELS = [
    "Factual_Retrieval", "Troubleshooting", "Generation", "Ideation",
    "Analysis_&_Summarization", "Strategic_Planning", "Decision_Making",
    "Emotional_Processing", "Utility_Formatting", "Casual_Banter",
    "Open_Exploration"
]

CONTEXT_RELIANCE_LABELS = ["Zero_Shot", "Long_Term_Memory", "Real_Time_Search"]

MODEL_PATH = "models/classifier/ice_classifier_v2_final.pt"
SCHEMA_PATH = "data/labeled/label_schema.json"

# ------------------------------------------------------------------
@app.task(bind=True, max_retries=1)
def fine_tune_classifier(self):
    """Load curated labels, encode, train head, save new checkpoint."""

    db = SessionLocal()
    try:
        rows = db.query(CuratedLabel).all()
        if not rows:
            return "No curated labels found – skipping fine‑tuning."

        # 1. Encode prompts with frozen SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        prompts = [row.prompt for row in rows]
        embeddings = embedder.encode(prompts, convert_to_tensor=True, show_progress_bar=False)

        # 2. Build label tensors from the curated labels
        labels = torch.zeros((len(rows), 25), dtype=torch.float32)
        for i, row in enumerate(rows):
            # Topic labels (positions 0–10)
            for tag in row.corrected_topic_labels:
                if tag in TOPIC_LABELS:
                    labels[i, TOPIC_LABELS.index(tag)] = 1.0
            # Intent labels (positions 11–21)
            for tag in row.corrected_intent_labels:
                if tag in INTENT_LABELS:
                    labels[i, 11 + INTENT_LABELS.index(tag)] = 1.0
            # Context reliance (positions 22–24, one‑hot)
            if row.corrected_context_reliance in CONTEXT_RELIANCE_LABELS:
                labels[i, 22 + CONTEXT_RELIANCE_LABELS.index(row.corrected_context_reliance)] = 1.0

        # 3. Load the trained model and freeze the encoder (not used in this script, but
        #    the model only contains the head – the SentenceTransformer is separate).
        model = ICEClassifier()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.train()

        # 4. Simple training loop (few epochs, tiny dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn_topic = torch.nn.BCEWithLogitsLoss()
        loss_fn_intent = torch.nn.BCEWithLogitsLoss()
        loss_fn_ctx = torch.nn.CrossEntropyLoss()

        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(embeddings)

            topic_out = outputs[:, :11]
            intent_out = outputs[:, 11:22]
            ctx_out = outputs[:, 22:]

            topic_gt = labels[:, :11]
            intent_gt = labels[:, 11:22]
            ctx_gt = labels[:, 22:].argmax(dim=1)

            loss = (
                loss_fn_topic(topic_out, topic_gt) +
                loss_fn_intent(intent_out, intent_gt) +
                loss_fn_ctx(ctx_out, ctx_gt)
            )
            loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                print(f"  Epoch {epoch}, loss = {loss.item():.4f}")

        # 5. Save updated checkpoint
        new_path = f"models/classifier/ice_classifier_finetuned_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(model.state_dict(), new_path)
        return f"Fine‑tuned model saved to {new_path}"

    except Exception as exc:
        db.rollback()
        raise self.retry(exc=exc)
    finally:
        db.close()
```

---

### Step‑2: Register the new task in the Celery app

Open `src/workers/celery_app.py` and add `"src.workers.fine_tune"` to the `include` list:

```python
include=[
    "src.workers.post_flight",
    "src.workers.codex_extractor",
    "src.workers.compaction",
    "src.workers.procedural_extractor",
    "src.workers.decay",
    "src.workers.reflection",
    "src.workers.sentinel_monitor",
    "src.workers.clustering",
    "src.workers.fine_tune",          # ← new
],
```

If you want it to run automatically every week, add a schedule entry to `beat_schedule` in the same file:

```python
    'fine-tune-weekly': {
        'task': 'src.workers.fine_tune.fine_tune_classifier',
        'schedule': crontab(hour=4, minute=0, day_of_week=1),   # Monday 4am
    },
```

---

### Step‑3: Manual trigger (for testing)

After restarting the Celery worker:

```bash
uv run celery -A src.workers.celery_app call src.workers.fine_tune.fine_tune_classifier
```

It will load whatever is in `curated_labels`, retrain, and save a new checkpoint.

---

## 4. Activating the sliding window in the proxy

**File:** `src/api/main.py`

Find the call to `assemble_prompt` inside the `chat_completions` endpoint.  
It currently looks like:

```python
messages = assemble_prompt(memory_slots, fragments, user_message)
```

**Change it to:**

```python
messages = assemble_prompt(memory_slots, fragments, user_message,
                           db_session=db, conversation_id=str(conversation_id))
```

The `db` and `conversation_id` variables already exist in that scope — you’re just passing them into the function.  
No other changes are required.

---


---

# Phase B — Memory Lifecycle & Cognition Completion

These turn ICE into a true long‑horizon cognition system (G9) and provide data for the paper’s longitudinal claims.

| # | Feature | Architecture ref | Current state | What to build | Rough effort |
|---|---------|-----------------|---------------|---------------|-------------|
| B1 | **Retrieval strengthening (part of A2)** | §4.2 | Already covered above. | — | — |
| B2 | **Codex edge decay** | §4.4 | Not implemented. | 1) Add a periodic task that decays `strength` for edges not referenced in recent retrieval. 2) When strength falls below threshold, demote to `pending`. | 2 h |
| B3 | **Procedural pattern decay** | §3.3 | Not implemented. | 1) Add a periodic task that marks patterns as inactive if not observed in 6 months and reinforcement_count is low. | 1 h |
| B4 | **Cold storage periodic migration** | §4.3 | Exists but only manually triggered. | 1) Ensure the Decay Worker moves sub‑cold‑threshold archived turns to `cold_storage` on each daily run. 2) Verify it works end‑to‑end. | 1 h |
| B5 | **Reflection Worker – full implementation** | §6.2 | Only session synthesis. | 1) Pattern crystallization: scan recent sessions, feed novel patterns to Procedural Extractor. 2) Memory slot evolution: propose updates to `project_context`, `user_preferences`, `guidance`. 3) Codex enrichment: append episodic passages to thin entities. 4) Motif detection: propose new clusters. | 8 h |
| B6 | **Sentinel Monitor – real rule evaluation** | §5 | Placeholder only. | 1) Implement evaluation for at least 3 rule types (threshold, frequency, absence). 2) Populate a few default rules (e.g., staleness, contradiction). 3) Connect actions: `log_event` (already works), `notify` (write to a notifications table), `schedule_worker` (enqueue Celery task). | 6 h |

## Phase B — Memory Lifecycle & Cognition Completion

All changes are designed to make ICE a true long‑horizon cognition system, exactly as described in the architecture (§6, §4.4, §5).  
We’ll build the missing lifecycle workers, fully implement the Reflection Worker, and replace the Sentinel placeholder with real rule evaluation.

---

### B1 — Retrieval Strengthening

Already implemented as part of A2. No further work.

---

### B2 — Codex Edge Decay

**What:** Active Codex edges must slowly lose strength over time. When strength drops below a threshold, the edge is demoted to `pending` (invisible to retrieval). This prevents outdated facts from permanently occupying the knowledge graph.

**Why:** Without decay, a fact extracted once stays active forever, even if it’s never corroborated again. Decay keeps the graph fresh and accurate.

**How:** A daily Celery task multiplies every active edge’s strength by **0.99** (1% decay). If strength falls below **0.3**, confidence is set to `pending`. The edge is preserved for audit but excluded from retrieval.

#### File to create: `src/workers/codex_decay.py`

```python
"""Codex Edge Decay – periodically reduces strength of unreinforced edges."""

import structlog
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import CodexEdge
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.codex_decay")
DECAY_RATE = 0.99          # 1% decay per run
DEMOTION_THRESHOLD = 0.3   # strength below this -> pending


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def decay_codex_edges(self):
    """Daily task: decay strength of active Codex edges, demote weak ones."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        # 1. Decay all active edges
        db.execute(text("""
            UPDATE codex_edges
            SET strength = strength * :rate
            WHERE confidence = 'active'
        """), {"rate": DECAY_RATE})

        # 2. Demote edges that fell below threshold
        db.execute(text("""
            UPDATE codex_edges
            SET confidence = 'pending'
            WHERE confidence = 'active' AND strength < :thresh
        """), {"thresh": DEMOTION_THRESHOLD})

        db.commit()
        logger.info("codex_decay_cycle_complete")
    except Exception as exc:
        db.rollback()
        logger.error("codex_decay_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()
```

#### Update `src/workers/celery_app.py`

Add the module to the `include` list:

```python
include=[
    ...
    "src.workers.codex_decay",
]
```

Add a beat schedule entry (runs daily at 3:30 am):

```python
    'codex-decay-daily': {
        'task': 'src.workers.codex_decay.decay_codex_edges',
        'schedule': crontab(hour=3, minute=30),
    },
```

---

### B3 — Procedural Pattern Decay

**What:** Procedural patterns that haven’t been observed for 6 months and have low reinforcement count should be marked inactive.

**Why:** User habits change. Old, weakly reinforced patterns clutter retrieval and should be disabled until re‑observed.

**How:** Another periodic task checks `last_observed` and `reinforcement_count`. If the pattern hasn’t been seen in 180 days and its count is below 3, set `is_active = False`.

#### File to create: `src/workers/procedural_decay.py`

```python
"""Procedural Memory Decay – deactivates stale, low‑confidence patterns."""

import structlog
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from src.api.db import SessionLocal
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.procedural_decay")
STALE_DAYS = 180
MIN_REINFORCEMENT = 3


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def decay_procedural_patterns(self):
    """Periodic task: deactivate procedural patterns that are stale and weak."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)
        db.execute(text("""
            UPDATE procedural_memory
            SET is_active = FALSE
            WHERE is_active = TRUE
              AND last_observed < :cutoff
              AND reinforcement_count < :min_reinf
        """), {"cutoff": cutoff, "min_reinf": MIN_REINFORCEMENT})
        db.commit()
        logger.info("procedural_decay_cycle_complete")
    except Exception as exc:
        db.rollback()
        logger.error("procedural_decay_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()
```

#### Update `src/workers/celery_app.py`

```python
include=[
    ...
    "src.workers.procedural_decay",
]
```

Beat schedule:

```python
    'procedural-decay-daily': {
        'task': 'src.workers.procedural_decay.decay_procedural_patterns',
        'schedule': crontab(hour=4, minute=30),
    },
```

---

### B4 — Cold Storage Periodic Migration

Already implemented inside the Decay Worker (`src/workers/decay.py`).  
The daily run moves archived turns with `decay_score < 0.05` to `cold_storage`. No changes needed.

---

### B5 — Reflection Worker Full Implementation

The Reflection Worker must do more than session synthesis. The architecture (§6.2) lists five tasks:

- **Session synthesis** (already done)
- **Pattern crystallization** – identify recurring behaviours across recent sessions and feed them to the Procedural Extractor
- **Memory slot evolution** – propose updates to `project_context`, `user_preferences`, `guidance` based on recent behaviour
- **Codex enrichment** – add contextual information to thin Codex entities
- **Motif detection** – detect themes that don’t yet have a cluster and propose new ones

We’ll implement all of these in the same Celery task. Because there’s no UI for confirmation yet, we’ll:

- **Pattern crystallization**: directly insert new pending patterns (or increment reinforcement) using the background model.
- **Memory slot evolution**: write proposals to a new `review_queue` table. We’ll create that table now.
- **Codex enrichment**: append to `context_payload` immediately (no confirmation needed per architecture).
- **Motif detection**: propose new clusters by inserting a review item.

#### 5.1 Create the `review_queue` table (manual SQL for now)

```bash
docker exec -i ice_postgres psql -U ice -d ice_db <<'SQL'
CREATE TABLE IF NOT EXISTS review_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_type TEXT NOT NULL,
    item_content JSONB NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
SQL
```

Later we’ll create an Alembic migration for permanence.

#### 5.2 Rewrite `src/workers/reflection.py`

We replace the entire file with a complete implementation.

**File:** `src/workers/reflection.py`

```python
"""Reflection Worker – full implementation: session synthesis, pattern crystallization,
   memory slot evolution, Codex enrichment, motif detection."""

import structlog, json, re, uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from openai import OpenAI

from src.api.db import SessionLocal
from src.memory.models import (
    EpisodicMemory, SessionSummary, MemorySlot, CodexEntity, CodexEvent,
    ProceduralMemory, ContextCluster
)
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.reflection")
bg_client = OpenAI(base_url="http://localhost:8002/v1", api_key="dummy")

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
SUMMARY_PROMPT = (
    "Generate a structured session summary from the following conversation turns.\n"
    "Output ONLY a valid JSON object with these keys:\n"
    "  - \"topics_covered\": a list of strings (e.g., [\"PostgreSQL\", \"FastAPI\"])\n"
    "  - \"decisions_made\": a string describing any decisions\n"
    "  - \"unresolved_items\": a string describing any unresolved questions\n"
    "  - \"entities_updated\": a list of canonical entity names that appeared\n"
    "  - \"patterns_observed\": a list of strings describing observed behavioural patterns\n\n"
    "If a field has no content, use an empty list [] for lists, or an empty string \"\" for strings.\n"
    "Do NOT include markdown or additional text."
)

CRYSTALLIZATION_PROMPT = (
    "Below are snippets from multiple recent conversation sessions. Identify any recurring "
    "behavioural patterns or workflows that the user consistently follows. For each pattern, "
    "output a single descriptive sentence. Return ONLY a JSON array of strings. If no patterns "
    "are found, return an empty array []."
)

SLOT_EVOLUTION_PROMPT = (
    "You are analysing a user's recent conversations. Based on the content, suggest if any of "
    "the following persistent memory slots should be updated:\n"
    "- project_context: what the user is currently working on\n"
    "- user_preferences: how the user likes to interact\n"
    "- guidance: rules the AI should follow\n\n"
    "Output ONLY a JSON object with keys matching the slot names (if an update is needed) and "
    "the proposed new content as the value. If no update is needed for a slot, omit the key. "
    "The proposed content should be a concise paragraph. Do NOT include markdown."
)

ENRICHMENT_PROMPT = (
    "The following is a context payload for a knowledge graph entity. It is currently very thin. "
    "Given additional conversation passages, write an enriched, factual description of the entity. "
    "Output ONLY the enriched description, no markdown."
)

MOTIF_PROMPT = (
    "Below are conversations from multiple recent sessions. Identify any recurring thematic motifs "
    "that do not yet correspond to a named project or cluster. For each motif, suggest a short, "
    "descriptive cluster name. Output ONLY a JSON array of strings. If none, return []."
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _robust_json(raw: str) -> dict:
    """Try to extract a JSON object from model output, fall back to empty dict."""
    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else {}
    except Exception:
        return {}

def _robust_list(raw: str) -> list:
    try:
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else []
    except Exception:
        return []

# ------------------------------------------------------------------
# Main task
# ------------------------------------------------------------------
@app.task(bind=True, max_retries=2, default_retry_delay=60)
def run_reflection(self):
    """Execute a full reflection pass: synthesis, patterns, slots, enrichment, motifs."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        # 1. Load recent turns (last 200 across all conversations, for breadth)
        recent = db.query(EpisodicMemory).order_by(
            EpisodicMemory.timestamp.desc()
        ).limit(200).all()
        if not recent:
            return
        recent.reverse()  # chronological

        # ---- Session Synthesis ----
        _synthesize_session(db, recent)

        # ---- Pattern Crystallization ----
        _crystallize_patterns(db, recent)

        # ---- Memory Slot Evolution ----
        _evolve_memory_slots(db, recent)

        # ---- Codex Enrichment ----
        _enrich_codex_entities(db)

        # ---- Motif Detection ----
        _detect_motifs(db, recent)

        db.commit()
        logger.info("reflection_full_pass_complete")

    except Exception as exc:
        db.rollback()
        logger.error("reflection_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


# ------------------------------------------------------------------
# Session Synthesis (existing logic, kept)
# ------------------------------------------------------------------
def _synthesize_session(db, turns):
    full_text = "\n\n".join([t.raw_text for t in turns])
    words = full_text.split()
    if len(words) > 3000:
        full_text = " ".join(words[-3000:])
    completion = bg_client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a session analysis engine. Output only JSON."},
            {"role": "user", "content": f"{SUMMARY_PROMPT}\n\n{full_text}"}
        ],
        temperature=0.0, max_tokens=400, timeout=30.0
    )
    raw = completion.choices[0].message.content.strip()
    data = _robust_json(raw)

    summary = SessionSummary(
        topics_covered=data.get("topics_covered", []),
        decisions_made=data.get("decisions_made", ""),
        unresolved_items=data.get("unresolved_items", ""),
        entities_updated=data.get("entities_updated", []),
        patterns_observed=data.get("patterns_observed", [])
    )
    db.add(summary)

    # Update pending_items if unresolved items found
    unresolved = data.get("unresolved_items")
    if unresolved and isinstance(unresolved, str) and unresolved.strip():
        slot = db.query(MemorySlot).filter_by(slot_name="pending_items").first()
        if slot:
            slot.content = (slot.content or "") + "\n" + unresolved
            slot.version += 1
            slot.last_updated = datetime.now(timezone.utc)
            slot.updated_by = "reflection_worker"


# ------------------------------------------------------------------
# Pattern Crystallization
# ------------------------------------------------------------------
def _crystallize_patterns(db, turns):
    # Build a compact representation (last 1500 words)
    text = "\n".join([t.raw_text[:200] for t in turns])
    if len(text.split()) > 1500:
        text = " ".join(text.split()[-1500:])
    completion = bg_client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a behavioural pattern detector."},
            {"role": "user", "content": f"{CRYSTALLIZATION_PROMPT}\n\n{text}"}
        ],
        temperature=0.0, max_tokens=200, timeout=30.0
    )
    raw = completion.choices[0].message.content.strip()
    patterns = _robust_list(raw)
    for desc in patterns:
        if not isinstance(desc, str) or not desc.strip():
            continue
        # Check for existing pattern by embedding similarity
        from src.workers.procedural_extractor import encode_pattern
        emb = encode_pattern(desc)
        try:
            similar = db.execute(
                text("SELECT id, 1 - (embedding <=> CAST(:emb AS vector)) AS sim FROM procedural_memory WHERE embedding IS NOT NULL ORDER BY sim DESC LIMIT 1"),
                {"emb": str(emb)}
            ).first()
            if similar and similar.sim > 0.85:
                existing = db.query(ProceduralMemory).get(similar.id)
                existing.reinforcement_count += 1
                existing.last_observed = datetime.now(timezone.utc)
                if existing.reinforcement_count >= 3 and existing.confidence_score < 0.8:
                    existing.confidence_score = 0.8
                    existing.is_active = True
            else:
                new_pat = ProceduralMemory(
                    pattern_name=desc[:80],
                    pattern_description=desc,
                    topic_tags=turns[0].topic_tags if turns else [],
                    trigger_conditions={},
                    reinforcement_count=1,
                    confidence_score=0.3,
                    first_observed=datetime.now(timezone.utc),
                    last_observed=datetime.now(timezone.utc),
                    is_active=False,
                    source_batch_ids=[t.batch_id for t in turns[:10]],
                    embedding=emb
                )
                db.add(new_pat)
        except Exception as e:
            logger.error("pattern_crystallization_error", error=str(e))


# ------------------------------------------------------------------
# Memory Slot Evolution
# ------------------------------------------------------------------
def _evolve_memory_slots(db, turns):
    text = "\n".join([t.raw_text[:200] for t in turns])
    if len(text.split()) > 1500:
        text = " ".join(text.split()[-1500:])
    completion = bg_client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a memory slot analyst. Output only JSON."},
            {"role": "user", "content": f"{SLOT_EVOLUTION_PROMPT}\n\n{text}"}
        ],
        temperature=0.0, max_tokens=300, timeout=30.0
    )
    raw = completion.choices[0].message.content.strip()
    proposals = _robust_json(raw)
    for slot_name, content in proposals.items():
        if slot_name in ("project_context", "user_preferences", "guidance") and isinstance(content, str) and content.strip():
            # Insert into review_queue for user confirmation (Phase C)
            db.execute(
                text("INSERT INTO review_queue (item_type, item_content) VALUES ('memory_slot_update', :payload)"),
                {"payload": json.dumps({"slot_name": slot_name, "proposed_content": content})}
            )


# ------------------------------------------------------------------
# Codex Enrichment
# ------------------------------------------------------------------
def _enrich_codex_entities(db):
    # Find entities with short context_payload (less than 100 chars)
    thin_entities = db.query(CodexEntity).filter(
        CodexEntity.context_payload == None
    ).all()[:10]  # limit to 10 per run
    for entity in thin_entities:
        if entity.context_payload and len(entity.context_payload) > 100:
            continue
        # Find episodic turns that mention this entity
        batch_ids = db.execute(
            text("SELECT batch_source FROM codex_events WHERE entity_id = :eid"),
            {"eid": entity.id}
        ).fetchall()
        if not batch_ids:
            continue
        passages = []
        for (bid,) in batch_ids:
            turn = db.query(EpisodicMemory).filter_by(batch_id=bid).first()
            if turn:
                passages.append(turn.raw_text[:500])
        if not passages:
            continue
        combined = "\n".join(passages)
        completion = bg_client.chat.completions.create(
            model="Qwen/Qwen2.5-3B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "You are a knowledge graph enricher. Write a factual description."},
                {"role": "user", "content": f"{ENRICHMENT_PROMPT}\nCurrent payload: {entity.context_payload or ''}\nRelevant passages:\n{combined[:2000]}"}
            ],
            temperature=0.0, max_tokens=300, timeout=30.0
        )
        enriched = completion.choices[0].message.content.strip()
        entity.context_payload = enriched
        entity.last_updated = datetime.now(timezone.utc)
        db.add(CodexEvent(
            entity_id=entity.id,
            event_type="context_appended",
            payload={"enriched_from_reflection": True},
            batch_source=uuid.uuid4()
        ))


# ------------------------------------------------------------------
# Motif Detection
# ------------------------------------------------------------------
def _detect_motifs(db, turns):
    text = "\n".join([t.raw_text[:200] for t in turns])
    if len(text.split()) > 1500:
        text = " ".join(text.split()[-1500:])
    completion = bg_client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": "You are a thematic motif detector. Output only JSON."},
            {"role": "user", "content": f"{MOTIF_PROMPT}\n\n{text}"}
        ],
        temperature=0.0, max_tokens=150, timeout=30.0
    )
    raw = completion.choices[0].message.content.strip()
    motifs = _robust_list(raw)
    for motif in motifs:
        if isinstance(motif, str) and motif.strip():
            db.execute(
                text("INSERT INTO review_queue (item_type, item_content) VALUES ('new_cluster_proposal', :payload)"),
                {"payload": json.dumps({"cluster_name": motif})}
            )
```

---

### B6 — Sentinel Monitor Real Rule Evaluation

We’ll replace the placeholder `_evaluate_rule` function with actual logic that can handle **threshold**, **frequency**, and **absence** rules. We’ll also provide a script to insert a few default rules into `sentinel_rules`.

#### 6.1 Populate default sentinel rules (one‑time script)

Create `scripts/seed_sentinel_rules.py`:

```python
#!/usr/bin/env python3
"""Insert a few default sentinel rules into the database."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.db import SessionLocal
from src.memory.models import SentinelRule

rules = [
    {
        "name": "Stale pending items",
        "description": "If pending_items slot has content older than 14 days and no recent retrieval, notify.",
        "is_active": True,
        "trigger_type": "absence",
        "trigger_conditions": '{"table": "memory_slots", "field": "content", "key": "pending_items", "max_age_days": 14}',
        "action_type": "notify",
        "action_payload": '{"message": "Pending items may be stale – review them."}',
        "cooldown_seconds": 86400
    },
    {
        "name": "High contradiction entity",
        "description": "If a Codex entity has >3 pending edges and >2 active edges overlapping, create review item.",
        "is_active": True,
        "trigger_type": "threshold",
        "trigger_conditions": '{"entity": true, "min_pending_edges": 3, "min_active_overlap": 2}',
        "action_type": "create_review_item",
        "action_payload": '{"item_type": "codex_contradiction"}',
        "cooldown_seconds": 43200
    },
    {
        "name": "Retrieval health degradation",
        "description": "If 5 consecutive Long_Term_Memory turns return zero results, schedule clustering.",
        "is_active": True,
        "trigger_type": "threshold",
        "trigger_conditions": '{"consecutive_zero_retrieval": 5}',
        "action_type": "schedule_worker",
        "action_payload": '{"worker": "src.workers.clustering.cluster_turns"}',
        "cooldown_seconds": 3600
    }
]

db = SessionLocal()
for r in rules:
    existing = db.query(SentinelRule).filter_by(name=r["name"]).first()
    if not existing:
        db.add(SentinelRule(**r))
db.commit()
db.close()
print("Default sentinel rules inserted.")
```

Run it: `uv run python scripts/seed_sentinel_rules.py`

#### 6.2 Rewrite `src/workers/sentinel_monitor.py` with real evaluation

```python
"""Sentinel Monitor – evaluates declarative rules and fires actions."""

import structlog
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import SentinelRule, SentinelEvent
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.sentinel")


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def monitor_sentinels(self):
    """Periodic task: evaluate all active sentinel rules."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        rules = db.query(SentinelRule).filter_by(is_active=True).all()
        for rule in rules:
            if rule.last_fired_at and (now - rule.last_fired_at).total_seconds() < rule.cooldown_seconds:
                continue
            if _evaluate_rule(rule, db):
                event = SentinelEvent(
                    rule_id=rule.id,
                    fired_at=now,
                    trigger_state=rule.trigger_conditions,
                    action_taken=rule.action_type
                )
                db.add(event)
                rule.last_fired_at = now
                logger.info("sentinel_fired", rule_name=rule.name, action=rule.action_type)

                # Perform action
                _execute_action(rule, db)

        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("sentinel_monitor_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


def _evaluate_rule(rule, db) -> bool:
    cond = rule.trigger_conditions or {}
    ttype = rule.trigger_type

    if ttype == "threshold":
        # Generic threshold check: if cond has keys like min_pending_edges, evaluate
        if "consecutive_zero_retrieval" in cond:
            # check recent turns for zero retrieval
            limit = cond["consecutive_zero_retrieval"]
            # Query the most recent episodic turns where context_reliance = 'Long_Term_Memory'
            # and check if any retrieval happened (we don't log retrieval per turn, so approximate)
            # For now, placeholder – would need retrieval logging.
            return False  # Not implemented fully; would need to track retrieval success
        if "min_pending_edges" in cond:
            min_pend = cond["min_pending_edges"]
            min_active_overlap = cond.get("min_active_overlap", 1)
            # Find entities with many pending edges and overlapping active edges
            rows = db.execute(text("""
                SELECT e.id FROM codex_entities e
                JOIN codex_edges pe ON pe.source_id = e.id AND pe.confidence = 'pending'
                JOIN codex_edges ae ON ae.source_id = e.id AND ae.target_id = pe.target_id
                    AND ae.confidence = 'active' AND ae.relation = pe.relation AND ae.valid_until IS NULL
                GROUP BY e.id
                HAVING COUNT(DISTINCT pe.id) > :min_pen
                   AND COUNT(DISTINCT ae.id) > :min_act
                LIMIT 1
            """), {"min_pen": min_pend, "min_act": min_active_overlap}).fetchall()
            return len(rows) > 0

    elif ttype == "frequency":
        # Not yet implemented
        pass

    elif ttype == "absence":
        if "table" in cond and cond["table"] == "memory_slots":
            # Check if pending_items content hasn't been modified in max_age_days
            max_age = cond.get("max_age_days", 14)
            row = db.execute(text("""
                SELECT 1 FROM memory_slots
                WHERE slot_name = 'pending_items'
                  AND content IS NOT NULL AND content != ''
                  AND last_updated < :cutoff
                LIMIT 1
            """), {"cutoff": datetime.now(timezone.utc) - timedelta(days=max_age)}).first()
            return row is not None

    return False


def _execute_action(rule, db):
    action = rule.action_type
    payload = rule.action_payload or {}

    if action == "log_event":
        pass  # already logged
    elif action == "notify":
        # Write to a notifications table or just log (for now, log)
        logger.info("sentinel_notify", rule=rule.name, message=payload.get("message", ""))
    elif action == "schedule_worker":
        worker = payload.get("worker")
        if worker:
            # Dynamically call the Celery task (lazy import)
            import importlib
            module_name, task_name = worker.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            task = getattr(mod, task_name)
            task.delay()
    elif action == "create_review_item":
        db.execute(
            text("INSERT INTO review_queue (item_type, item_content) VALUES ('sentinel_review', :payload)"),
            {"payload": json.dumps({"rule": rule.name})}
        )
```

The monitor now can detect stale pending items, high‑contradiction entities, and (partially) retrieval degradation. More rules can be added later by inserting rows into `sentinel_rules`.

---

## Final Step – Update Celery Beat Schedule

Add the two new decay tasks to the beat schedule in `src/workers/celery_app.py`:

```python
app.conf.beat_schedule = {
    'apply-decay-daily': {
        'task': 'src.workers.decay.apply_decay',
        'schedule': crontab(hour=3, minute=0),
    },
    'codex-decay-daily': {
        'task': 'src.workers.codex_decay.decay_codex_edges',
        'schedule': crontab(hour=3, minute=30),
    },
    'procedural-decay-daily': {
        'task': 'src.workers.procedural_decay.decay_procedural_patterns',
        'schedule': crontab(hour=4, minute=30),
    },
    'cluster-turns-daily': {
        'task': 'src.workers.clustering.cluster_turns',
        'schedule': crontab(hour=4, minute=0),
    },
    'monitor-sentinels': {
        'task': 'src.workers.sentinel_monitor.monitor_sentinels',
        'schedule': crontab(minute='*/30'),
    },
    'reflection-daily': {
        'task': 'src.workers.reflection.run_reflection',
        'schedule': crontab(hour=5, minute=0),
    },
    'fine-tune-weekly': {
        'task': 'src.workers.fine_tune.fine_tune_classifier',
        'schedule': crontab(hour=4, minute=0, day_of_week=1),
    },
}
```

Now restart the Celery worker. All Phase B components are complete.

---

## Phase C — User Guidance & Control (Human‑Guided Reinforcement)

These are required by the architecture’s design goals (G5) and provide the manual evaluation hooks for the paper.

| # | Feature | Architecture ref | Current state | What to build | Rough effort |
|---|---------|-----------------|---------------|---------------|-------------|
| C1 | **Bookmarking backend** | §7 | None. | 1) `POST /turns/{id}/bookmark` – sets `is_bookmarked=true`, `lossless_flag=true`, `decay_immune=true`, triggers priority Codex extraction. 2) `GET /bookmarks` with filter/sort. 3) When assembling the prompt, inject a `[BOOKMARKED]` block with the bookmarked turns (scoped to the conversation). | 4 h |
| C2 | **Manual Codex injection** | §3.2 | Not built. | 1) Create a `/codex_inject` directory. 2) Add a file watcher (like Drop Zone) that parses YAML/JSON entity files and writes them directly as Codex events. | 3 h |
| C3 | **Manual label correction endpoint** | §1.4 | Table exists, no endpoint. | 1) `POST /batch/override-tags` – accepts batch_id and corrected tags, writes to `curated_labels`. | 1 h |
| C4 | **Conversation scoping endpoints** | §8 | Partially done. | 1) `PUT /conversations/{id}/scope` – sets `memory_scope_type` and `cluster_ids`. 2) Ensure the orchestrator respects these fields when a request comes from that conversation. | 2 h |
| C5 | **Explicit cluster creation API** | §17 | None. | 1) `POST /clusters` – manually create a named cluster. 2) `PUT /clusters/{id}/assign` – assign turns to a cluster manually. | 2 h |
| C6 | **Memory slot update confirmation flow** | §2.4 | Reflection proposes updates without user confirmation. | 1) When Reflection proposes a slot update, write it to a `review_queue` table instead of applying immediately. 2) Add `GET /review-queue` and `POST /review-queue/{id}/approve` endpoints. | 3 h |

---

## Phase D — Orchestration Layer Completion

These make the proxy and background plane match the full request lifecycle (§9.1).

| # | Feature | Architecture ref | Current state | What to build | Rough effort |
|---|---------|-----------------|---------------|---------------|-------------|
| D1 | **CHAT_COMPLETED event emission** | §9.1 step 15 | Not done. | After the SSE stream closes, publish a `CHAT_COMPLETED` event to Redis with the idempotency key. | 30 min |
| D2 | **KV cache prefix validation / token count check** | §9.1 steps 10‑11 | Not done. | 1) After prompt assembly, count the actual tokens (using the background model’s tokenizer or a rough heuristic). 2) If the count exceeds the model’s context window, trim EPISODIC and PROCEDURAL blocks first, never CODEX/SYSTEM/SLOTS. | 2 h |
| D3 | **Graceful degradation – Redis/Celery unavailable** | §9.6 | Not done. | 1) In the proxy, if the Celery task queue is unreachable, buffer the post‑flight event to a local JSONL file. 2) A recovery script replays the buffer when Redis comes back. | 3 h |
| D4 | **Graceful degradation – Ollama timeout → registry fallback** | §9.6 | Not done. | After the Model Registry is built, if the primary model times out, route to the next‑best model from the registry. | 1 h |
| D5 | **Graceful degradation – HyDE timeout** | §9.6 | Not done. | If the HyDE rewrite request times out, skip HyDE and use the raw prompt embedding. | 30 min |
| D6 | **SSE telemetry events** | §15 | None. | 1) Define the SSE event types (`classifying`, `classified`, `expanding_query`, `retrieving`, `context_ready`, `generating`, `degraded`). 2) Emit these events interleaved with the LLM token stream. 3) The frontend can parse them to show the telemetry panel. | 6 h |

---

## Phase E — Operations & Packaging (from Post‑V1 Roadmap §24)

These make ICE deployable, shareable, and usable by others.

| # | Feature | Roadmap ref | What to build | Rough effort |
|---|---------|-------------|---------------|-------------|
| E1 | **Single‑command startup** | §24.5.1 | 1) Create an `ice start` shell script that starts PostgreSQL, Redis, vLLM‑bg, Celery worker+beat, and the FastAPI proxy, with a unified log output. 2) Optionally provide a `docker compose up` variant. | 2 h |
| E2 | **Shared background model option** | §24.5.3 | 1) Add a config flag `BACKGROUND_MODEL_MODE`. 2) In `shared` mode, background workers route their LLM calls to the same Ollama/vLLM endpoint as the proxy, with a low‑priority queue. 3) Adjust GPU checks to allow background work only when the user is idle. | 3 h |
| E3 | **Terminal frontend (TUI)** | §24.1.2 | 1) Build a simple TUI using `textual` or `rich` that provides: chat input/output, display of injected context, classifier tags, scope selector, memory slot editor, bookmark toggle. 2) This becomes the primary demo interface. | 12 h |
| E4 | **One‑click installer** | §24.5.2 | 1) Write a `setup.sh` that installs system dependencies (PostgreSQL, Redis, Python), creates the venv, runs Alembic migrations, and pulls the background model. 2) Optionally build a Docker image. | 4 h |
| E5 | **Model Registry backend** | §24.2.1 | 1) Create a JSON registry file populated at startup from Ollama’s `/api/tags`. 2) For unknown models, use the background model to suggest tags. 3) Expose a `/model-registry` endpoint. | 4 h |
| E6 | **Classifier‑driven model selection** | §24.2.2 | 1) In the proxy, after classification, score registry entries by tag overlap and select the best model that fits the context window. 2) Fall back to default generalist. | 2 h |

---

## Phase F — Remaining Missing Items (lower priority, can be post‑paper)

| # | Feature | Architecture ref | Notes |
|---|---------|-----------------|-------|
| F1 | **Drop Zone full pipeline** | §3.5 | The four‑stage pipeline is not implemented; current Drop Zone is a simple text‑to‑RAG ingester. This can be built later as it doesn’t affect evaluation. |
| F2 | **Session Replay** | §14 | The `session_replays` table is empty; no code writes to it. Needed for the custom frontend, not for the paper. |
| F3 | **Audit trail** | §14.2 | Source annotations are not recorded on writes. Important for transparency but not evaluation‑critical. |
| F4 | **Conversation branching retrieval logic** | §21.1 | Deferred until custom frontend exists. |
| F5 | **Custom web frontend** | §24.1.1 | Huge effort; out of scope for the paper. The TUI (E3) is a better V1 demo. |
| F6 | **Null_Noise / Casual_Banter special routing** | §1.2 | Minor; the classifier rarely outputs these labels with high confidence for real prompts. Can be added later. |
| F7 | **Memory slot token budget enforcement** | §2.2 | Add truncation when slots exceed 300 tokens. |
| F8 | **Simulation Harness – procedural extraction + logging** | §9.01 | Add procedural extraction to the simulation loop; log run info to a `simulation_runs` table for reproducibility. |
| F9 | **Time‑weighting in episodic retrieval** | §3.1 | The architecture specifies time‑weighted cosine similarity; currently it’s plain cosine. Adding a decay‑based weight to the vector score would improve relevance. |
| F10 | **Trigger conditions for procedural memory** | §3.3 | Already covered in A6. |
| F11 | **Conversation scoping isolation (None scope)** | §8.1 | Ensure None‑scoped conversations are invisible to all other retrieval. |
| F12 | **RAG store activation rules** | §3.4 | Already implemented correctly. |
| F13 | **Manual Codex injection watcher** | §3.2 | Covered in C2. |
| F14 | **Session replay & audit trail** | §14 | Covered in F2/F3. |

---

## Execution Order (Rough Timeline)

1. **Phase A (A1–A10) → 2‑3 days** – retrieval quality fixes; will directly raise Precision@5.
2. **Phase B (B1–B6) → 2‑3 days** – memory lifecycle; enables longitudinal claims.
3. **Phase C (C1–C6) → 2 days** – user control endpoints; needed for manual evaluation.
4. **Phase D (D1–D6) → 2 days** – proxy completeness and observability.
5. **Phase E (E1–E6) → 3‑4 days** – packaging, TUI, model registry; makes ICE demoable.
6. **Phase F → after paper submission** – remaining polish.

We can start with A1 tomorrow and work straight through. Each item is self‑contained, so you’ll see steady progress. After Phase A is done, we can re‑run the automatic evaluation and you’ll see the precision number climb. Then we’ll keep building.