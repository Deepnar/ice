"""Hybrid Retrieval Orchestrator – Phase A hardened: decay filtering, access-weighting,
wide‑net full‑vector, Codex/Procedural scoping, HyDE rewriting, procedural trigger matching,
micro‑NER integration, and dynamic token budget."""

import hashlib
import os
import re
import uuid
from typing import List, Optional, Dict
from dataclasses import dataclass
from openai import OpenAI
import structlog
from sqlalchemy.orm import Session
from sqlalchemy import bindparam, text
from pgvector.sqlalchemy import Vector as PgVector
from transformers import AutoTokenizer
import torch

from src.api.config import settings
from src.memory.models import (
    EpisodicMemory,
    CodexEntity,
    CodexEdge,
    ProceduralMemory,
    MemorySlot,
)
from src.classifier.schemas import ClassificationResult
logger = structlog.get_logger("ice.retrieval")


@dataclass(frozen=True)
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
        self.max_retrieval_tokens = 5000

        # Load micro‑NER model (fallback to None if not available)
        self.ner_model = self._load_ner_model()
        self.ner_tokenizer = self._load_ner_tokenizer()

    def _load_ner_model(self):
        from src.classifier.ner_model import MicroNER
        model = MicroNER()
        path = "models/ner/ner_model.pt"
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            return model
        return None

    def _load_ner_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        except Exception:
            return None

    def _extract_entities_with_ner(self, text: str) -> List[str]:
        # Fallback to regex if NER not ready
        if self.ner_model is None or self.ner_tokenizer is None:
            return list(set(re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', text)))

        encoding = self.ner_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = encoding["input_ids"]
        if not token_ids:
            return []
        token_strs = self.ner_tokenizer.convert_ids_to_tokens(token_ids)
        offsets = encoding["offset_mapping"]

        embeddings = self.embedder.encode(token_strs, convert_to_tensor=True, show_progress_bar=False)
        model_device = next(self.ner_model.parameters()).device
        if embeddings.device != model_device:
            embeddings = embeddings.to(model_device)
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.float()

        with torch.no_grad():
            logits = self.ner_model(embeddings.unsqueeze(0))  # (1, T, 3)
            preds = torch.argmax(logits, dim=-1).squeeze(0)   # (T,)

        # ---- Step 1: standard BIO merging ----
        entities = []          # list of (start_char, end_char, entity_string)
        current_start = None
        current_end = None
        current_tokens = []

        for i, p in enumerate(preds.tolist()):
            tok_start, tok_end = offsets[i]
            if p == 0:  # B-ENT
                if current_start is not None:
                    # save previous entity using original character span
                    entities.append((current_start, current_end,
                                     text[current_start:current_end].strip()))
                    current_tokens = []
                current_start = tok_start
                current_end = tok_end
                current_tokens.append(token_strs[i])
            elif p == 1 and current_start is not None:  # I-ENT
                current_end = tok_end
                current_tokens.append(token_strs[i])
            else:
                if current_start is not None:
                    entities.append((current_start, current_end,
                                     text[current_start:current_end].strip()))
                    current_tokens = []
                    current_start = None
        if current_start is not None:
            entities.append((current_start, current_end,
                             text[current_start:current_end].strip()))

        # ---- Step 2: glue consecutive entities that are adjacent in the text ----
        if len(entities) >= 2:
            glued = []
            prev_start, prev_end, prev_str = entities[0]
            for i in range(1, len(entities)):
                curr_start, curr_end, curr_str = entities[i]
                # If there is only whitespace between the two entities, merge them
                if text[prev_end:curr_start].strip() == "":
                    # Merge: extend the previous entity
                    prev_end = curr_end
                    prev_str = text[prev_start:prev_end].strip()
                else:
                    glued.append((prev_start, prev_end, prev_str))
                    prev_start, prev_end, prev_str = curr_start, curr_end, curr_str
            glued.append((prev_start, prev_end, prev_str))
            entities = glued

        # Return only the entity strings, discarding empty ones
        return [e[2] for e in entities if len(e[2]) > 0]

    def set_budget_from_turn_count(self, turn_count: int):
        """CL4: Dynamic token budget based on conversation depth.
        Short conversations (<60 turns) get 3,000 tokens;
        longer ones scale linearly with turns, up to 15,000 tokens.
        """
        self.max_retrieval_tokens = min(15000, max(2000, turn_count * 50))

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
        # ── Safety override: if we have a conversation scope, always check memory ──
        if classification.context_reliance == "Zero_Shot" and conversation_id:
            classification.context_reliance = "Long_Term_Memory"

        # Keep the explicit creative/lore guard (belt and suspenders)
        if "Creative_&_Media" in classification.topic_tags:
            classification.context_reliance = "Long_Term_Memory"

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
            hyde_prompt = self._hyde_rewrite(classification.prompt, conversation_id)
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

        # ── Dynamic leg weighting (blended over all active intents) ──

        # Base balanced weights
        base_weights = {
            "bm25": 0.8,
            "vector": 1.0,
            "codex": 0.5,
            "procedural": 0.2,
            "rag": 1.0,
        }

        # Profile definitions: each profile → (intents, weight_override)
        PROFILES = [
            ({"Factual_Retrieval", "Utility_Formatting"},
            {"vector": 1.2, "bm25": 1.0, "codex": 0.1, "procedural": 0.1}),
            ({"Troubleshooting", "Strategic_Planning"},
            {"vector": 1.0, "bm25": 0.8, "codex": 0.3, "procedural": 1.2}),
            ({"Generation", "Ideation", "Open_Exploration"},
            {"vector": 0.6, "bm25": 0.4, "codex": 1.2, "procedural": 0.1}),
            ({"Emotional_Processing", "Analysis_&_Summarization", "Decision_Making"},
            {"vector": 1.1, "bm25": 0.5, "codex": 0.9, "procedural": 0.0}),
            ({"Casual_Banter", "Null_Noise"},
            {"vector": 0.5, "bm25": 0.2, "codex": 0.0, "procedural": 0.0}),
        ]

        # Build a mapping from intent label → its profile’s override
        intent_to_profile_weights = {}
        for intents_in_profile, override in PROFILES:
            for intent in intents_in_profile:
                intent_to_profile_weights[intent] = override

        # Blend: each active intent contributes equally
        active_intents = classification.intent_tags
        num_active = len(active_intents) if active_intents else 1

        blend_weights = {leg: 0.0 for leg in base_weights}
        for tag in active_intents:
            profile_weights = intent_to_profile_weights.get(tag)
            if profile_weights:
                for leg, w in profile_weights.items():
                    blend_weights[leg] += w / num_active
            else:
                # Unknown intent – use base weights
                for leg, w in base_weights.items():
                    blend_weights[leg] += w / num_active

        # Fallback to base weights if nothing matched
        if all(v == 0.0 for v in blend_weights.values()):
            blend_weights = dict(base_weights)

        # --- TOPIC OVERRIDES (cumulative) ---
        if "Creative_&_Media" in set(classification.topic_tags):
            blend_weights["codex"] = blend_weights.get("codex", 0.5) + 0.3
        if "Software_&_Tech" in set(classification.topic_tags):
            blend_weights["procedural"] = blend_weights.get("procedural", 0.2) + 0.4

        # Clamp to zero (no negative weights)
        for leg in blend_weights:
            blend_weights[leg] = max(0.0, blend_weights[leg])

        # Fuse, diversify, deduplicate, trim
        fused = self._apply_rrf(legs, alpha_map=blend_weights)
        diversified = self._session_diversify(fused, conversation_id, max_per_conversation=3)
        deduped = self._deduplicate(diversified)
        final = self._enforce_token_budget(deduped)

        # Strengthen retrieved turns (access count + decay boost)
        self._strengthen_retrieved(final)

        return final

    # ------------------------------------------------------------------
    # HyDE rewriting
    # ------------------------------------------------------------------
    def _hyde_rewrite(self, prompt: str, conversation_id: str = None) -> Optional[str]:
        # Collect the last 5 turns as context for the rewrite
        context_text = ""
        if conversation_id:
            try:
                recent = self.db.query(EpisodicMemory).filter_by(
                    conversation_id=conversation_id
                ).order_by(EpisodicMemory.timestamp.desc()).limit(5).all()
                recent.reverse()
                parts = []
                for t in recent:
                    parts.append(t.raw_text[:300])
                context_text = "\n".join(parts)
            except Exception:
                pass

        rewrite_prompt = (
            f"Recent conversation:\n{context_text}\n\nUser's question:\n{prompt}"
            if context_text else prompt
        )

        try:
            resp = self.bg_client.chat.completions.create(
                model="Qwen/Qwen2.5-3B-Instruct-AWQ",
                messages=[
                    {"role": "system", "content": (
                        "You are a query rewriting engine. Take the user's question and rewrite it "
                        "as a dense, factual search query that would retrieve the relevant past conversation. "
                        "Include key entities and omit polite phrasing. Output ONLY the rewritten query, no other text."
                    )},
                    {"role": "user", "content": rewrite_prompt},
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

        topic_filter = ""
        conv_filter = "AND conversation_id = :conv_id" if conv_id else ""

        query = text(f"""
            SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id, is_bookmarked,
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
            LIMIT 20
        """)

        params = {"search_terms": search_terms, "min_decay": 0.2}
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
        topic_filter = ""
        conv_filter = "AND conversation_id = :conv_id" if conv_id else ""

        query = text(f"""
            SELECT id, raw_text, summary_text, lossless_flag, inject_raw, conversation_id, is_bookmarked,
                (1 - (embedding <=> :prompt_embedding)) * COALESCE(decay_score, 1.0) as score
            FROM episodic_memory
            WHERE embedding IS NOT NULL
            {topic_filter}
            {conv_filter}
            AND decay_score > :min_decay
            AND is_archived = false
            ORDER BY score DESC
            LIMIT 20
        """).bindparams(bindparam("prompt_embedding", type_=PgVector))
        params = {"prompt_embedding": prompt_embedding, "min_decay": 0.2}
        if conv_id:
            params["conv_id"] = conv_id

        try:
            rows = self.db.execute(query, params).fetchall()
            return self._rows_to_fragments(rows, "episodic")
        except Exception as err:
            logger.error("vector_retrieval_failed", error=str(err))
            self.db.rollback()
            return []
        
    
    def _match_entities_by_similarity(self, entity_strings: List[str], threshold: float = 0.85) -> List:
        """Match extracted entity strings to CodexEntity rows using vector similarity.
        Falls back to canonical name / alias exact match when embeddings are unavailable."""
        if not entity_strings:
            return []

        # Embed all candidate strings
        candidate_embeddings = self.embedder.encode(entity_strings, convert_to_tensor=False, show_progress_bar=False)
        # Fetch all entities that have embeddings
        all_entities = self.db.query(CodexEntity).filter(CodexEntity.embedding != None).all()

        matched = []
        seen_ids = set()
        for candidate_str, candidate_emb in zip(entity_strings, candidate_embeddings):
            # 1) Vector similarity
            best_score = 0.0
            best_entity = None
            for ent in all_entities:
                if ent.id in seen_ids:
                    continue
                emb = ent.embedding
                if emb is None:
                    continue
                # cosine similarity = dot product of normalized vectors
                dot = sum(a * b for a, b in zip(candidate_emb, emb))
                score = dot  # embeddings are already normalised by SentenceTransformer
                if score > best_score and score >= threshold:
                    best_score = score
                    best_entity = ent
            if best_entity is not None:
                matched.append(best_entity)
                seen_ids.add(best_entity.id)
                continue

            # 2) Fallback: exact canonical name / alias match
            from sqlalchemy import or_
            norm = candidate_str.lower().strip()
            fallback = self.db.query(CodexEntity).filter(
                or_(CodexEntity.canonical_name == norm, CodexEntity.aliases.any(norm))
            ).first()
            if fallback and fallback.id not in seen_ids:
                matched.append(fallback)
                seen_ids.add(fallback.id)

        return matched

    # ------------------------------------------------------------------
    # Codex graph traversal (conversation‑scoped, NER‑powered)
    # ------------------------------------------------------------------
    def _codex_graph(self, classification, scope: Optional[dict] = None) -> List[ContextFragment]:
        prompt = classification.prompt
        entity_strings = self._extract_entities_with_ner(prompt)

        if entity_strings:
            matched = self._match_entities_by_similarity(entity_strings)
        else:
            from src.retrieval.mera import is_mera_candidate, map_category_to_filters, enumerate_entities
            if is_mera_candidate(prompt):
                filters = map_category_to_filters(self.db, prompt)
                matched = enumerate_entities(self.db, filters.get("tags", []), filters.get("relations", []))
            else:
                return []

        if not matched:
            return []

        try:
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
            for entity in matched:
                if allowed_entity_ids is not None and entity.id not in allowed_entity_ids:
                    continue
                self._traverse_graph(entity, 0, 3, visited, context_texts)   # depth 3

            if context_texts:
                combined = "\n\n".join(context_texts)
                score = 1.0
                # Score boost for active/high‑strength edges
                if any(e.confidence == "active" and e.strength >= 2.0 for e in
                    self.db.query(CodexEdge).filter(
                        CodexEdge.source_id.in_([e.id for e in matched]),
                        CodexEdge.valid_until == None
                    ).all()):
                    score *= 1.5

                return [ContextFragment(
                    text=combined,
                    source_type="codex",
                    score=score,
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
                if allowed_batch_ids is not None:
                    if not any(bid in allowed_batch_ids for bid in (pattern.source_batch_ids or [])):
                        continue
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
    # RAG lookup
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
    def _apply_rrf(self, legs: Dict[str, List[ContextFragment]], alpha_map: Dict[str, float] = None, k: int = 60) -> List[ContextFragment]:
        from dataclasses import replace
        if alpha_map is None:
            alpha_map = {}

        rrf_scores: Dict[str, float] = {}
        fragment_registry: Dict[str, ContextFragment] = {}

        for leg_name, fragments in legs.items():
            weight = alpha_map.get(leg_name, 1.0)
            fragments.sort(key=lambda x: x.score, reverse=True)
            for rank, frag in enumerate(fragments, start=1):
                frag_hash = hashlib.sha256(frag.text.encode('utf-8')).hexdigest()
                if frag_hash not in fragment_registry:
                    fragment_registry[frag_hash] = frag
                rrf_scores[frag_hash] = rrf_scores.get(frag_hash, 0.0) + (weight / (k + rank))

        fused = []
        for frag_hash, score in rrf_scores.items():
            original = fragment_registry[frag_hash]
            new_frag = replace(original, score=score)
            fused.append(new_frag)

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

    def _enforce_token_budget(self, fragments, max_tokens=None):
        if max_tokens is None:
            max_tokens = self.max_retrieval_tokens
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

        fused = self._apply_rrf({"fallback": fragments}, alpha_map={"fallback": 1.0})
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
            if getattr(row, "is_bookmarked", False):
                score_val *= 1.5

            fragments.append(ContextFragment(
                text=text,
                source_type=source_type,
                score=float(score_val),
                token_count=int(len(text.split()) * 1.33),
                source_batch_id=str(row.id),
                conversation_id=str(row.conversation_id) if row.conversation_id else None
            ))
        return fragments