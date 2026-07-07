import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from .model import ICEClassifier
from .di3 import run_di3
from .schemas import ClassificationResult
from sqlalchemy.orm import Session


class PyTorchClassifier:
    def __init__(self, model_path="models/classifier/ice_classifier.pt",
                 schema_path="data/labeled/label_schema.json"):
        # These lists are fixed – the order must match training
        self.TOPIC_LABELS = [
            "Software_&_Tech", "STEM_&_Academics", "Business_&_Finance",
            "Creative_&_Media", "Admin_&_Productivity", "Lifestyle_&_Health",
            "Social_&_Relationships", "World_&_Current_Events", "Meta_AI",
            "Null_Noise", "General_Reference_&_Trivia"
        ]
        self.INTENT_LABELS = [
            "Factual_Retrieval", "Troubleshooting", "Generation", "Ideation",
            "Analysis_&_Summarization", "Strategic_Planning", "Decision_Making",
            "Emotional_Processing", "Utility_Formatting", "Casual_Banter",
            "Open_Exploration"
        ]
        self.CONTEXT_RELIANCE_LABELS = [
            "Zero_Shot", "Long_Term_Memory", "Real_Time_Search"
        ]

        # Load model on CPU
        self.model = ICEClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        # Embedder also on CPU
        # Embedder also on CPU – Qwen3-Embedding truncated to 384 dim for compatibility
        self.embedder = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            device="cpu",
            truncate_dim=384
        )

    def _get_context_turns(self, conversation_id: str, n: int = 3, max_total_words: int = 500) -> str:
        """Return a truncated, summary‑preferring context string from the last *n* turns."""
        # Local import to avoid circular dependency at module level
        from src.api.db import SessionLocal
        db = SessionLocal()
        try:
            from src.memory.models import EpisodicMemory
            turns = (
                db.query(EpisodicMemory)
                .filter_by(conversation_id=conversation_id)
                .order_by(EpisodicMemory.timestamp.desc())
                .limit(n)
                .all()
            )
            turns.reverse()
            parts = []
            total_words = 0
            for t in turns:
                # Prefer summary, fall back to raw text (truncated)
                text = t.summary_text or ""
                if not text and t.raw_text:
                    words = t.raw_text.split()
                    text = " ".join(words[:150]) + "…" if len(words) > 150 else t.raw_text
                if not text:
                    continue
                word_count = len(text.split())
                if total_words + word_count > max_total_words:
                    remaining = max_total_words - total_words
                    if remaining > 20:
                        w = text.split()
                        text = " ".join(w[:remaining]) + "…"
                        parts.append(text)
                    break
                parts.append(text)
                total_words += word_count
            return "\n".join(parts)
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def classify(
        self,
        prompt: str,
        conversation_history: Optional[List[str]] = None,
        conversation_length: int = 0,
        conversation_id: Optional[str] = None,
    ) -> ClassificationResult:
        """Public entry point.  Runs DI3 first, falls back to ML.
        When *conversation_id* is given, the last 3 turns are used as context
        (auto‑truncated) to improve the ML classifier's accuracy.
        """
        if conversation_history is None:
            conversation_history = []
        di3_result = run_di3(prompt, conversation_length, conversation_history)
        if di3_result is not None:
            # DI3 fired its reference/anaphora rule (blank topic/intent, ctx=LTM):
            # get the *real* tags + ctx probabilities from the ML head and carry
            # the anaphora as a signal for B2's combination — NOT a forced LTM.
            if not di3_result.topic_tags or not di3_result.intent_tags:
                ml_result = self._run_ml_classifier(prompt, conversation_id)
                if di3_result.context_reliance == "Long_Term_Memory":
                    ml_result.reference_signal = True
                return ml_result
            # DI3 fast-path (noise/code/sentiment/meta): keep its decision, but
            # derive a p_ltm scalar so B2 can still combine it.
            self._finalize_confidence(di3_result)
            return di3_result

        return self._run_ml_classifier(prompt, conversation_id)

    def _run_ml_classifier(self, prompt: str, conversation_id: Optional[str] = None) -> ClassificationResult:
        """Original ML classification path (now private)."""
        with torch.no_grad():
            # Build context text if conversation_id is available
            context_text = None
            if conversation_id:
                try:
                    context_text = self._get_context_turns(conversation_id)
                except Exception:
                    context_text = None

            if context_text:
                prefixed_prompt = (
                    f"Conversation context (summarized):\n{context_text}\n\n"
                    f"Given the above conversation and the user's latest prompt, "
                    f"predict:\n"
                    f"1. TOPIC: what is the subject (Software_&_Tech, Creative_&_Media, etc.)\n"
                    f"2. INTENT: what is the user trying to do (Factual_Retrieval, Troubleshooting, etc.)\n"
                    f"3. CONTEXT RELIANCE: does the user need memory (Zero_Shot, Long_Term_Memory, Real_Time_Search)\n\n"
                    f"User prompt: {prompt}"
                )
            else:
                prefixed_prompt = (
                    f"Given a user prompt, predict:\n"
                    f"1. TOPIC: what is the subject (Software_&_Tech, Creative_&_Media, etc.)\n"
                    f"2. INTENT: what is the user trying to do (Factual_Retrieval, Troubleshooting, etc.)\n"
                    f"3. CONTEXT RELIANCE: does the user need memory (Zero_Shot, Long_Term_Memory, Real_Time_Search)\n\n"
                    f"User prompt: {prompt}"
                )
            embedding = self.embedder.encode(prefixed_prompt, convert_to_tensor=True).unsqueeze(0).float()
            outputs = self.model(embedding)                     # (1, 25)

            topic_out = outputs[:, :11]                         # (1, 11)
            intent_out = outputs[:, 11:22]                      # (1, 11)
            ctx_out = outputs[:, 22:]                           # (1, 3)

            topic_probs = torch.sigmoid(topic_out).squeeze(0)   # (11,)
            intent_probs = torch.sigmoid(intent_out).squeeze(0) # (11,)
            ctx_probs = torch.softmax(ctx_out, dim=1).squeeze(0) # (3,)

        # Build tag lists
        topic_tags = [self.TOPIC_LABELS[i] for i in range(len(self.TOPIC_LABELS))
                      if topic_probs[i] > 0.3]
        intent_tags = [self.INTENT_LABELS[i] for i in range(len(self.INTENT_LABELS))
                       if intent_probs[i] > 0.3]
        if not topic_tags:
            topic_tags = [self.TOPIC_LABELS[torch.argmax(topic_probs).item()]]
        if not intent_tags:
            intent_tags = [self.INTENT_LABELS[torch.argmax(intent_probs).item()]]
        context_reliance = self.CONTEXT_RELIANCE_LABELS[torch.argmax(ctx_probs).item()]

        # Combine probabilities
        raw_probs = topic_probs.tolist() + intent_probs.tolist() + ctx_probs.tolist()
        max_confidence = max(raw_probs)

        result = ClassificationResult(
            topic_tags=topic_tags,
            intent_tags=intent_tags,
            context_reliance=context_reliance,
            raw_probs=raw_probs,
            max_confidence=max_confidence,
            prompt=prompt,
        )
        self._finalize_confidence(result)
        return result

    def _finalize_confidence(self, result: ClassificationResult) -> None:
        """Populate the B2 context-reliance confidence scalars (p_ltm / p_rts /
        ctx_confidence) on *result*.

        The classifier no longer *forces* Long_Term_Memory — the old
        creative/software hard overrides are gone; those signals are now bumps
        in ``src.api.memory_decision``. This just exposes an honest, per-head
        confidence for that decision to combine.
        """
        probs = result.raw_probs or []
        ctx = probs[22:25] if len(probs) >= 25 else []
        if ctx and any(v > 0.0 for v in ctx):
            # ML head — order is [Zero_Shot, Long_Term_Memory, Real_Time_Search].
            result.p_ltm = float(ctx[1])
            result.p_rts = float(ctx[2])
            ordered = sorted(ctx, reverse=True)
            result.ctx_confidence = float(ordered[0] - ordered[1])
        else:
            # DI3 fast-path (raw_probs all zero): derive a prior from the label
            # DI3 chose so B2 still has a scalar to combine.
            prior = {
                "Long_Term_Memory": (0.85, 0.05),
                "Zero_Shot": (0.12, 0.05),
                "Real_Time_Search": (0.15, 0.70),
            }.get(result.context_reliance, (0.30, 0.05))
            result.p_ltm, result.p_rts = prior
            result.ctx_confidence = float(result.max_confidence)