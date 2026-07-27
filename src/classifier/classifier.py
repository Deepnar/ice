import structlog
import torch
from typing import List, Optional

from src.memory.embedder import get_embedder, slice384

from . import templates
from .di3 import run_di3
from .model import load_checkpoint
from .schema import (CONTEXT_RELIANCE, INTENT, LONG_TERM_MEMORY, TOPIC,
                     ZERO_SHOT, finalize_context_scalars, load_schema)
from .schemas import ClassificationResult

logger = structlog.get_logger("ice.classifier")


class PyTorchClassifier:
    """The pre-flight prompt classifier.

    B1 made this schema-driven end to end: the label lists, the head widths and
    the slice offsets all come from ``label_schema.json`` via ``schema.py``, and
    the encoder input comes from ``templates.py`` — the same renderer the
    training pipeline uses, which is what closes the train/inference mismatch.

    It serves **either** checkpoint generation. Since B1's promotion
    (2026-07-27) the live path holds a **v2** checkpoint — 27 logits, all-sigmoid,
    native 1024; a v1 checkpoint is 25 logits, softmax ctx head, 384-dim input and
    survives as the rollback artifact and as D5's comparison baseline. The loaded
    checkpoint declares which it is and this class adapts — callers see the same
    ``ClassificationResult`` either way, which is what makes a rollback a file
    swap rather than a code change.
    """

    def __init__(self, model_path="models/classifier/ice_classifier.pt",
                 schema_path="data/labeled/label_schema.json"):
        self.schema = load_schema(schema_path)
        self.model, self.meta = load_checkpoint(model_path, schema=self.schema)

        # A v1 checkpoint brings its own (frozen) schema — the live schema file
        # describes v2 heads the old weights don't have.
        self.active_schema = getattr(self.model, "schema", self.schema)
        self.schema_version = int(self.meta.get("schema_version", 1))
        self.template_version = int(self.meta.get("template_version",
                                                  self.active_schema.template_version))
        self.input_dim = int(self.meta.get("input_dim", self.active_schema.input_dim))

        # Calibration rides with the weights — see _tags_above.
        from src.api.config import settings as _settings
        self.tag_threshold = float(self.meta.get("tag_threshold")
                                   or _settings.classifier_threshold)

        self.TOPIC_LABELS = list(self.active_schema.labels(TOPIC))
        self.INTENT_LABELS = list(self.active_schema.labels(INTENT))
        self.CONTEXT_RELIANCE_LABELS = list(self.active_schema.labels(CONTEXT_RELIANCE))

        # G23/C17: the process-shared native-width embedder
        # (src/memory/embedder.py) — retrieval and every store writer reach this
        # same instance via `classifier.embedder`. A v1 head still consumes the
        # 384-dim MRL prefix (slice384) of that same encode; v2 heads take the
        # native width, which is what retires the slice (A9).
        self.embedder = get_embedder()

        logger.info("classifier_loaded", schema_version=self.schema_version,
                    template_version=self.template_version,
                    input_dim=self.input_dim, heads=self.active_schema.head_widths,
                    tag_threshold=self.tag_threshold, path=model_path)

    def _get_context_turns(self, conversation_id: str, n: int = templates.CONTEXT_TURNS,
                           max_total_words: int = templates.CONTEXT_MAX_WORDS) -> str:
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
            texts = []
            for t in turns:
                # Prefer summary, fall back to raw text (truncated)
                text = t.summary_text or templates.cap_turn(t.raw_text or "")
                if text:
                    texts.append(text)
            # Shared budget logic — the offline pipeline truncates identically.
            return templates.truncate_context(texts, max_total_words=max_total_words)
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
                if di3_result.context_reliance == LONG_TERM_MEMORY:
                    ml_result.reference_signal = True
                return ml_result
            # DI3 fast-path (noise/code/sentiment/meta): keep its decision, but
            # derive a p_ltm scalar so B2 can still combine it.
            self._finalize_confidence(di3_result)
            return di3_result

        return self._run_ml_classifier(prompt, conversation_id)

    def _encode(self, text: str) -> torch.Tensor:
        """Render → encode → match the head's expected width."""
        vec = self.embedder.encode(text, convert_to_tensor=True)
        if self.input_dim != vec.shape[-1]:
            # Only legal narrowing is the v1 MRL prefix (bit-identical to the
            # old truncate_dim=384 output — see embedder.slice384).
            if self.input_dim == 384:
                vec = slice384(vec)
            else:
                raise ValueError(
                    f"classifier expects {self.input_dim}-dim input but the "
                    f"embedder produced {vec.shape[-1]}")
        return vec.unsqueeze(0).float()

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
                # G26 validation: surface that CL7's prior-turn prefix is live.
                logger.info("cl7_context_prefix", words=len(context_text.split()))

            rendered = templates.render(prompt, context_text,
                                        version=self.template_version)
            logits = self.model(self._encode(rendered))

            # Schema-driven: no head offsets appear in this file.
            probs_by_head = {}
            for head in self.active_schema.heads:
                block = logits[:, head.slice]
                if head.activation == "softmax":
                    probs_by_head[head.name] = torch.softmax(block, dim=1).squeeze(0)
                else:
                    probs_by_head[head.name] = torch.sigmoid(block).squeeze(0)

        # Context-reliance probabilities aren't thresholded into tags — they are
        # read off raw_probs by the derivation layer below (D6).
        topic_tags = self._tags_above(TOPIC, probs_by_head[TOPIC])
        intent_tags = self._tags_above(INTENT, probs_by_head[INTENT])

        raw_probs = []
        for head in self.active_schema.heads:
            raw_probs.extend(probs_by_head[head.name].tolist())
        max_confidence = max(raw_probs)

        result = ClassificationResult(
            topic_tags=topic_tags,
            intent_tags=intent_tags,
            context_reliance=ZERO_SHOT,   # replaced by _finalize_confidence
            raw_probs=raw_probs,
            max_confidence=max_confidence,
            prompt=prompt,
        )
        result.head_confidences = {name: float(p.max())
                                   for name, p in probs_by_head.items()}
        self._finalize_confidence(result)
        return result

    def _tags_above(self, head_name: str, probs) -> List[str]:
        """Labels over the tag threshold, falling back to the single argmax.

        **The threshold travels with the checkpoint.** A decision threshold is a
        property of the trained weights, not of the installation: v1 was
        calibrated at 0.3, and the v2 head's own sweep puts its optimum at 0.65
        (fitted on val, B1 run 2). One global setting cannot be right for both,
        and both remain in play — v2 is live since 2026-07-27 and the v1
        checkpoint is still loadable for D5's gate and for rollback.
        So ``tag_threshold`` is stamped into the checkpoint by
        ``sweep_threshold.py`` and read here, with
        ``settings.classifier_threshold`` as the fallback for checkpoints that
        predate the stamp. Promoting a model therefore promotes its calibration
        with it, and no .env edit has to be remembered.

        Z1-prep's decision-threshold stage re-sweeps this; it writes the same
        field.
        """
        threshold = self.tag_threshold
        labels = self.active_schema.labels(head_name)
        tags = [labels[i] for i in range(len(labels)) if probs[i] > threshold]
        if not tags:
            tags = [labels[int(torch.argmax(probs).item())]]
        return tags

    def _finalize_confidence(self, result: ClassificationResult) -> None:
        """Populate the B2 context-reliance scalars on *result*.

        Delegates to ``schema.finalize_context_scalars`` — the derivation is pure
        label logic and lives where the label layout does, so it can be tested
        (and reasoned about) without loading a checkpoint.

        The classifier no longer *forces* Long_Term_Memory — the old
        creative/software hard overrides are gone; those signals are now bumps in
        ``src.api.memory_decision``. This just exposes an honest, per-head
        confidence for that decision to combine.
        """
        finalize_context_scalars(result, self.active_schema)
