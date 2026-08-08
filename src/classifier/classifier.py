import structlog
import torch
from typing import List, Optional

from src.memory.embedder import fit_width, get_embedder
from src.paths import resolve

from . import templates
from .model import load_checkpoint
from .schema import (CONTEXT_RELIANCE, INTENT, TOPIC, ZERO_SHOT,
                     finalize_context_scalars, load_schema)
from .schemas import ClassificationResult

logger = structlog.get_logger("ice.classifier")


class PyTorchClassifier:
    """The pre-flight prompt classifier — since D8, the *only* one.

    B1 made this schema-driven end to end: the label lists, the head widths and
    the slice offsets all come from ``label_schema.json`` via ``schema.py``, and
    the encoder input comes from ``templates.py`` — the same renderer the
    training pipeline uses, which is what closes the train/inference mismatch.

    **DI3 is gone (D8, 2026-07-27.)** A rule-based pre-classifier used to short-
    circuit this one on five density heuristics. Measured against the promoted v2
    head on the 9,441 held-out rows it would have intercepted, it lost every
    slice on every metric, so keeping it would have meant two disagreeing
    classifiers where the worse one wins by running first. `eval_di3.py` in the
    pipeline holds the numbers.

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
        # G31: anchored to the install, not the CWD. `load_schema` already did
        # this for itself; the checkpoint did not, so from the wrong directory
        # construction died on a FileNotFoundError for a file that was there.
        model_path = resolve(model_path)
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
        # same instance via `classifier.embedder`. v2 heads take the native
        # width; a rolled-back v1 head takes the 384-dim MRL prefix of the same
        # encode, and `_encode` asks embedder.fit_width which one applies (A9a).
        self.embedder = get_embedder()

        logger.info("classifier_loaded", schema_version=self.schema_version,
                    template_version=self.template_version,
                    input_dim=self.input_dim, heads=self.active_schema.head_widths,
                    tag_threshold=self.tag_threshold, path=model_path)

    def _get_context_turns(self, conversation_id: str, n: int = None,
                           max_total_words: int = None) -> str:
        """Return a truncated, summary‑preferring context string from the last *n* turns.

        G9: both default to None and resolve from settings in the body — as
        default arguments they were evaluated once at import, so a Z1 sweep of
        the classifier's context window would have changed nothing."""
        # Local import, matching this module's existing idiom (see the
        # threshold resolution above) — classifier.py keeps config out of its
        # module scope on purpose.
        from src.api.config import settings as _settings
        if n is None:
            n = _settings.classifier_context_turns
        if max_total_words is None:
            max_total_words = _settings.classifier_context_max_words
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
    def classify(self, prompt: str,
                 conversation_id: Optional[str] = None) -> ClassificationResult:
        """Public entry point. When *conversation_id* is given, the last 3 turns
        are used as context (auto-truncated) to improve accuracy."""
        return self._run_ml_classifier(prompt, conversation_id)

    def _encode(self, text: str) -> torch.Tensor:
        """Render → encode → match the head's expected width → the head's device.

        C16: the embedder now runs on the GPU by default while the classifier
        head is loaded with `map_location="cpu"`, so the encode comes back as a
        cuda tensor and the head's first Linear raises "Expected all tensors to
        be on the same device". Moving the vector rather than the head is
        deliberate: the head is ~25k parameters and its forward pass is not
        worth a device transfer of its own, while the 0.6B encoder genuinely
        wants the GPU. One 1024-float copy per classification is free.
        """
        vec = fit_width(self.embedder.encode(text, convert_to_tensor=True),
                        self.input_dim)
        head_device = next(self.model.parameters()).device
        return vec.unsqueeze(0).float().to(head_device)

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
