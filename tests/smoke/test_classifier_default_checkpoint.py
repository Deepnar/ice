"""G39: a bare ``PyTorchClassifier()`` must load the checkpoint the SETTING names.

It used to default to the literal `models/classifier/ice_classifier.pt` — the
**v1** head (25 logits, softmax context head, 384-dim input) — while
`settings.classifier_model_path` pointed at `ice_classifier_v4_schema2.pt`
(27 logits, all-sigmoid, native 1024).

The severity is that it did not fail. That v1 file is still on disk, so the bare
constructor loaded two-month-old weights and returned plausible labels; it had
already produced one wrong measurement before it was caught. The request path
was never exposed — `api/core.py` passes the setting explicitly — so the blast
radius was scratch scripts and new tests, which are exactly the callers that
write a bare constructor and the ones whose output nobody cross-checks.

`load_checkpoint` and `load_schema` are stubbed: the question is *which path is
chosen*, and answering it by loading 200 MB of weights would make this a slow
test of torch rather than a fast test of a default.

Run:  uv run pytest tests/smoke/test_classifier_default_checkpoint.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.config import settings  # noqa: E402


def _paths_used(monkeypatch, **kwargs):
    """Construct with *kwargs* and report the paths the loaders were handed."""
    import src.classifier.classifier as mod

    seen = {}

    class _Schema:
        template_version = 2
        input_dim = 1024
        head_widths = (11, 12, 4)

        def labels(self, head):
            return []

    def fake_load_schema(path):
        seen["schema"] = path
        return _Schema()

    def fake_load_checkpoint(path, schema=None):
        seen["model"] = path
        return object(), {"schema_version": 2, "tag_threshold": 0.65}

    monkeypatch.setattr(mod, "load_schema", fake_load_schema)
    monkeypatch.setattr(mod, "load_checkpoint", fake_load_checkpoint)
    # The real one loads the shared Qwen encoder; this test is about a default,
    # not about the encoder.
    monkeypatch.setattr(mod, "get_embedder", lambda: object())
    mod.PyTorchClassifier(**kwargs)
    return seen


def test_bare_constructor_uses_the_settings_checkpoint(monkeypatch):
    seen = _paths_used(monkeypatch)
    assert seen["model"].endswith(Path(settings.classifier_model_path).name), (
        f"a bare constructor loaded {seen['model']!r}, not the configured "
        f"{settings.classifier_model_path!r}")


def test_bare_constructor_uses_the_settings_schema(monkeypatch):
    seen = _paths_used(monkeypatch)
    assert seen["schema"] == settings.label_schema_path


def test_the_v1_checkpoint_is_never_the_default(monkeypatch):
    """Named explicitly, because this is the specific artifact that was loaded.

    Not folded into the test above: that one would still pass if the default
    moved to some third stale file, and the point is that a *valid but wrong*
    checkpoint is what makes this failure symptomless.
    """
    seen = _paths_used(monkeypatch)
    assert not seen["model"].endswith("ice_classifier.pt"), (
        "the bare constructor is back on the v1 head")


def test_an_explicit_path_still_wins(monkeypatch):
    """The other side — if the argument were ignored rather than defaulted,
    every assertion above would pass while the class silently lost the ability
    to load a rollback or D5's comparison baseline."""
    seen = _paths_used(monkeypatch,
                       model_path="models/classifier/ice_classifier_v2_final.pt",
                       schema_path="data/labeled/label_schema.json")
    assert seen["model"].endswith("ice_classifier_v2_final.pt")
    assert seen["schema"] == "data/labeled/label_schema.json"
