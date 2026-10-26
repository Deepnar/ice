"""B7: the classifier's input-normalisation and init variants build, run, and survive
a checkpoint round trip; the shipped configuration is unchanged."""

import torch

from src.classifier.model import ICEClassifier, load_checkpoint


def test_variants_round_trip_and_default_is_unchanged(tmp_path):
    x = torch.randn(64, 1024) * 0.03
    for norm in ("none", "standardize", "layernorm"):
        model = ICEClassifier(input_norm=norm, init="kaiming" if norm == "layernorm" else "default").eval()
        model.fit_input_stats(x)
        path = tmp_path / f"{norm}.pt"
        torch.save(model.checkpoint_dict(), path)
        loaded, meta = load_checkpoint(str(path))
        assert meta["input_norm"] == norm
        assert torch.allclose(model(x), loaded.eval()(x), atol=1e-6)
    plain = ICEClassifier()
    assert plain.input_norm == "none" and not hasattr(plain, "in_mean") and not hasattr(plain, "in_norm")
