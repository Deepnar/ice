"""FINAL cloud-answerer subset: asks read under every compared condition, a fixed hash
order, and the unit files copied unchanged."""

import json

from experiments.final import cloud_subset as C
from experiments.z2 import read_passes as R


def test_choose_and_stage(tmp_path):
    read = tmp_path / "read"
    for c in C.CONDITIONS:
        for i in range(5):
            R.write_unit(R.unit_path(read, c, f"S9:F{i:03d}@50#0"), dict(condition=c, n=i))
    R.write_unit(R.unit_path(read, "full_default", "S9:F999@50#0"), dict(condition="full_default"))  # ICE only: excluded
    asks = C.choose(read, 3)
    assert len(asks) == 3 and "S9:F999@50#0" not in asks and asks == C.choose(read, 3)
    out = C.stage_units(read, tmp_path / "cloud", asks)
    for c in C.CONDITIONS:
        assert sorted(p.stem for p in (out / "units" / c).glob("*.json")) == sorted(asks)
    assert json.loads((out / "units" / "full_default" / f"{asks[0]}.json").read_text())["condition"] == "full_default"
