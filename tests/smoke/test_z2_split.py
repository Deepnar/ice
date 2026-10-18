"""Z2 S6 split (spec §5 item 4): 60/40 by base family, variants follow their base,
the FINAL part is sealed, and Z2 loading never opens the sealed file."""

import builtins
import json

import pytest

from experiments.z2 import split as S


def families(n=200):
    out = [dict(id=f"B:fam{i:03d}", type="fact_recall") for i in range(n)]
    out += [dict(id=f"B:fam{i:03d}:messy", type="messy_variant", base_family=f"B:fam{i:03d}") for i in range(0, n, 7)]
    return out


def test_share_is_about_sixty_percent_and_variants_follow_their_base():
    z2, final = S.split(families())
    bases = [f for f in z2 + final if f["type"] != "messy_variant"]
    share = sum(S.goes_to_z2(f) for f in bases) / len(bases)
    assert 0.5 < share < 0.7
    z2_ids = {f["id"] for f in z2}
    for f in z2 + final:
        if f["type"] == "messy_variant":
            assert (f["id"] in z2_ids) == (f["base_family"] in z2_ids)


def test_split_is_deterministic_and_never_resealed(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = S.write_split(tmp_path / "a", "B", families())
    b = S.write_split(tmp_path / "b", "B", families())
    assert a["final_sha256"] == b["final_sha256"] and S.verify_seal(tmp_path / "a")
    with pytest.raises(RuntimeError, match="already sealed"):
        S.write_split(tmp_path / "a", "B", families())


def test_z2_loading_never_opens_the_sealed_file(tmp_path, monkeypatch):
    S.write_split(tmp_path, "B", families(), controls=[dict(id="B:ctl000")])
    opened = []
    real_open = builtins.open
    def spy(path, *a, **k):
        opened.append(str(path))
        return real_open(path, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    data = S.load_z2(tmp_path / "z2-families.json")
    assert data["families"] and data["controls"] == [dict(id="B:ctl000")]
    assert not any(p.endswith(S.SEALED_NAME) for p in opened)
    with pytest.raises(PermissionError):
        S.load_z2(tmp_path / S.SEALED_NAME)


def test_a_final_family_smuggled_into_z2_is_refused(tmp_path):
    z2, final = S.split(families())
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(dict(families=z2 + final[:1])))
    with pytest.raises(ValueError, match="FINAL family"):
        S.load_z2(path)
