"""Z2 S1b detail pass, offline: specifics from a long reply are admitted only with
real quotes, checked by the other family, and disputes go to the tie-breaker."""
import json
from types import SimpleNamespace as NS

from experiments.z2 import ledger, ledger_detail as D

LONG = " ".join(["filler"] * 520) + " Option one is Rust. Option two is Go because builds are fast."
PAIRS = [dict(turn_number=1, original_pair_sha256="s1", prompt="which language?", response=LONG)]


def test_detail_entries_checked_and_tie_broken(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "load_history", lambda h: PAIRS)
    (tmp_path / "led").mkdir()
    (tmp_path / "led" / "ledger.json").write_text(json.dumps(dict(entries=[])))
    proposed = {"entries": [
        dict(kind="advice", subject="option one", value="The assistant suggested Rust as option one.",
             speaker="assistant", modality="suggested",
             citations=[dict(turn=1, role="assistant", quote="Option one is Rust.")], supersedes=None, change=None),
        dict(kind="advice", subject="option two", value="The assistant suggested Go because builds are fast.",
             speaker="assistant", modality="suggested",
             citations=[dict(turn=1, role="assistant", quote="Option two is Go because builds are fast.")],
             supersedes=None, change=None),
        dict(kind="advice", subject="made up", value="Invented.", speaker="assistant", modality="suggested",
             citations=[dict(turn=1, role="assistant", quote="Option three is Zig.")], supersedes=None, change=None)]}

    def model(messages, **kw):
        system = messages[0]["content"]
        if system == D.DETAIL_SYSTEM:
            return NS(text=json.dumps(proposed), usage=None)
        lines = [json.loads(l) for l in messages[1]["content"].split("=====")[1].splitlines() if l.startswith("{")]
        key = "checks" if "CHECK" in messages[1]["content"] else "decisions"
        out = []
        for row in lines:                      # Rust confirmed; Go disputed by the checker, dropped by the tie-breaker
            ok = "Rust" in row["value"] and key == "checks"
            out.append(dict(id=row["id"], said_by="assistant" if ok else "nobody",
                            stated_as="suggested" if ok else "not_stated", wrong_detail=None, match=None))
        return NS(text=json.dumps({key: out}), usage=None)

    models = {r: (model, {"p": r}) for r in ("deepseek", "mimo", "muse")}
    stats = D.run("X", tmp_path / "led", tmp_path / "out", models=models)
    led = json.loads((tmp_path / "out" / "ledger.json").read_text())["entries"]
    assert stats["proposed"] == 3 and stats["dropped"] == 1           # the invented quote never enters
    assert {e["subject"]: e["status"] for e in led} == {"option one": "verified", "option two": "unconfirmed"}


def test_halves_split_at_a_paragraph_and_stay_substrings():
    from experiments.z2.ledger_detail import halves
    text = "Para one is here.\n\nPara two is longer, much longer.\n\nPara three."
    a, b = halves(text)
    assert a + b == text and a.endswith("\n\n") and text.startswith(a)


def test_a_detail_pass_with_failed_turns_is_not_finished(monkeypatch, tmp_path):
    """Z3 2026-11-04: failed turns were recorded and the pass exited 0, so the chain built questions on
    an answer key with gaps; now it exits 3 and the chain stops for a rerun."""
    import sys
    from experiments.z2 import ledger_detail as D
    from experiments.z2 import profiles
    monkeypatch.setattr(D, "run", lambda *a, **k: dict(failed_turns=[dict(turn=5, error="x")]))
    monkeypatch.setattr(profiles, "generator", lambda r: type("G", (), {"generate": None})())
    monkeypatch.setattr(profiles, "metadata", lambda r: {})
    monkeypatch.setattr("experiments.lme.cloud_provider.load_selected_env", lambda: None)
    out = D.ROOT / "logs" / "scratch-detail-exit-test"
    monkeypatch.setattr(sys, "argv", ["ledger_detail.py", "--history", "X", "--ledger", str(out), "--out", str(out)])
    assert D.main() == 3
