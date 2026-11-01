"""Private outputs stay under logs/: checked on the path as given, so a linked logs folder passes and
a ".." escape does not (2026-11-01: a linked logs/z2-run-main refused every question of a run)."""
from scripts.z1 import private_logs as P


def test_a_linked_folder_under_logs_passes_and_an_escape_does_not(tmp_path, monkeypatch):
    elsewhere = tmp_path / "main-checkout-logs"
    elsewhere.mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "z2-run-main").symlink_to(elsewhere)
    monkeypatch.setattr(P, "LOGS", tmp_path / "logs")
    monkeypatch.chdir(tmp_path)
    assert P.under_logs("logs/z2-run-main/s12-r10") and P.under_logs(tmp_path / "logs" / "x.json")
    assert not P.under_logs("logs/../src") and not P.under_logs(elsewhere / "x.json")


def test_the_replay_writer_accepts_a_linked_folder(tmp_path, monkeypatch):
    from scripts.z1 import seed_v3
    elsewhere = tmp_path / "main-checkout-logs"
    elsewhere.mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "linked").symlink_to(elsewhere)
    monkeypatch.setattr(P, "LOGS", tmp_path / "logs")
    monkeypatch.chdir(tmp_path)
    assert seed_v3.private_output("logs/linked/q1/seed.jsonl").parent.exists()
