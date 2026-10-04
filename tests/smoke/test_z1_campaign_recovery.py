"""Recover observed transient failures; never retry stale/invalid/quota state."""
import json
import subprocess
import sys
import urllib.error

import pytest

from scripts.z1 import campaign_recovery as recovery
from scripts.z1 import run_v3_campaign as campaign


@pytest.mark.parametrize("stage,evidence,expected", [
    ("seed", {"error_type": "ExtractionOutputError"}, True),
    ("seed", {"error_type": "OperationalError"}, True),
    ("seed", {"error_type": "MemoryError"}, False),
    ("seed", {"error_type": "ValueError"}, False),
    ("seed", {"error_type": "KeyError"}, False),
    ("answers", {"error_type": "APITimeoutError"}, True),
    ("answers", {"status": 503}, True),
    ("answers", {"status": 429}, False),
    ("answers", {"error_type": "ProviderAccessError", "status": 503}, False),
    ("answers", {"error_type": "RuntimeError"}, False),
    ("judge", {"reason": "api_http_503"}, True),
    ("judge", {"reason": "api_http_503", "operator_required": True}, False),
    ("judge", {"reason": "api_http_429"}, False),
    ("judge", {"reason": "unparseable"}, False),
    ("snapshot", {"error_type": "TimeoutError"}, False),
    ("report", {"error_type": "TimeoutError"}, False),
])
def test_failure_classification(stage, evidence, expected):
    assert recovery.recoverable(stage, evidence) is expected


def test_old_failure_cannot_mask_new_identity_or_startup_error(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text(json.dumps({"event": "failed", "error_type": "ExtractionOutputError"}) + "\n")
    before = recovery.stamp(path)
    assert not recovery.recoverable("seed", recovery.failure_evidence(tmp_path, "seed", "", before))
    with path.open("a") as f:
        f.write(json.dumps({"event": "failed", "error_type": "ValueError"}) + "\n")
    assert recovery.failure_evidence(tmp_path, "seed", "", before)["error_type"] == "ValueError"


def test_judge_network_failure_receipt_reaches_recovery(tmp_path, monkeypatch):
    from scripts.z1 import judge_answers
    monkeypatch.setattr(judge_answers, "_env", lambda key: {
        "PROBE_API_KEY": "synthetic-test-credential",
        "PROBE_API_BASE_URL": "https://test.invalid/v1",
        "PROBE_MODEL": "synthetic-judge"}[key])
    def disconnected(*_a, **_kw):
        raise urllib.error.URLError(ConnectionRefusedError("fixture transport outage"))
    monkeypatch.setattr(judge_answers.urllib.request, "urlopen", disconnected)
    path = recovery.failure_path(tmp_path, "judge", "vector")
    before = recovery.stamp(path)
    verdict = judge_answers.judge_one("question", "source", "a", "b", expected_answer="fact")
    assert verdict["verdict"] == "ERROR" and verdict["error_type"] == "URLError"
    path.write_text(json.dumps({"results": [{"order_verdicts": [{"raw": verdict}]}]}))
    assert recovery.recoverable("judge", recovery.failure_evidence(tmp_path, "judge", "vector", before))


def test_actual_child_failure_then_resume_is_bounded_and_saved(tmp_path, monkeypatch):
    """Real Python child writes a failure, then requires --resume to succeed."""
    scripts = tmp_path / "scripts/z1"
    scripts.mkdir(parents=True)
    (scripts / "seed_v3.py").write_text('''import json,sys
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--out')+1])
if '--resume' not in sys.argv:
 p.write_text(json.dumps({'event':'failed','stage':'post_flight','error_type':'ExtractionOutputError'})+'\\n')
 sys.exit(1)
with p.open('a') as f:f.write(json.dumps({'event':'clock','conversation':'fixture','turn':1})+'\\n')
''')
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "database_environment", lambda *_a: {})
    monkeypatch.setattr(campaign, "progress_totals", lambda *_a: (1, 0))
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    cfg = {"checkpoint_every": 10, "probe_panel": "existing"}
    assert campaign.execute(tmp_path, cfg, "seed") == 0
    receipts = [json.loads(s) for s in (tmp_path / "campaign-attempts.jsonl").read_text().splitlines()]
    assert [r['event'] for r in receipts] == ['started','retrying','started','completed']
    assert not (tmp_path / "campaign-pause.json").exists()
    assert json.loads((tmp_path / "stage-status.json").read_text())["seed"] == "complete_diagnostic"


def test_persistent_failure_pauses_after_two_real_children(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts/z1"
    scripts.mkdir(parents=True)
    (scripts / "seed_v3.py").write_text('''import json,sys
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--out')+1])
with p.open('a') as f:f.write(json.dumps({'event':'failed','stage':'post_flight','error_type':'ExtractionOutputError'})+'\\n')
sys.exit(1)
''')
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "database_environment", lambda *_a: {})
    monkeypatch.setattr(campaign, "progress_totals", lambda *_a: (1, 0))
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    with pytest.raises(subprocess.CalledProcessError):
        campaign.execute(tmp_path, {"checkpoint_every": 10, "probe_panel": "existing"}, "seed")
    rows = [json.loads(s) for s in (tmp_path / "campaign-attempts.jsonl").read_text().splitlines()]
    assert sum(r['event'] == 'started' for r in rows) == 2
    pause = json.loads((tmp_path / "campaign-pause.json").read_text())
    assert pause['attempts'] == 2 and pause['progress_retained']


def test_corrupt_configuration_gets_clean_operator_error(tmp_path, monkeypatch, capsys):
    (tmp_path / "campaign.json").write_text('{')
    monkeypatch.setattr(campaign, "private_path", lambda p: p)
    monkeypatch.setattr(sys, "argv", ['run_v3_campaign.py','--run-dir',str(tmp_path),'--run'])
    assert campaign.main() == 1
    assert 'cannot start/resume' in capsys.readouterr().err


def test_seed_failure_releases_owned_models_without_hiding_original_error(monkeypatch):
    from scripts.z1 import seed_v3
    from src.workers import bg_client_factory as factory
    monkeypatch.setattr(sys, 'argv', ['seed_v3.py','--probe-panel','existing','--out','logs/fixture.jsonl'])
    monkeypatch.setattr(seed_v3, 'load_plan', lambda **_kw: ({'fixture': []}, {}, [], []))
    calls = []
    def fail(*_a, **_kw):
        raise ConnectionError('fixture worker outage')
    def cleanup():
        calls.append('release')
        raise OSError('fixture cleanup outage')
    monkeypatch.setattr(seed_v3, 'run', fail)
    monkeypatch.setattr(factory, 'release_owned_models', cleanup)
    with pytest.raises(ConnectionError, match='worker outage'):
        seed_v3.main()
    assert calls == ['release']
