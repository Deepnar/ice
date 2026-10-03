"""Progress counts saved work, survives rollback, and sees real child writes."""
import io
import json
import subprocess
import sys

from scripts.z1.campaign_progress import ArtifactProgress, TerminalProgress
from scripts.z1.replay_checkpoint import atomic_json


def append(path, row):
    with path.open("a") as sink:
        sink.write(json.dumps(row) + "\n")


def clock(number):
    return {"event": "clock", "conversation": "synthetic", "turn": number}


def test_seed_counts_full_turn_not_write_and_resets_after_checkpoint_restore(tmp_path):
    trace = tmp_path / "seed.jsonl"
    reader = ArtifactProgress(tmp_path, "seed", turns=10, probes=2)
    append(trace, {"event": "written", "conversation": "synthetic", "turn": 1})
    assert reader.observe().done == 0
    append(trace, {"event": "as_of_probe", "probe_id": "p"})
    append(trace, clock(1))
    append(trace, clock(1))  # Duplicate observations cannot inflate work.
    assert reader.observe().done == 1
    assert "probes 1/2" in reader.observe().detail
    checkpoint = tmp_path / "seed.recovery"
    (checkpoint / "generation-fixture").mkdir(parents=True)
    atomic_json(checkpoint / "generation-fixture" / "recovery.json", {"state": {"completed": {"synthetic": 1}}})
    atomic_json(checkpoint / "current.json", {"generation": "generation-fixture"})
    assert "durable 1" in reader.observe().detail
    committed = trace.read_text()
    append(trace, {"event": "as_of_probe", "probe_id": "unfinished"})
    append(trace, clock(2))
    assert reader.observe().done == 2
    trace.write_text(committed)
    append(trace, {"event": "resume", "completed_turns": {"synthetic": 1}})
    # Rollback + append can already be bigger than the previously observed file.
    # Prefix comparison must still discard the unfinished probe/turn.
    assert reader.observe().done == 1
    assert "probes 1/2" in reader.observe().detail


def test_partial_jsonl_event_is_not_consumed_or_counted(tmp_path):
    path = tmp_path / "seed.jsonl"
    reader = ArtifactProgress(tmp_path, "seed", turns=10)
    value = json.dumps(clock(1))
    path.write_text(value[:10])
    assert reader.observe().done == 0
    with path.open("a") as sink:
        sink.write(value[10:] + "\n")
    assert reader.observe().done == 1


def test_answers_count_only_successful_unique_records(tmp_path):
    atomic_json(tmp_path / "answers-full.json", {"records": [
        {"probe_id": "a", "answer": "answer"}, {"probe_id": "a", "answer": "answer"},
        {"probe_id": "b", "answer": "", "error": "outage"},
        {"probe_id": "c", "answer": "prefix", "error": "incomplete"}]})
    value = ArtifactProgress(tmp_path, "answers", arm="full", admitted=3).observe()
    assert (value.done, value.total) == (1, 3)


def test_judge_counts_saved_first_order_when_second_fails_and_deduplicates(tmp_path):
    path = tmp_path / "judge-full-vs-recent_only.partial.json"
    reader = ArtifactProgress(tmp_path, "judge", arm="recent_only", admitted=2)
    first = {"verdict": "TIE"}
    atomic_json(path, {"pending_probe": {"probe_id": "a", "first_order": first}})
    assert reader.observe().done == 1
    failed = {"probe_id": "a", "winner": "ERROR", "order_verdicts": [
        {"winner": "TIE", "raw": first}, {"winner": "ERROR", "raw": {"verdict": "ERROR"}}]}
    atomic_json(path, {"results": [failed]})
    assert reader.observe().done == 1
    completed = {**failed, "winner": "UNCERTAIN", "order_verdicts": [
        {"winner": "full", "raw": {"verdict": "A"}}, {"winner": "TIE", "raw": first}]}
    atomic_json(path, {"results": [completed], "failed_attempts": [failed],
                       "pending_probe": {"probe_id": "b", "first_order": {"verdict": "ERROR"}}})
    assert (reader.observe().done, reader.observe().total) == (2, 4)


def test_actual_child_writes_advance_observer_and_failure_stops_thread(tmp_path):
    stream = io.StringIO()
    reader = ArtifactProgress(tmp_path, "seed", turns=2, probes=0)
    child = """import json, sys, time
with open(sys.argv[1], 'w') as sink:
    for number in (1, 2):
        time.sleep(0.12)
        sink.write(json.dumps({'event': 'clock', 'conversation': 'synthetic', 'turn': number}) + '\\n')
        sink.flush()
"""
    with TerminalProgress(reader, "1/5 seed", stream=stream, interval=0.01) as progress:
        subprocess.run([sys.executable, "-c", child, str(tmp_path / "seed.jsonl")], check=True)
    display = stream.getvalue()
    assert "0/2" in display and "1/2" in display and "2/2" in display
    assert "completed" in display and not progress.thread.is_alive()
    try:
        with TerminalProgress(reader, "1/5 seed", stream=stream, interval=0.01) as stopped:
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert "stopped; state retained" in stream.getvalue()
    assert not stopped.thread.is_alive()


def test_tty_updates_one_line_and_snapshot_does_not_invent_a_percentage(tmp_path):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    stream = Tty()
    with TerminalProgress(ArtifactProgress(tmp_path, "snapshot"), "2/5 snapshot", stream=stream):
        pass
    assert "\r\033[2K" in stream.getvalue()
    assert "[working]" in stream.getvalue() and "completed\n" in stream.getvalue()


def test_observation_error_is_visible_and_does_not_fail_replay(tmp_path):
    (tmp_path / "answers-full.json").write_text("broken JSON")
    stream = io.StringIO()
    with TerminalProgress(ArtifactProgress(tmp_path, "answers", arm="full"), "3/5 answers", stream=stream):
        pass
    assert "progress unavailable (JSONDecodeError)" in stream.getvalue()
