"""Z2 S1 ledger extraction, offline: windows, citation admission, change links,
output-cap splitting and resume — with a scripted fake writer, no cloud call."""

import json
from types import SimpleNamespace as NS

import pytest

from experiments.lme.cloud_provider import CloudCompletionError
from experiments.z2 import ledger

PAIRS = [dict(turn_number=i, original_pair_sha256=f"sha{i}",
              prompt=f"user line {i}: my cat is called Miso" if i == 3 else
                     (f"user line {i}: actually the cat is called Tofu now" if i == 30 else f"user line {i}"),
              response=f"assistant reply {i}") for i in range(1, 46)]
META = {"profile": "fake"}


class Writer:
    """Scripted replies keyed by the first new turn; counts calls."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def __call__(self, messages, *, temperature, max_output_tokens, session_id):
        first = int(messages[1]["content"].split("NEW turns ")[1].split("-")[0])
        self.calls.append((first, messages[1]["content"]))
        reply = self.script[first]
        if reply == "CUT":
            raise CloudCompletionError("cloud answer completion was not successful: length")
        return NS(text=json.dumps(reply), usage={"total_tokens": 1})


def entry(subject, value, turn, quote, **extra):
    return dict(kind="fact", subject=subject, value=value, speaker="user", modality="asserted",
                citations=[dict(turn=turn, role="user", quote=quote)],
                supersedes=None, change=None, **extra)


@pytest.fixture
def history(monkeypatch):
    monkeypatch.setattr(ledger, "load_history", lambda h: PAIRS)


def test_windows_cover_every_turn_once_as_new(history):
    plan = ledger.windows(PAIRS)
    assert [w["new"][0] for w in plan] == [1, 26]
    assert plan[1]["context"] == [21, 22, 23, 24, 25]
    assert sorted(t for w in plan for t in w["new"]) == list(range(1, 46))


def test_extraction_links_change_drops_bad_quotes_and_resumes(history, tmp_path):
    first_id = "deepseek:X:w0.0.0"
    writer = Writer({
        1: {"entries": [entry("user's cat", "The user's cat is called Miso.", 3, "my cat is called Miso"),
                        entry("bad", "Paraphrased.", 3, "my kitten is named Miso")]},
        26: {"entries": [entry("user's cat", "The user's cat is now called Tofu.", 30,
                               "the cat is called Tofu now")
                         | dict(supersedes=first_id, change="correction"),
                         entry("context", "Only in context.", 22, "user line 22")]},
    })
    path = ledger.run_writer("X", "deepseek", tmp_path, generate=writer, writer_meta=META)
    rows = [json.loads(line) for line in path.open()]
    assert [len(r["entries"]) for r in rows] == [1, 1]
    assert [d["reason"] for d in rows[0]["dropped"]] == ["quote_not_found"]
    assert [d["reason"] for d in rows[1]["dropped"]] == ["context_only"]
    second = rows[1]["entries"][0]
    assert second["supersedes"] == first_id and second["valid_from_turn"] == 30
    assert "Miso" in writer.calls[1][1].split("CONTEXT")[0]           # current ledger shown
    assert [e["subject"] for e in ledger.current_ledger(rows[0]["entries"] + rows[1]["entries"])] \
        == ["user's cat"] and ledger.current_ledger(rows[0]["entries"] + rows[1]["entries"])[0]["value"].endswith("Tofu.")
    again = Writer({})
    ledger.run_writer("X", "deepseek", tmp_path, generate=again, writer_meta=META)
    assert again.calls == []                                          # resume: no new calls
    assert path.read_text().count("\n") == 2


def test_changed_input_refuses_to_resume(history, tmp_path):
    ledger.run_writer("X", "deepseek", tmp_path, limit=1, generate=Writer({1: {"entries": []}}),
                      writer_meta=META)
    with pytest.raises(RuntimeError, match="differs"):
        ledger.run_writer("X", "deepseek", tmp_path, limit=1, generate=Writer({}),
                          writer_meta={"profile": "other"})


def test_output_cap_splits_window_in_halves(history, tmp_path):
    writer = Writer({1: "CUT", 13: {"entries": []}})
    writer.script[1] = "CUT"
    calls = []

    def generate(messages, **kw):
        first = int(messages[1]["content"].split("NEW turns ")[1].split("-")[0])
        last = int(messages[1]["content"].split("NEW turns ")[1].split("-")[1].split(":")[0])
        calls.append((first, last))
        if last - first + 1 > 13:
            raise CloudCompletionError("cloud answer completion was not successful: length")
        return NS(text='{"entries": []}', usage=None)

    path = ledger.run_writer("X", "deepseek", tmp_path, limit=1, generate=generate, writer_meta=META)
    record = json.loads(path.read_text())
    assert calls == [(1, 25), (1, 12), (13, 25)]
    assert [p["new"][0] for p in record["parts"]] == [1, 13]


def test_unparseable_reply_is_retried_once_and_never_repaired(history, tmp_path):
    replies = iter(['{"entries": [', "still not json"])
    def generate(messages, **kw):
        return NS(text=next(replies), usage=None)
    path = ledger.run_writer("X", "mimo", tmp_path, limit=1, generate=generate, writer_meta=META)
    record = json.loads(path.read_text())
    assert record["entries"] == []
    assert record["parse_failures"] == [
        {"part": 0, "error": "JSONDecodeError", "text": '{"entries": ['},
        {"part": 0, "error": "JSONDecodeError", "retry": True}]
    assert record["parts"][0]["retried"] is True


def test_retry_that_parses_is_kept(history, tmp_path):
    good = {"entries": [entry("user's cat", "The user's cat is called Miso.", 3, "my cat is called Miso")]}
    replies = iter(['{"entries": [}', json.dumps(good)])
    path = ledger.run_writer("X", "mimo", tmp_path, limit=1,
                             generate=lambda m, **k: NS(text=next(replies), usage=None), writer_meta=META)
    record = json.loads(path.read_text())
    assert len(record["entries"]) == 1 and len(record["parse_failures"]) == 1


def test_interrupted_run_resumes_at_the_next_window(history, tmp_path):
    script = {1: {"entries": [entry("user's cat", "The user's cat is called Miso.", 3, "my cat is called Miso")]},
              26: {"entries": []}}
    first = Writer(script)
    ledger.run_writer("X", "deepseek", tmp_path, limit=1, generate=first, writer_meta=META)
    resumed = Writer(script)
    path = ledger.run_writer("X", "deepseek", tmp_path, generate=resumed, writer_meta=META)
    assert [c[0] for c in first.calls] == [1] and [c[0] for c in resumed.calls] == [26]
    assert "Miso" in resumed.calls[0][1].split("CONTEXT")[0]          # earlier entries still shown
    assert [json.loads(line)["index"] for line in path.open()] == [0, 1]


def test_context_overflow_refusal_splits_like_a_cut(history, tmp_path):
    class TooLong(Exception):
        status_code = 400
    calls = []

    def generate(messages, **kw):
        first = int(messages[1]["content"].split("NEW turns ")[1].split("-")[0])
        last = int(messages[1]["content"].split("NEW turns ")[1].split("-")[1].split(":")[0])
        calls.append((first, last))
        if last - first + 1 > 13:
            raise TooLong("This model's maximum context length is 131072 tokens")
        return NS(text='{"entries": []}', usage=None)

    ledger.run_writer("X", "deepseek", tmp_path, limit=1, generate=generate, writer_meta=META)
    assert calls == [(1, 25), (1, 12), (13, 25)]


def test_context_pairs_are_trimmed_to_budget(history, monkeypatch):
    monkeypatch.setattr(ledger, "pair_tokens", lambda originals, turn: 5_000)
    plan = ledger.windows(PAIRS)
    later = plan[1:]
    assert later and all(w["context"] == [w["new"][0] - 2, w["new"][0] - 1] for w in later)  # 2 × 5,000 fits 12,000


def test_transient_provider_errors_retry_but_quota_does_not():
    import httpx
    import openai
    from experiments.lme.cloud_provider import ProviderAccessError
    from experiments.z2 import profiles
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    server = openai.InternalServerError("boom", response=httpx.Response(500, request=request), body=None)
    attempts = []

    def flaky(*a, **k):
        attempts.append(1)
        if len(attempts) < 3:
            raise server
        return "ok"
    waits = []
    assert profiles.retrying(flaky, "mimo", sleep=waits.append)() == "ok"
    assert len(attempts) == 3 and waits == [10, 30]

    def always(*a, **k):
        raise server
    with pytest.raises(openai.InternalServerError):
        profiles.retrying(always, "mimo", sleep=lambda s: None)()

    calls = []
    def quota(*a, **k):
        calls.append(1)
        raise ProviderAccessError("mimo", "quota/rate-limit", 429, "limit reached")
    with pytest.raises(ProviderAccessError):
        profiles.retrying(quota, "mimo", sleep=lambda s: None)()
    assert len(calls) == 1


def test_literal_tabs_inside_quotes_parse():
    reply = '{"entries": [{"kind": "fact", "quote": "Section\tTime\tScored?"}]}'
    assert ledger.parse(reply)[0]["quote"] == "Section\tTime\tScored?"


def test_non_string_supersedes_is_rejected_not_a_crash(history):
    raw = entry("user's cat", "The cat is Tofu.", 30, "the cat is called Tofu now") | dict(
        supersedes={"id": "x"}, change="correction")
    entry_, reason = ledger.admit(PAIRS, raw, dict(new=[30], context=[]), {"x"}, "id")
    assert entry_ is None and reason == "bad_change"


def test_an_aborted_generation_retries_but_a_length_cut_does_not():
    from experiments.lme.cloud_provider import CloudCompletionError
    from experiments.z2 import profiles
    assert profiles.transient(CloudCompletionError("cloud answer completion was not successful: abort", receipt={}))
    assert not profiles.transient(CloudCompletionError("cloud answer completion was not successful: length", receipt={}))
