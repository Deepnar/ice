"""Token counter: concurrent first use waits for the tokenizer instead of silently
estimating, and every estimate fallback warns."""
import threading
import time

from src.memory import tokens


class SlowTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_concurrent_first_use_waits_for_the_load(monkeypatch):
    monkeypatch.setattr(tokens, "_tokenizer", None)
    monkeypatch.setattr(tokens, "_load_attempted", False)
    loads = []

    def slow_load(*_a, **_k):
        loads.append(1)
        time.sleep(0.3)
        return SlowTokenizer()

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", slow_load)
    text = "one two three four five six seven eight"          # 8 words, chars/4 = 9
    results = []
    threads = [threading.Thread(target=lambda: results.append(tokens.count(text))) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [8] * 6 and loads == [1]


def test_estimate_fallback_warns_every_time(monkeypatch):
    seen = []
    monkeypatch.setattr(tokens, "_tokenizer", None)
    monkeypatch.setattr(tokens, "_load_attempted", True)          # a failed load
    monkeypatch.setattr(tokens.logger, "warning", lambda event, **kw: seen.append((event, kw)))
    assert tokens.count("abcdefgh") == tokens.estimate("abcdefgh")
    tokens.count("ijklmnop")
    assert [e for e, _ in seen] == ["token_count_estimated", "token_count_estimated"]
    assert seen[0][1]["reason"] == "tokenizer_unavailable"
