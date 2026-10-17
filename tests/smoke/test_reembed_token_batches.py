"""G78: re-embed groups texts by PADDED tokens, not by row count."""

from types import SimpleNamespace as NS

from src.memory.reembed import token_batches


class _Tok:
    def __call__(self, texts, **_kw):
        return {"input_ids": [[0] * len(t.split()) for t in texts]}


ENC = NS(tokenizer=_Tok(), max_seq_length=10_000)


def _padded(group, texts):
    return len(group) * max(len(texts[i].split()) for i in group)


def test_padded_budget_every_text_once_and_long_text_alone():
    texts = ["w " * 8000] + ["w " * n for n in (10, 300, 50, 700, 5, 400)] * 6
    groups = token_batches(ENC, texts, budget=2048)
    flat = sorted(i for g in groups for i in g)
    assert flat == list(range(len(texts)))                     # nothing dropped or doubled
    assert [0] in groups                                       # the 8K text alone
    assert all(_padded(g, texts) <= 2048 for g in groups if g != [0])
    assert len(groups) > 2                                     # the budget actually split them


def test_large_budget_keeps_one_batch_and_cap_bounds_length():
    texts = ["w " * n for n in (10, 20, 30)]
    assert token_batches(ENC, texts, budget=10_000) == [[2, 1, 0]]
    capped = NS(tokenizer=_Tok(), max_seq_length=8)
    assert token_batches(capped, ["w " * 100] * 4, budget=32) == [[0, 1, 2, 3]]
