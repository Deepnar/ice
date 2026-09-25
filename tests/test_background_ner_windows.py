"""Background NER must reach late entities without GLiNER word truncation."""

import re
from types import SimpleNamespace as NS

from src.retrieval import ner_utils


class WordBoundModel:
    def __init__(self, limit=8):
        self.config = NS(max_len=limit)
        self.data_processor = NS(words_splitter=self.split)
        self.calls = []

    @staticmethod
    def split(text):
        return [(match.group(), match.start(), match.end())
                for match in re.finditer(r"\w+|[^\w\s]", text)]

    def predict_entities(self, text, labels, threshold):
        assert len(self.split(text)) <= self.config.max_len
        self.calls.append(text)
        found = text.find('TailEntity')
        return ([{'text': 'TailEntity', 'label': 'person', 'start': found,
                  'end': found + len('TailEntity')}]
                if found >= 0 else [])


def test_model_word_windows_cover_punctuation_heavy_tail(monkeypatch):
    model = WordBoundModel()
    monkeypatch.setattr(ner_utils, '_load_background_ner', lambda: model)
    source = ' '.join(f'term{i}!' for i in range(30)) + ' TailEntity'
    assert ner_utils._extract_background(source, labels=['person']) == ['TailEntity']
    assert len(model.calls) > 1
    assert all(f'term{i}!' in '\n'.join(model.calls) for i in range(30))


def test_short_background_piece_is_sent_once(monkeypatch):
    model = WordBoundModel()
    monkeypatch.setattr(ner_utils, '_load_background_ner', lambda: model)
    assert ner_utils._extract_background('TailEntity arrived.', labels=['person']) == ['TailEntity']
    assert model.calls == ['TailEntity arrived.']


def test_unknown_model_boundary_declines_background_tier(monkeypatch):
    model = WordBoundModel()
    model.data_processor = NS()
    monkeypatch.setattr(ner_utils, '_load_background_ner', lambda: model)
    assert ner_utils._extract_background('TailEntity arrived.', labels=['person']) is None
    assert model.calls == []
