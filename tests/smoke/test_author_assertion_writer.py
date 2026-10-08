import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from author_assertion_case import exercise_writer


@pytest.mark.skipif(not os.getenv('ICE_TEST_DATABASE'), reason='requires disposable PostgreSQL')
def test_author_check_changes_graph_decision_and_keeps_source(monkeypatch):
    assert exercise_writer(monkeypatch)['withheld_relations'] == 3
