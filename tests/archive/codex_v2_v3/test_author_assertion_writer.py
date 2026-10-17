import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))
from author_assertion_case import exercise_writer


@pytest.mark.skipif(not os.getenv('ICE_TEST_DATABASE'), reason='requires disposable PostgreSQL')
def test_author_check_changes_graph_decision_and_keeps_source(monkeypatch):
    assert exercise_writer(monkeypatch)['withheld_relations'] == 3


@pytest.mark.skipif(not os.getenv('ICE_TEST_DATABASE'), reason='requires disposable PostgreSQL')
def test_argument_mismatch_withholds_even_when_act_and_support_are_affirmative(monkeypatch):
    assert exercise_writer(monkeypatch, argument_only=True)['withheld_relations'] == 3


@pytest.mark.skipif(not os.getenv('ICE_TEST_DATABASE'), reason='requires disposable PostgreSQL')
def test_opposing_actor_source_retains_exact_original_with_three_admitted_relations(monkeypatch):
    assert exercise_writer(monkeypatch, argument_only=True, argument_case=True)['claims'] == 3
