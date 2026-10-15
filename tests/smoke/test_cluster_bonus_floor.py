"""Audit A12 (2026-10-15): cluster bonuses separate similar turns, never rescue unrelated ones.

Measured: tags ≤0.10 + entities ≤0.30 + session 0.10 lifted raw cosines near 0.1
over the 0.6 threshold; the dev history became one cluster and the largest
store's biggest cluster held 1,075/1,471 turns (51% below 0.6 raw).

Run:  uv run --no-sync pytest tests/smoke/test_cluster_bonus_floor.py -q
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.workers.clustering as cw  # noqa: E402
from src.api.config import settings  # noqa: E402


def _unit(deg):
    r = math.radians(deg)
    return [math.cos(r), math.sin(r)]


STATE = dict(cluster=NS(tags=["Software_&_Tech"]), embedding=_unit(0),
             cast={"atlas", "postgres", "fastapi", "ollama"}, sessions={"s1"})
FULL_BONUS = dict(turn_tags=["Software_&_Tech"], turn_entities={"atlas", "postgres", "fastapi", "ollama"},
                  turn_session="s1")


def test_an_unrelated_turn_cannot_be_carried_in_by_bonuses():
    raw = math.cos(math.radians(66))           # 0.41
    score = cw._score_against_cluster(_unit(66), state=STATE, **FULL_BONUS)
    assert math.isclose(score, raw) and score < settings.cluster_similarity_threshold


def test_a_borderline_similar_turn_still_gets_its_bonuses():
    score = cw._score_against_cluster(_unit(55), state=STATE, **FULL_BONUS)   # raw 0.574
    assert score >= settings.cluster_similarity_threshold


def test_pair_grouping_obeys_the_same_floor():
    a = dict(emb=_unit(0), entities={"atlas", "postgres", "fastapi", "ollama"}, session_id="s1")
    b = dict(emb=_unit(66), entities={"atlas", "postgres", "fastapi", "ollama"}, session_id="s1")
    assert cw._pair_score(a, b) < settings.cluster_similarity_threshold
