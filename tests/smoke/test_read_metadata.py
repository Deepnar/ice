"""FINAL revision 5: every read records each input of the B2 decision and what preparing memory
cost in time (scripts/z1/production_parity.provenance_fields, seed_v3.prepare_with_trace)."""
from types import SimpleNamespace

from scripts.z1 import production_parity as pp


def test_the_gate_record_carries_the_breakdown_and_the_read_seconds():
    pre = pp.Preamble(classification=SimpleNamespace(p_ltm=0.2, p_temporal=0.0), scope={}, prompt_embedding=[],
                      conversation_id="c", turn_count=3, total_tokens=10.0, total_budget=1000, model_name="m",
                      timescope_mode="current", retrieve=False, p_need_mem=0.3,
                      b2_breakdown=dict(p_ltm=0.2, referential=True, logit=-0.8))
    memory = SimpleNamespace(source_checked=False, retrieve=True, action="rescue", rescue_score=1.5, read_seconds=12.5)
    gate = pp.provenance_fields(pre, memory)
    assert gate["b2_breakdown"] == dict(p_ltm=0.2, referential=True, logit=-0.8) and gate["read_seconds"] == 12.5
    assert pp.provenance_fields(pre)["read_seconds"] is None               # B2 prior only: nothing was read


def test_the_read_is_timed_around_the_shared_preparation():
    import inspect
    from scripts.z1 import seed_v3
    src = inspect.getsource(seed_v3.prepare_with_trace)
    assert src.index("started = time.perf_counter()") < src.index("memory = pp.prepare(") \
        < src.index("memory.read_seconds = ")
