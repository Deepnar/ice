"""The GPU day's stage list: reads are held behind the correctness gates; evaluations are not."""
from experiments.z2 import gpu_day as G


def test_reads_wait_for_the_gates_and_evaluations_do_not():
    stages = {s.key: s for s in G.stages()}
    for key, stage in stages.items():
        if isinstance(stage, G.Reads) or key == "r9-locomo-slice":
            assert set(G.GATES) <= set(stage.needs), key
        if key.startswith("r8-"):
            assert not stage.needs, key
    assert stages["r2-coverage"].needs == tuple(f"r2-B-{c.removeprefix('retest_')}" for c in G.SWEEP)


def test_retest_reads_never_write_into_z2s_own_folders():
    for stage in G.stages():
        if isinstance(stage, G.Reads):
            assert "z2-retest-" in str(stage.out(None))
