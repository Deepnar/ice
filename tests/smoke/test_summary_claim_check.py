"""Fix pass: the optional second reading of summary claims keeps only exact
restatements, and a failed check keeps the NLI verdicts with a warning."""

from src.memory import summary_support as S


def records():
    ok = {"status": "supported", "reason": "entailed", "scores": {}}
    return [dict(text="A asked B.", premise="A asked B.", verification=dict(ok)),
            dict(text="B asked A.", premise="A asked B.", verification=dict(ok)),
            dict(text="C left.", premise="?", verification={"status": "unknown", "reason": "x", "scores": {}})]


def test_not_exact_claims_become_unknown_and_unchecked_ones_stay():
    seen = []
    def checker(items):
        seen.extend(items)
        return [c == s for c, s in items]
    out = S.check_claims(records(), checker=checker)
    assert [r["verification"]["status"] for r in out] == ["supported", "unknown", "unknown"]
    assert out[1]["verification"]["reason"] == "claim_check_not_exact"
    assert len(seen) == 2                                   # only NLI-supported claims are checked


def test_a_failed_check_keeps_the_nli_verdicts():
    def checker(items):
        raise TimeoutError("bg model")
    out = S.check_claims(records(), checker=checker)
    assert [r["verification"]["status"] for r in out] == ["supported", "supported", "unknown"]
