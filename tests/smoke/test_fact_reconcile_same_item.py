"""Z2 S8 supersession guard: a newer fact replaces an earlier one only when the model
says both are about the same specific thing; restatements stay "same", and two
different runs/files/batches of one kind stay both_true."""

from types import SimpleNamespace as NS

from src.memory import facts


def verdict(**answer):
    earlier = NS(text="The 20-prompt test showed 6 LTM.", modality="shared", mentioned_at=None)
    claim = {"text": "The 58-prompt test showed 21 LTM.", "modality": "shared"}
    return facts.reconcile(claim, earlier, ask=lambda e, n: answer, when=None)


def test_replaces_needs_the_same_item():
    assert verdict(same_information=False, both_can_be_true=False, same_item=True) == "replaces"
    assert verdict(same_information=False, both_can_be_true=False, same_item=False) == "both_true"
    assert verdict(same_information=True, both_can_be_true=True, same_item=True) == "same"
    assert verdict(same_information=False, both_can_be_true=False) == "both_true"   # missing -> keep both


def test_screen_flags_only_same_item_conflicts():
    claim = {"text": "newer"}
    earlier = [NS(text="a"), NS(text="b"), NS(text="c")]
    answers = [dict(same_information=False, both_can_be_true=False, same_item=False),
               dict(same_information=False, both_can_be_true=False, same_item=True),
               dict(same_information=True, both_can_be_true=True, same_item=True)]
    assert facts.screen(claim, earlier, ask_many=lambda e, n: answers) == [False, True, True]
