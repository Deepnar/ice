"""Verified source conversation order is not a fabricated calendar date."""

import re

from src.memory.source import digest

METHOD = "ordered_conversation_v1"


def sequence_receipt(raw, sequence_sha256, position):
    if (not isinstance(sequence_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", sequence_sha256)
            or type(position) is not int or position < 1):
        raise ValueError("invalid source sequence identity/position")
    return dict(method=METHOD, sequence_sha256=sequence_sha256,
                position=position, raw_sha256=digest(raw))


def attach_sequence(spans, raw, receipt):
    """Only an explicit upstream receipt grants ordinal authority."""
    if receipt is None:
        return spans
    if (not isinstance(receipt, dict)
            or set(receipt) != {"method", "sequence_sha256", "position", "raw_sha256"}
            or receipt.get("method") != METHOD
            or receipt != sequence_receipt(raw, receipt.get("sequence_sha256"), receipt.get("position"))
            or spans.get("raw_sha256") != digest(raw)):
        raise ValueError("source sequence is not bound to the stored original")
    return {**spans, "sequence": dict(receipt)}


def verified_sequence(row):
    raw = getattr(row, "raw_text", None) or ""
    spans = getattr(row, "source_spans", None)
    if not isinstance(spans, dict) or spans.get("raw_sha256") != digest(raw):
        return None
    receipt = spans.get("sequence")
    if receipt is None:
        return None
    try:
        attach_sequence(spans, raw, receipt)
    except (TypeError, ValueError):
        return None
    return dict(receipt)


def source_order_stamp(row):
    receipt = verified_sequence(row)
    if receipt is None:
        return ""
    calendar = ("calendar recorded separately"
                if getattr(row, "ts_provenance", None) == "original" else "calendar unknown")
    return f"[Verified conversation order: {receipt['position']}; {calendar}] "


def compare_observations(old, new):
    """Choose comparable source units, never infer whether a claim replaces one.

    Every contributing observation must share a speaker and conversation. A
    calendar is authoritative only when the caller supplied an original aware
    timestamp. An ordinal is comparable only in one explicitly verified source
    sequence. Neither ingestion time nor the model's proposed event date counts.
    """
    if not old or len(new) != 1:
        return None, "missing_or_ambiguous_source"
    observations = old + new
    if len({u['conversation_id'] for u in observations}) != 1:
        return None, "different_conversations"
    if len({u['role'] for u in observations}) != 1:
        return None, "different_speakers"
    dates = [u.get('recorded_at') for u in observations]
    calendars = all(d is not None and d.tzinfo is not None and d.utcoffset() is not None
                    for d in dates)
    sequences = [u.get('sequence') for u in observations]
    shared_sequence = (all(sequences)
                       and len({s['sequence_sha256'] for s in sequences}) == 1)
    if not calendars and not shared_sequence:
        return None, "source_order_unknown"
    if shared_sequence:
        # Independently known calendars must agree with the ordinal, including
        # ties. Otherwise choosing either clock silently hides inconsistent input.
        dated = sorted((u for u in observations if u.get('recorded_at') is not None
                        and u['recorded_at'].tzinfo is not None
                        and u['recorded_at'].utcoffset() is not None),
                       key=lambda u: u['recorded_at'])
        for a, b in zip(dated, dated[1:]):
            da, db = a['recorded_at'], b['recorded_at']
            pa, pb = a['sequence']['position'], b['sequence']['position']
            if (da > db) - (da < db) != (pa > pb) - (pa < pb):
                return None, "calendar_sequence_disagreement"
    basis = "original_calendar" if calendars else "conversation_sequence"
    key = (lambda u: u['recorded_at']) if calendars else (lambda u: u['sequence']['position'])
    latest = max(key(u) for u in old)
    selected = [u for u in old if key(u) == latest]
    if len(selected) != 1:
        return None, "ambiguous_latest_source"
    if key(new[0]) <= latest:
        return None, "new_source_not_later"
    receipt = dict(basis=basis, conversation_id=new[0]['conversation_id'], role=new[0]['role'],
                   old_recorded_at=selected[0]['recorded_at'].isoformat()
                   if selected[0].get('recorded_at') is not None else None,
                   new_recorded_at=new[0]['recorded_at'].isoformat()
                   if new[0].get('recorded_at') is not None else None)
    if shared_sequence:
        receipt.update(sequence_sha256=sequences[0]['sequence_sha256'],
                       old_position=selected[0]['sequence']['position'],
                       new_position=new[0]['sequence']['position'])
    return (selected[0], {**new[0], 'ordering': receipt}), "eligible"
