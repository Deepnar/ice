"""Closed, read-only compatibility for already completed v3 answer artifacts."""
import hashlib
import json
from pathlib import Path

from scripts.z1.cloud_recovery import terminal_answer_fault

REGISTRY = Path(__file__).with_name("answer_resume_repairs.json")


def registered_runner_transition(before, after):
    registry = json.loads(REGISTRY.read_text())
    if registry.get("format") != "ice-v3-completed-answer-repairs-1":
        raise ValueError("unsupported completed-answer repair registry")
    return _reachable(registry, before, after)


def _reachable(registry, before, after):
    seen, frontier = set(), [before]
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        for row in registry['repairs']:
            if row['before'] == node:
                if row['after'] == after:
                    return True
                frontier.append(row['after'])
    return False


def _compatible_components(old, new, changed):
    """Exact inspected telemetry revisions, never arbitrary adapter changes."""
    raw = REGISTRY.read_bytes()
    registry = json.loads(raw)
    for row in registry['repairs']:
        components = row.get('read_only_components', {})
        if not changed <= components.keys():
            continue
        if not (old.get('answer_runner') == row['before']
                or _reachable(registry, old.get('answer_runner'), row['before'])):
            continue
        if not (new['answer_runner'] == row['after']
                or _reachable(registry, row['after'], new['answer_runner'])):
            continue
        if all(old[k] == components[k]['before'] and new[k] == (
            hashlib.sha256(raw).hexdigest() if components[k]['after'] == 'current_registry_sha256'
            else components[k]['after']) for k in changed):
            return True
    return False


def completed_identity_matches(saved, current):
    """Never authorize calls or changed inputs under an older runner identity."""
    if saved.get("version") != "v3" or not (saved.get("complete") or saved.get("processing_complete")):
        return False
    if any(saved.get(k) != v for k, v in current.items() if k != "implementation_sha256"):
        return False
    old, new = saved.get("implementation_sha256", {}), current["implementation_sha256"]
    added = {"completed_resume", "completed_resume_registry"}
    changed = {k for k in old if k != 'answer_runner' and old[k] != new.get(k)}
    if (set(old) - set(new) or set(new) - set(old) - added
            or changed and not _compatible_components(old, new, changed)
            or not registered_runner_transition(old.get("answer_runner"), new["answer_runner"])):
        return False
    records = saved.get("records", [])
    ids = [r.get("probe_id") for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(current["probe_ids"]):
        return False
    return all((r.get("answer") and not r.get("error")) or
               (current["cloud_failure_policy"] == "continue" and terminal_answer_fault(r))
               for r in records)
