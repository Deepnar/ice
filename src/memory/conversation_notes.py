"""Read-only contract for indexed, independently sourced conversation notes."""
from src.memory.tokens import count as count_tokens


def part_batches(part, manifest):
    batches = {item['id']: item['batch_id'] for item in manifest['sources']}
    return [batches[sid] for sid in part['source_ids']]


def note_matches(note, part, manifest, ordinal):
    try:
        return (note.ordinal == ordinal and note.text == part['text']
                and note.mode == part['mode']
                and note.recorded_range == part.get('recorded_range')
                and note.source_ids == part['source_ids']
                and note.batch_ids == part_batches(part, manifest))
    except (KeyError, TypeError):
        return False


def indexed_parts(manifest, notes):
    parts = manifest.get('parts') or []
    return (len(parts) == len(notes)
            and all(note_matches(note, parts[i], manifest, i + 1)
                    for i, note in enumerate(sorted(notes, key=lambda n: n.ordinal))))


def render_note(note):
    return (f"[Original-source segment {note.ordinal}; {note.mode} evidence; "
            f"recorded range: {note.recorded_range or 'unknown'}]\n{note.text}")


def select_note_options(notes, max_tokens, scores=None):
    """Whole-part prompt alternatives, from fullest fit to best single part."""
    if max_tokens <= 0:
        return []
    ranked = sorted(notes, key=lambda n: (
        (scores or {}).get(n.ordinal, float('-inf')), n.ordinal), reverse=True)
    picked = []
    for note in ranked:
        proposed = picked + [note]
        rendered = '\n\n'.join(render_note(n) for n in sorted(proposed, key=lambda n: n.ordinal))
        if count_tokens(rendered) > max_tokens:
            continue
        picked = proposed
    return ['\n\n'.join(render_note(n) for n in sorted(picked[:end], key=lambda n: n.ordinal))
            for end in range(len(picked), 0, -1)]
