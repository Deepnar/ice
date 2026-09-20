"""Rolling-summary cache freshness, distinct from semantic source support."""
from datetime import datetime

import structlog
from sqlalchemy import text

from src.memory.source import digest

logger = structlog.get_logger('ice.memory.summary_snapshot')


def source_snapshot(db, conversation_id):
    """Transfer source identities/fingerprints, never full raw text to readers."""
    rows = db.execute(text('''
        SELECT e.id::text AS id, e.timestamp,
               md5(jsonb_build_array(e.batch_id, e.raw_text, e.source_spans,
                   e.timestamp, e.ts_provenance, e.is_private,
                   e.summary_text, e.inject_raw, e.summary_coverage)::text) AS fingerprint
        FROM episodic_memory e
        WHERE e.conversation_id = :cid
        ORDER BY e.timestamp, e.id
    '''), {'cid': str(conversation_id)}).fetchall()
    return [{'id': row.id, 'fingerprint': row.fingerprint,
             'timestamp': row.timestamp.isoformat() if row.timestamp else None}
            for row in rows]


def bind_snapshot(sources, summary):
    return {'version': 1, 'summary_sha256': digest(summary), 'sources': sources}


def snapshot_matches(record, summary, current, *, allow_newer=False):
    if not isinstance(record, dict) or record.get('version') != 1:
        return False
    if record.get('summary_sha256') != digest(summary or ''):
        return False
    saved = record.get('sources')
    if not isinstance(saved, list) or not saved:
        return False
    if any(not isinstance(item, dict) or not item.get('id') for item in saved):
        return False
    old = {item['id']: item for item in saved}
    now = {item['id']: item for item in current}
    if len(old) != len(saved) or any(now.get(key) != item for key, item in old.items()):
        return False
    extra = [item for item in current if item['id'] not in old]
    if not extra:
        return True
    if not allow_newer:
        return False
    try:
        latest = max(datetime.fromisoformat(item['timestamp']) for item in saved)
        return all(datetime.fromisoformat(item['timestamp']) > latest for item in extra)
    except (TypeError, ValueError, KeyError):
        return False


def summary_snapshot_readable(db, row):
    current = source_snapshot(db, row.conversation_id)
    matches = snapshot_matches(row.source_manifest, row.summary_text, current,
                               allow_newer=True)
    if not matches:
        logger.warning('conversation_summary_source_stale',
                       conversation_id=str(row.conversation_id),
                       reason='missing, changed or backfilled source snapshot; rebuild required')
    return matches
