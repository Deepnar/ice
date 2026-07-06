#!/usr/bin/env python3
"""Watch /codex_inject directory for YAML/JSON entity definition files
   and insert them directly as Codex events (bypassing LLM extraction)."""

import os, time, json, uuid, yaml, shutil
from datetime import datetime, timezone
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.api.db import SessionLocal
from src.memory.models import CodexEntity, CodexEdge, CodexEvent

WATCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../codex_inject'))
PROCESSED_DIR = os.path.join(WATCH_DIR, "processed")

class CodexInjectHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        filepath = event.src_path
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in ('.yaml', '.yml', '.json'):
            return
        self.process_file(filepath)

    def process_file(self, filepath):
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        db = SessionLocal()
        try:
            # Expected format: list of objects with keys: canonical_name, aliases?, tags?,
            #   type?, description? (or legacy context_payload), properties?,
            #   relations? (list of {target, relation, negated?})
            from src.workers.codex_extractor import _regenerate_context_payload  # A7: rich note assembly
            entities = data if isinstance(data, list) else [data]
            touched = set()   # entity ids whose note must be regenerated
            for ent in entities:
                name = ent.get("canonical_name")
                if not name:
                    continue
                # A7.1: manual descriptive text is the note BODY (description),
                # since context_payload is now auto-assembled. Accept legacy
                # `context_payload` as description for backward compatibility.
                desc = ent.get("description") or ent.get("context_payload", "")
                entity = db.query(CodexEntity).filter_by(canonical_name=name.lower().strip()).first()
                if not entity:
                    entity = CodexEntity(
                        id=uuid.uuid5(uuid.NAMESPACE_DNS, name.lower().strip()),
                        canonical_name=name.lower().strip(),
                        aliases=ent.get("aliases", []),
                        tags=ent.get("tags", []),
                        entity_type=(ent.get("type") or ent.get("entity_type") or "entity"),
                        description=desc,
                        properties=ent.get("properties", {}),
                        last_updated=datetime.now(timezone.utc)
                    )
                    db.add(entity)
                    db.flush()
                else:
                    if desc:
                        entity.description = desc
                    if ent.get("type") or ent.get("entity_type"):
                        entity.entity_type = ent.get("type") or ent.get("entity_type")
                    if ent.get("properties"):
                        entity.properties = {**(entity.properties or {}), **ent["properties"]}
                touched.add(entity.id)
                # Process relations
                relations = ent.get("relations", [])
                for rel in relations:
                    target_name = rel.get("target")
                    relation = rel.get("relation")
                    if not target_name or not relation:
                        continue
                    negated = bool(rel.get("negated", False))
                    target_entity = db.query(CodexEntity).filter_by(
                        canonical_name=target_name.lower().strip()
                    ).first()
                    if not target_entity:
                        target_entity = CodexEntity(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, target_name.lower().strip()),
                            canonical_name=target_name.lower().strip(),
                            last_updated=datetime.now(timezone.utc)
                        )
                        db.add(target_entity)
                        db.flush()
                    touched.add(target_entity.id)   # so backlinks render on the target too
                    # Check for existing edge (same polarity)
                    existing = db.query(CodexEdge).filter_by(
                        source_id=entity.id,
                        target_id=target_entity.id,
                        relation=relation,
                        negated=negated,
                        valid_until=None
                    ).first()
                    if not existing:
                        new_edge = CodexEdge(
                            id=uuid.uuid4(),
                            source_id=entity.id,
                            target_id=target_entity.id,
                            relation=relation,
                            strength=2.0,    # manual injection gives high confidence
                            source_batch=uuid.uuid4(),
                            confidence="active",
                            extraction_confidence=1.0,   # A3: manual = fully trusted
                            negated=negated,             # A8: polarity
                            valid_from=datetime.now(timezone.utc)
                        )
                        db.add(new_edge)
                        db.add(CodexEvent(
                            entity_id=entity.id,
                            event_type="edge_added",
                            payload={"manual_injection": True, "relation": relation,
                                     "target": target_name, "negated": negated},
                            batch_source=uuid.uuid4()
                        ))
            # A7.1: assemble the rich note (description + links + backlinks) and
            # infer entity_type for every entity touched, now that edges exist.
            db.flush()
            for eid in touched:
                e = db.query(CodexEntity).get(eid)
                if e:
                    _regenerate_context_payload(e, db)
            db.commit()
            print(f"Injected {filepath}")
        except Exception as e:
            db.rollback()
            print(f"Error processing {filepath}: {e}")
        finally:
            db.close()

        # Move to processed
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        shutil.move(filepath, os.path.join(PROCESSED_DIR, os.path.basename(filepath)))

def main():
    os.makedirs(WATCH_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    handler = CodexInjectHandler()
    observer = Observer()
    observer.schedule(handler, WATCH_DIR, recursive=False)
    observer.start()
    print(f"Codex Inject watcher watching {WATCH_DIR}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()