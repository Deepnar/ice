#!/usr/bin/env python3
"""Drop Zone – watches a directory and safely ingests files into ICE memory."""

import hashlib
import os
import shutil
import time
from datetime import datetime, timezone
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import RAGDocument, RAGChunk
from src.classifier.classifier import PyTorchClassifier

WATCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ingest_inbox'))
PROCESSED_DIR = os.path.join(WATCH_DIR, 'processed')

# G1: single source of truth for the classifier path (was hardcoded v2_final).
classifier = PyTorchClassifier(
    model_path=settings.classifier_model_path,
    schema_path=settings.label_schema_path,
)


def wait_for_file_to_settle(filepath: str, check_interval: float = 0.5, timeout: float = 10.0) -> bool:
    """Waits until a file's size stops changing, ensuring the OS has finished writing it."""
    start_time = time.time()
    previous_size = -1
    while time.time() - start_time < timeout:
        try:
            current_size = os.path.getsize(filepath)
            if current_size == previous_size and current_size > 0:
                return True
            previous_size = current_size
        except OSError:
            pass  # File might be temporarily locked
        time.sleep(check_interval)
    return False


class IngestHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        filepath = event.src_path
        if not os.path.isfile(filepath):
            return
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in ('.txt', '.jsonl', '.md'):
            return
        print(f"Waiting for OS to release {filepath}...")
        if not wait_for_file_to_settle(filepath):
            print(f"Timeout waiting for {filepath} to settle. Skipping.")
            return
        print(f"Processing {filepath}...")
        self.ingest_file(filepath)
    
    def on_moved(self, event):
        """Catch files that are moved into the watched directory."""
        if event.dest_path.startswith(WATCH_DIR):
            # Treat the moved-in file as a new file
            self.on_created(type('FakeEvent', (object,), {
                'is_directory': False,
                'src_path': event.dest_path
            })())

    def ingest_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        db = SessionLocal()
        try:
            doc = RAGDocument(
                filename=os.path.basename(filepath),
                file_type=os.path.splitext(filepath)[1].lower(),
                token_count=int(len(content.split()) * 1.33)
            )
            db.add(doc)
            db.flush()

            chunk_size = 512  # words
            words = content.split()
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i+chunk_size]
                chunk_text = ' '.join(chunk_words)
                embedding = classifier.embedder.encode(chunk_text, convert_to_tensor=False).tolist()
                chunk = RAGChunk(
                    document_id=doc.id,
                    chunk_index=i // chunk_size,
                    chunk_text=chunk_text,
                    embedding=embedding
                )
                db.add(chunk)

            db.commit()
            print(f"  Ingested as RAG document {doc.id}")

        except Exception as e:
            db.rollback()
            print(f"  Error ingesting {filepath}: {e}")
        finally:
            db.close()

        # Move processed file using shutil (safe across filesystems)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        dest = os.path.join(PROCESSED_DIR, os.path.basename(filepath))
        shutil.move(filepath, dest)


def main():
    os.makedirs(WATCH_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    event_handler = IngestHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    print(f"Drop Zone watching {WATCH_DIR}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()