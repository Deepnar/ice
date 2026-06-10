#!/usr/bin/env python3
"""Post‑simulation processing: run all background workers to maturity."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import time
from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory
from src.workers.reflection import run_reflection
from src.workers.clustering import cluster_turns
from src.workers.decay import apply_decay
from src.workers.sentinel_monitor import monitor_sentinels

def wait_for_task(task, timeout=120):
    """Block until the Celery task finishes (polling)."""
    start = time.time()
    while time.time() - start < timeout:
        if task.ready():
            return task.get(disable_sync_subtasks=False)
        time.sleep(2)
    raise TimeoutError(f"Task {task.id} did not finish within {timeout}s")

def main():
    # 1. Run clustering until no unassigned turns remain
    print("=== CLUSTERING ===")
    consecutive_fails = 0
    while True:
        db = SessionLocal()
        unassigned_count = db.query(EpisodicMemory).filter_by(cluster_id=None).count()
        db.close()
        if unassigned_count == 0:
            print("All turns assigned to clusters.")
            break
        print(f"  {unassigned_count} unassigned turns remaining – triggering clustering…")
        task = cluster_turns.delay()
        wait_for_task(task)
        # Check if any turns were actually assigned
        db = SessionLocal()
        new_unassigned = db.query(EpisodicMemory).filter_by(cluster_id=None).count()
        db.close()
        if new_unassigned == unassigned_count:
            consecutive_fails += 1
            if consecutive_fails >= 3:
                print("Clustering stalled after 3 attempts with no progress. Stopping.")
                break
        else:
            consecutive_fails = 0

    # 2. Run reflection (once for the most recent session)
    print("\n=== REFLECTION ===")
    task = run_reflection.delay()
    wait_for_task(task)
    print("Reflection complete.")

    # 3. Run decay (once, processes all old turns)
    print("\n=== DECAY ===")
    task = apply_decay.delay()
    wait_for_task(task)
    print("Decay applied.")

    # 4. Run sentinel (optional, just to validate)
    print("\n=== SENTINEL ===")
    task = monitor_sentinels.delay()
    wait_for_task(task)
    print("Sentinel check complete.")

    print("\nAll post‑simulation processing finished.")

if __name__ == "__main__":
    main()