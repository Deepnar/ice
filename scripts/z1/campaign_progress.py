"""Read-only terminal progress for the ICE v3 manual campaign.

Artifacts remain the authority. This observer never calls a model, changes a
checkpoint, or writes a replay event. A processed turn is not yet durable.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Observation:
    done: int | None
    total: int | None
    detail: str


class ArtifactProgress:
    def __init__(self, root: Path, stage: str, *, arm: str = "",
                 turns: int = 0, probes: int = 0, admitted: int = 0):
        self.root, self.stage, self.arm = root, stage, arm
        self.turns, self.probes, self.admitted = turns, probes, admitted
        self.offset = 0
        self.inode = None
        self.tail = b""
        self.completed = {}
        self.probe_ids = set()
        self.phase = "initializing / waiting for saved work"
        self.cache = {}

    def json_file(self, path: Path):
        if not path.exists():
            return None
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        if path not in self.cache or self.cache[path][0] != stamp:
            value = json.loads(path.read_text())
            self.cache[path] = (stamp, value)
        return self.cache[path][1]

    def read_seed(self):
        path = self.root / "seed.jsonl"
        if not path.exists():
            return
        stat = path.stat()
        with path.open("rb") as source:
            source.seek(max(0, self.offset - len(self.tail)))
            unchanged = source.read(len(self.tail)) == self.tail
            if stat.st_ino != self.inode or stat.st_size < self.offset or not unchanged:
                self.offset = 0
                self.completed = {}
                self.probe_ids = set()
                self.phase = "initializing / waiting for saved work"
            self.inode = stat.st_ino
            source.seek(self.offset)
            while True:
                start = source.tell()
                line = source.readline()
                if not line.endswith(b"\n"):
                    source.seek(start)  # Wait for a complete JSONL event.
                    break
                row = json.loads(line)
                event = row.get("event")
                location = f"{row.get('conversation', '')} turn {row.get('turn', '')}"
                if event == "resume":
                    self.completed = dict(row["completed_turns"])
                    self.phase = "restored durable checkpoint"
                elif event == "clock":
                    slug = row["conversation"]
                    self.completed[slug] = max(self.completed.get(slug, 0), row["turn"])
                    self.phase = location + " processed"
                elif event == "complete":
                    self.completed = dict(row["turns_by_conversation"])
                    self.phase = "replay finalized"
                elif event == "as_of_probe":
                    self.probe_ids.add(row["probe_id"])
                    self.phase = "checkpoint probes"
                elif event == "maintenance":
                    self.phase = "last saved job: " + row["job"]
                elif event in {"turn", "written"}:
                    self.phase = location + (" preflight saved" if event == "turn" else " write saved")
                elif event == "failed":
                    self.phase = location + " FAILED at " + row["stage"]
            self.offset = source.tell()
            source.seek(max(0, self.offset - 256))
            self.tail = source.read(self.offset - source.tell())

    @staticmethod
    def judge_orders(data: dict) -> int:
        # A successful first order is saved before the reversed order is called.
        # Retried/failed rows may overlap results; count each probe/order once.
        orders = set()
        for row in [*data.get("failed_attempts", []), *data.get("results", [])]:
            for index, verdict in enumerate(row.get("order_verdicts") or []):
                if (verdict.get("raw") or {}).get("verdict") in {"A", "B", "TIE", "UNCERTAIN"}:
                    orders.add((row["probe_id"], index))
        pending = data.get("pending_probe")
        if pending and pending["first_order"].get("verdict") in {"A", "B", "TIE", "UNCERTAIN"}:
            orders.add((pending["probe_id"], 0))
        return len(orders)

    def observe(self) -> Observation:
        if self.stage == "seed":
            self.read_seed()
            durable = 0
            pointer = self.root / "seed.recovery" / "current.json"
            current = self.json_file(pointer)
            if current:
                generation = current["generation"]
                if Path(generation).name != generation:
                    raise ValueError("invalid checkpoint generation")
                manifest = self.json_file(pointer.parent / generation / "recovery.json")
                if manifest is None:
                    raise ValueError("checkpoint manifest unavailable")
                durable = sum(manifest["state"]["completed"].values())
            detail = (f"durable {durable} | probes {len(self.probe_ids)}/{self.probes}"
                      f" | {self.phase}")
            return Observation(sum(self.completed.values()), self.turns, detail)
        if self.stage == "answers":
            data = self.json_file(self.root / f"answers-{self.arm}.json") or {}
            successful = {r["probe_id"] for r in data.get("records", [])
                          if r.get("answer") and not r.get("error")}
            return Observation(len(successful), self.admitted, "successful saved answers")
        if self.stage == "judge":
            final = self.root / f"judge-full-vs-{self.arm}.json"
            candidates = [p for p in (final, final.with_suffix(".partial.json")) if p.exists()]
            path = max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else final
            data = self.json_file(path) or {}
            return Observation(self.judge_orders(data), self.admitted * 2,
                               "successful saved judge orders (two per pair)")
        return Observation(None, None, "working; this stage has no fractional counter")


class TerminalProgress:
    """Keep subprocess.run's normal interruption semantics; observe in a thread."""
    def __init__(self, reader: ArtifactProgress, label: str, *, stream=None, interval=1.0):
        self.reader, self.label = reader, label
        self.stream = stream if stream is not None else sys.stdout
        self.interval = interval
        self.tty = self.stream.isatty()
        self.stopped = threading.Event()
        self.started = time.monotonic()
        self.last = None
        self.last_print = 0.0
        self.thread = None

    def render(self, *, outcome=""):
        try:
            value = self.reader.observe()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Display failure is visible but cannot change the replay's outcome.
            value = Observation(None, None, f"progress unavailable ({type(exc).__name__}); see log")
        now = time.monotonic()
        elapsed = int(now - self.started)
        if value.total and value.done is not None:
            filled = min(20, int(20 * value.done / value.total))
            bar = "[" + "#" * filled + "-" * (20 - filled) + "]"
            counter = f" {value.done}/{value.total}"
        else:
            bar, counter = "[working]", ""
        message = (f"v3 {self.label} {bar}{counter} | {value.detail}"
                   f" | {elapsed // 60:02d}:{elapsed % 60:02d}"
                   + (f" | {outcome}" if outcome else ""))
        if self.tty:
            self.stream.write("\r\033[2K" + message + ("\n" if outcome else ""))
        elif value != self.last or outcome or now - self.last_print >= 30:
            self.stream.write(message + "\n")
        else:
            return
        self.stream.flush()
        self.last, self.last_print = value, now

    def _watch(self):
        while not self.stopped.wait(self.interval):
            self.render()

    def __enter__(self):
        self.render()
        self.thread = threading.Thread(target=self._watch, name="v3-progress", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, _exc, _tb):
        self.stopped.set()
        self.thread.join()
        self.render(outcome="completed" if exc_type is None else "stopped; state retained")
