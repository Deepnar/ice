"""Z2 S10: answers to every read-pass context with Luna (spec D7, D10, D11).

Every unit S9 wrote (one question under one condition) is answered from its exact
prompt messages; identical prompts are answered once (a gated-off question gives
the same prompt in several conditions). Answer output cap 4,000 tokens (D7);
the answerer never sees the condition's name.

Built for a long unattended run:
- every answer is a durable row in answers.jsonl keyed by the prompt's sha256;
  a rerun skips what is saved;
- Luna allows ~4,230 calls per 5 hours: calls are paced to LUNA_PER_5H inside any
  rolling 5-hour window, and a quota/rate-limit reply waits and retries;
- a rejected API key waits too, re-reading .env every few minutes, so replacing
  the key continues the run without a restart;
- an answer the provider returns incomplete is recorded as such — ungraded,
  never counted wrong (D11); any other error is retried on the next attempt and
  recorded as an error after three.

    uv run --no-sync python experiments/z2/answers.py --units logs/z2-run-main/s9-B \
        --out logs/z2-run-main/s10-B
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-answers-1"
ANSWERER = "luna"
OUTPUT_TOKENS = 4_000                    # D7
LUNA_PER_5H = 4_000                      # under the plan's 4,230 per 5 hours
WINDOW = 5 * 3600
WORKERS = 4
MAX_ERRORS = 3
AUTH_WAIT, QUOTA_WAIT = 300, 600


def prompt_sha(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_units(units_dir):
    """[(condition, ask_id, messages)] for every S9 unit."""
    out = []
    for path in sorted(Path(units_dir).glob("units/*/*.json")):
        unit = json.loads(path.read_text())
        out.append((unit["condition"], unit["ask"]["id"], unit["stage"]["prompt_messages"]))
    return out


class Store:
    """answers.jsonl: one row per prompt; thread-safe appends."""

    def __init__(self, path):
        self.path, self.rows, self.lock = path, {}, threading.Lock()
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["prompt_sha256"]] = row
        self.errors_path = path.with_name("answer-errors.json")
        self.errors = json.loads(self.errors_path.read_text()) if self.errors_path.exists() else {}

    def settled(self, sha):
        return sha in self.rows or self.errors.get(sha, {}).get("n", 0) >= MAX_ERRORS

    def add(self, row):
        with self.lock:
            with self.path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.rows[row["prompt_sha256"]] = row

    def failed(self, sha, exc):
        with self.lock:
            entry = self.errors.setdefault(sha, dict(n=0))
            entry.update(n=entry["n"] + 1, last=f"{type(exc).__name__}: {exc}"[:500])
            self.errors_path.write_text(json.dumps(self.errors, indent=1))


class Pacer:
    """At most `limit` calls in any rolling `window` seconds (counted from saved rows too)."""

    def __init__(self, walls, limit=LUNA_PER_5H, window=WINDOW, sleep=time.sleep, clock=time.time):
        self.walls, self.limit, self.window = sorted(walls), limit, window
        self.sleep, self.clock, self.lock = sleep, clock, threading.Lock()

    def take(self, say=print):
        while True:
            with self.lock:
                now = self.clock()
                self.walls = [w for w in self.walls if w > now - self.window]
                if len(self.walls) < self.limit:
                    self.walls.append(now)
                    return
                wait = self.walls[0] + self.window - now
            say(f"pacing: {self.limit} answers in the last 5 h; next slot in {int(wait)} s")
            self.sleep(min(60, max(1, wait)))


class Answerer:
    """The answer client, rebuildable: after a rejected key the provider variables are
    re-read from .env (load_selected_env never overrides values already set) and the
    client is built again, so a replaced key is used without restarting the run."""

    def __init__(self, role=ANSWERER):
        from experiments.lme.cloud_provider import load_selected_env
        from experiments.z2.profiles import generator
        load_selected_env()
        self.role, self._gen, self.lock = role, generator(role), threading.Lock()

    def generate(self, *args, **kwargs):
        return self._gen.generate(*args, **kwargs)

    def refresh(self):
        import os
        from experiments.lme.cloud_provider import MAIN_ENV
        from experiments.z2.profiles import Z2_PROFILES, generator
        profile = Z2_PROFILES.get(self.role)
        names = {n for n in (getattr(profile, "api_key_env", None), getattr(profile, "base_url_env", None)) if n}
        for raw in Path(MAIN_ENV).read_text().splitlines():
            key, _, value = raw.strip().partition("=")
            if key in names and value:
                os.environ[key] = value.strip().strip('"').strip("'")
        with self.lock:
            self._gen = generator(self.role)


def answer_one(generate, messages, sha, *, say=print, sleep=time.sleep, reload_env=None):
    """One answer row; waits through auth and quota failures, returns on anything final."""
    from experiments.lme.cloud_provider import CloudCompletionError, ProviderAccessError
    while True:
        started = time.time()
        try:
            result = generate(messages, temperature=0, max_output_tokens=OUTPUT_TOKENS,
                              session_id=f"ice-z2-answer-{sha[:24]}")
            return dict(prompt_sha256=sha, finish="complete", text=result.text, usage=result.usage,
                        seconds=round(time.time() - started, 2), wall=round(time.time(), 3))
        except CloudCompletionError as exc:
            return dict(prompt_sha256=sha, finish="incomplete", text="", detail=str(exc)[:300],
                        seconds=round(time.time() - started, 2), wall=round(time.time(), 3))
        except ProviderAccessError as exc:
            if exc.kind == "authentication":
                say(f"answerer rejected the API key ({exc.status_code}); update PROBE_API_KEY in .env — "
                    f"retrying in {AUTH_WAIT} s")
                sleep(AUTH_WAIT)
                if reload_env:
                    reload_env()
            else:
                # 5-hour, weekly or monthly limit: wait, and re-read the key each time, so a
                # key swapped into .env (another account) continues the run without a restart.
                say(f"answerer quota/rate limit ({exc.status_code}); waiting {QUOTA_WAIT} s — to continue "
                    f"sooner, put another PROBE_API_KEY in .env (picked up on the next retry)")
                sleep(QUOTA_WAIT)
                if reload_env:
                    reload_env()


def run(units_dir, out, *, generate, meta, say=print, reload_env=None, workers=WORKERS):
    from concurrent.futures import ThreadPoolExecutor
    out.mkdir(parents=True, exist_ok=True)
    store = Store(out / "answers.jsonl")
    units = load_units(units_dir)
    index = {f"{c}|{a}": prompt_sha(m) for c, a, m in units}
    prompts = {prompt_sha(m): m for _, _, m in units}
    todo = [s for s in prompts if not store.settled(s)]
    pacer = Pacer([r["wall"] for r in store.rows.values() if r.get("wall")])
    say(f"{len(units)} contexts, {len(prompts)} distinct prompts, {len(todo)} to answer")

    def work(sha):
        try:
            pacer.take(say)
            row = answer_one(generate, prompts[sha], sha, say=say, reload_env=reload_env)
            store.add(dict(row, policy=POLICY, answerer=meta))
        except Exception as exc:                  # this prompt only; retried next attempt
            store.failed(sha, exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))
    (out / "index.json").write_text(json.dumps(dict(policy=POLICY, index=index), indent=1))
    pending = [s for s in prompts if not store.settled(s)]
    status = dict(contexts=len(units), prompts=len(prompts), answered=sum(1 for s in prompts if s in store.rows),
                  incomplete=sum(1 for s in prompts if store.rows.get(s, {}).get("finish") == "incomplete"),
                  errors_given_up=sum(1 for s in prompts if s not in store.rows and store.settled(s)),
                  pending=len(pending), finished=not pending)
    (out / "status.json").write_text(json.dumps(status, indent=1))
    return status


def progress(units_dir, out):
    units = load_units(units_dir) if Path(units_dir).exists() else []
    prompts = {prompt_sha(m) for _, _, m in units}
    path = Path(out) / "answers.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["prompt_sha256"] for line in path.read_text().splitlines() if line.strip()}
    return len(prompts & done), len(prompts) or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--units", type=Path, required=True, help="an S9 output directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("answers are private: keep them under logs/")
    from experiments.z2.profiles import metadata
    answerer = Answerer()
    status = run(args.units, args.out, generate=answerer.generate, meta=metadata(ANSWERER),
                 reload_env=answerer.refresh)
    print(json.dumps(status))
    return 0 if status["finished"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
