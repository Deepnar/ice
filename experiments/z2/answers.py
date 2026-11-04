"""Z2 S10: answers to every read-pass context (spec D6c, D7, D10, D11).

Every unit S9 wrote (one question under one condition) is answered from its exact
prompt messages; identical prompts are answered once (a gated-off question gives
the same prompt in several conditions). The answerer never sees the condition's name.

Answerer (D6d, 2026-10-27): Muse 1.2 (`muse-spark-1.2-contributor`) on OpenCode Go,
temperature 0, 16,000-token output cap (reasoning included), four workers; the plan's
spend limit is met by waiting, never by pacing to the request table (TRAPS 148). Muse 1.3
judges (D6).
`--answerer local` keeps D6c's path: `qwen3.8:27b` through Ollama, thinking on,
num_ctx 49,152, output cap = what the window leaves (16,000 down to 8,000; a prompt that
leaves less is `too_long`, never truncated). `--answerer luna` keeps D7's paced path.
One answers.jsonl never mixes answerers: a run whose model differs from the saved rows'
stops before calling anything.

Built for a long unattended run:
- every answer is a durable row in answers.jsonl keyed by the prompt's sha256;
  a rerun skips what is saved;
- an answer cut at the output cap is recorded `incomplete` — ungraded, never counted
  wrong (D11); any other error is retried on the next attempt and recorded as an
  error after three;
- cloud only: Luna's calls are paced to LUNA_PER_5H inside any rolling 5-hour window; a
  quota/rate-limit reply waits and retries, and a rejected key waits too, re-reading
  .env every few minutes.

    uv run --no-sync python experiments/z2/answers.py --units logs/z2-run-main/s9-B \
        --out logs/z2-run-main/s10-B --answerer muse12
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-answers-2"
OUTPUT_TOKENS = {"local": 16_000, "luna": 4_000, "muse12": 16_000}    # D6c local ceiling; D7 Luna; D6d
LOCAL_MIN_OUTPUT = 8_000                 # the window must leave at least this much
LUNA_PER_5H = 4_000                      # under the plan's 4,230 per 5 hours
WINDOW = 5 * 3600
WORKERS = {"local": 1, "luna": 4, "muse12": 4}    # Ollama serves one request at a time
MAX_ERRORS = 3
REFUSED = "content_policy_violation"   # the provider's content filter: a final outcome, not an error
AUTH_WAIT, QUOTA_WAIT = 300, 600

LOCAL_MODEL = "qwen3.8:27b"
LOCAL_NUM_CTX = 49_152                   # largest Z2 prompt ~38.4k Qwen tokens + 8k output
LOCAL_KEEP_ALIVE = "30m"
LOCAL_SEED = 0
FIT_MARGIN = 1.05                        # embedder tokenizer vs the answerer's chat template


class Incomplete(Exception):
    """The answer hit the output cap."""


class TooLong(Exception):
    """The prompt plus the output cap does not fit the answerer's window."""


def prompt_sha(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def wanted(conditions):
    """{condition: the one ask kind answered, or None for every kind}; "recent_only:control" answers
    that condition only for the no-memory controls (LoCoMo's no-memory comparison, FINAL revision 5)."""
    return {c.split(":")[0]: (c.split(":")[1] if ":" in c else None) for c in conditions or ()}


def load_units(units_dir, conditions=None, checkpoints=None):
    """[(condition, ask_id, messages, cache session or None)] for every S9 unit. A unit may name
    a `cache_session` (FINAL's full-history prompts share one per conversation, so the provider
    can reuse the cached transcript); otherwise each prompt has its own session. *checkpoints*
    (FINAL's streaming cloud runner): only units of those checkpoints, whole checkpoints only, so a
    cache session's prompts are always answered together."""
    out, kinds = [], wanted(conditions)
    for path in sorted(Path(units_dir).glob("units/*/*.json")):
        if conditions and path.parent.name not in kinds:
            continue                              # read only as another condition's prompt frame
        unit = json.loads(path.read_text())
        if checkpoints is not None and str(unit.get("checkpoint")) not in checkpoints:
            continue
        if kinds.get(path.parent.name) and unit["ask"].get("kind", "question") != kinds[path.parent.name]:
            continue
        out.append((unit["condition"], unit["ask"]["id"], unit["stage"]["prompt_messages"],
                    unit["stage"].get("cache_session")))
    return out


class MixedAnswerers(Exception):
    """The saved answers come from a different model than this run's."""


class Store:
    """answers.jsonl: one row per prompt; thread-safe appends."""

    def __init__(self, path):
        self.path, self.rows, self.lock = path, {}, threading.Lock()
        if path.exists():
            from experiments.z2.ledger_merge import complete_lines
            for line in complete_lines(path):           # a last line cut by a stop is skipped, answered again
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
            tmp = self.errors_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.errors, indent=1))
            tmp.replace(self.errors_path)


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


class LocalAnswerer:
    """`qwen3.8:27b` through Ollama's native chat API (D6c): thinking on, fixed seed,
    explicit num_ctx and keep_alive (ICE manages model memory itself, not the host
    default), released by `close()`."""

    def __init__(self, model=LOCAL_MODEL, num_ctx=LOCAL_NUM_CTX, host=None, post=None):
        from src.api.config import settings
        self.model, self.num_ctx = model, num_ctx
        self.host = (host or settings.ollama_base_url).rstrip("/")
        self._post, self._tok = post or self._http_post, None

    def _http_post(self, path, body, timeout=3600):
        import urllib.request
        request = urllib.request.Request(self.host + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def estimate_tokens(self, messages):
        if self._tok is None:
            from transformers import AutoTokenizer

            from src.api.config import settings
            self._tok = AutoTokenizer.from_pretrained(settings.embedding_model_name)
        return sum(len(self._tok(m["content"])["input_ids"]) + 8 for m in messages)

    def generate(self, messages, *, temperature, max_output_tokens, session_id=None):
        # Long thinking is likeliest where memory is thinnest (recent-only reads), so a
        # fixed cap would turn exactly those failures into "ungraded": the cap is what the
        # window leaves after the prompt, up to max_output_tokens, never below 8,000.
        estimate = self.estimate_tokens(messages)
        cap = min(max_output_tokens, self.num_ctx - int(estimate * FIT_MARGIN))
        if cap < LOCAL_MIN_OUTPUT:
            raise TooLong(f"~{estimate} prompt tokens leave {cap} < {LOCAL_MIN_OUTPUT} output in num_ctx {self.num_ctx}")
        out = self._post("/api/chat", dict(
            model=self.model, messages=messages, stream=False, think=True, keep_alive=LOCAL_KEEP_ALIVE,
            options=dict(num_ctx=self.num_ctx, temperature=temperature, seed=LOCAL_SEED,
                         num_predict=cap)))
        message = out.get("message") or {}
        usage = dict(prompt_tokens=out.get("prompt_eval_count"), output_tokens=out.get("eval_count"), output_cap=cap,
                     thinking_chars=len(message.get("thinking") or ""), done_reason=out.get("done_reason"),
                     prefill_s=round((out.get("prompt_eval_duration") or 0) / 1e9, 2),
                     generate_s=round((out.get("eval_duration") or 0) / 1e9, 2))
        if (usage["prompt_tokens"] or 0) + (usage["output_tokens"] or 0) >= self.num_ctx:
            raise TooLong(f"window filled: {usage}")              # Ollama would have shifted the prompt
        if out.get("done_reason") == "length":
            raise Incomplete(f"output cap {cap} reached: {usage}")
        return SimpleNamespace(text=(message.get("content") or "").strip(), usage=usage)

    def metadata(self):
        return dict(provider="ollama", model=self.model, digest=self.digest(), think=True,
                    num_ctx=self.num_ctx, seed=LOCAL_SEED, keep_alive=LOCAL_KEEP_ALIVE)

    def digest(self):
        import urllib.request
        with urllib.request.urlopen(self.host + "/api/tags", timeout=30) as response:
            models = json.load(response).get("models", [])
        return next((m.get("digest") for m in models if m.get("name") == self.model), None)

    def close(self):
        try:
            self._post("/api/generate", dict(model=self.model, keep_alive=0), timeout=120)
        except Exception as exc:                                     # releasing is best effort
            print(f"WARNING: could not unload {self.model}: {exc}", flush=True)


class Answerer:
    """The answer client, rebuildable: after a rejected key the provider variables are
    re-read from .env (load_selected_env never overrides values already set) and the
    client is built again, so a replaced key is used without restarting the run."""

    def __init__(self, role="luna"):
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


def answer_one(generate, messages, sha, *, say=print, sleep=time.sleep, reload_env=None,
               output_tokens=OUTPUT_TOKENS["local"], session=None):
    """One answer row; waits through auth and quota failures, returns on anything final."""
    from experiments.lme.cloud_provider import CloudCompletionError, ProviderAccessError
    while True:
        started = time.time()
        try:
            result = generate(messages, temperature=0, max_output_tokens=output_tokens,
                              session_id=session or f"ice-z2-answer-{sha[:24]}")
            return dict(prompt_sha256=sha, finish="complete", text=result.text, usage=result.usage,
                        seconds=round(time.time() - started, 2), wall=round(time.time(), 3))
        except (CloudCompletionError, Incomplete) as exc:
            return dict(prompt_sha256=sha, finish="incomplete", text="", detail=str(exc)[:300],
                        seconds=round(time.time() - started, 2), wall=round(time.time(), 3))
        except TooLong as exc:
            return dict(prompt_sha256=sha, finish="too_long", text="", detail=str(exc)[:300],
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
        except Exception as exc:
            # The provider's content filter refusing the prompt (Z2 C, 2026-10-28: one recent-only
            # prompt holding a pasted, garbled model output) is final and its own outcome: counted,
            # ungraded like an incomplete answer, never retried as an error. Anything else is raised.
            if REFUSED not in str(exc):
                raise
            return dict(prompt_sha256=sha, finish="refused", text="", detail=str(exc)[:300],
                        seconds=round(time.time() - started, 2), wall=round(time.time(), 3))


def run(units_dir, out, *, generate, meta, say=print, reload_env=None, workers=1, pace=False,
        output_tokens=OUTPUT_TOKENS["local"], conditions=None, checkpoints=None, reuse=()):
    from concurrent.futures import ThreadPoolExecutor
    out.mkdir(parents=True, exist_ok=True)
    store = Store(out / "answers.jsonl")
    saved = {(r.get("answerer") or {}).get("model") for r in store.rows.values()}
    if saved - {meta.get("model")}:
        raise MixedAnswerers(f"{out / 'answers.jsonl'} holds answers from {sorted(map(str, saved))}; "
                             f"this run answers with {meta.get('model')} — move the old file aside first")
    units = load_units(units_dir, conditions, checkpoints)
    index = {f"{c}|{a}": prompt_sha(m) for c, a, m, _ in units}
    if checkpoints is not None and (out / "index.json").exists():   # a partial pass keeps earlier entries
        index = {**json.loads((out / "index.json").read_text())["index"], **index}
    prompts = {prompt_sha(m): m for _, _, m, _ in units}
    sessions = {prompt_sha(m): s for _, _, m, s in units if s}
    reused = reuse_answers(store, prompts, meta, reuse)
    if reused:
        say(f"{reused} prompts identical to ones already answered with this answerer: reused, not asked again")
    todo = [s for s in prompts if not store.settled(s)]
    pacer = Pacer([r["wall"] for r in store.rows.values() if r.get("wall")]) if pace else None
    say(f"{len(units)} contexts, {len(prompts)} distinct prompts, {len(todo)} to answer")

    def work(sha):
        try:
            if pacer:
                pacer.take(say)
            row = answer_one(generate, prompts[sha], sha, say=say, reload_env=reload_env,
                             output_tokens=output_tokens, session=sessions.get(sha))
            store.add(dict(row, policy=POLICY, answerer=meta))
        except Exception as exc:                  # this prompt only; retried next attempt
            store.failed(sha, exc)

    # Prompts sharing a cache session (FINAL: long context per checkpoint, full history per
    # conversation) run in order on one worker, groups side by side: every worker stays busy and
    # each prompt after a group's first finds the shared start already in the provider's cache.
    groups = {}
    for sha in todo:
        groups.setdefault(sessions.get(sha) or sha, []).append(sha)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda members: [work(sha) for sha in members], groups.values()))
    (out / "index.json").write_text(json.dumps(dict(policy=POLICY, index=index), indent=1))
    pending = [s for s in prompts if not store.settled(s)]
    status = dict(contexts=len(units), prompts=len(prompts), answered=sum(1 for s in prompts if s in store.rows),
                  incomplete=sum(1 for s in prompts if store.rows.get(s, {}).get("finish") == "incomplete"),
                  too_long=sum(1 for s in prompts if store.rows.get(s, {}).get("finish") == "too_long"),
                  refused=sum(1 for s in prompts if store.rows.get(s, {}).get("finish") == "refused"),
                  errors_given_up=sum(1 for s in prompts if s not in store.rows and store.settled(s)),
                  pending=len(pending), finished=not pending)
    (out / "status.json").write_text(json.dumps(status, indent=1))
    return status


def reuse_answers(store, prompts, meta, sources):
    """Copy, from earlier answer folders, the complete answers to prompts identical to this run's,
    given by the same answerer at the same output cap (R7: no memory and plain search often build
    the same prompt Z2 already answered). Each copied row records its source folder."""
    from experiments.z2.ledger_merge import complete_lines
    n = 0
    for source in sources or ():
        path = Path(source) / "answers.jsonl"
        if not path.exists():
            continue
        for line in complete_lines(path):
            row = json.loads(line)
            sha = row["prompt_sha256"]
            if sha in prompts and sha not in store.rows and row.get("finish") == "complete" and row.get("answerer") == meta:
                store.add(dict(row, reused_from=str(source)))
                n += 1
    return n


_PROMPTS = {}


def prompt_set(units_dir, conditions=None):
    """Distinct prompt hashes of an S9 directory, re-read only when its unit files change (the
    runner's meter asks every few seconds; S9 B alone is ~7 GB of unit JSON). Kept on disk too
    (<units_dir>/prompt-set[-<conditions>].json, keyed by the files' count and newest time): every
    fresh `status` and every runner start re-read all ~7 GB of B to draw a finished stage's meter
    (2026-10-28: a RAM spike that lagged the laptop)."""
    paths = sorted(p for p in Path(units_dir).glob("units/*/*.json") if not conditions or p.parent.name in wanted(conditions)) \
        if Path(units_dir).exists() else []
    key = [len(paths), max((p.stat().st_mtime for p in paths), default=0)]
    tag = "-" + hashlib.sha256(json.dumps(sorted(conditions)).encode()).hexdigest()[:8] if conditions else ""
    cache_key = f"{units_dir}|{tag}"
    cached = _PROMPTS.get(cache_key)
    if cached is None or cached[0] != key:
        disk = Path(units_dir) / f"prompt-set{tag}.json"
        saved = json.loads(disk.read_text()) if disk.exists() else None
        if saved and saved.get("key") == key:
            cached = (key, set(saved["prompts"]))
        else:
            cached = (key, {prompt_sha(m) for _, _, m, _ in load_units(units_dir, conditions)})
            if paths:
                disk.write_text(json.dumps(dict(key=key, prompts=sorted(cached[1]))) + "\n")
        _PROMPTS[cache_key] = cached
    return cached[1]


def progress(units_dir, out, conditions=None):
    prompts = prompt_set(units_dir, conditions)
    path = Path(out) / "answers.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["prompt_sha256"] for line in path.read_text().splitlines() if line.strip()}
    return len(prompts & done), len(prompts) or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--units", type=Path, required=True, help="an S9 output directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", help="answer only these conditions (comma-separated); others are prompt frames")
    parser.add_argument("--checkpoints", help="answer only these whole checkpoints (comma-separated; FINAL streaming)")
    parser.add_argument("--reuse", action="append", type=Path, default=[],
                        help="an earlier answers folder: identical prompts, same answerer, reused (repeatable)")
    parser.add_argument("--answerer", default="muse12",
                        help="muse12 (D6d, default) | local | luna | cloud:<role> (FINAL's cloud-answerer subset)")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT / "logs"):
        raise ValueError("answers are private: keep them under logs/")
    from experiments.z2.ledger_merge import one_writer
    writer = one_writer(args.out)  # noqa: F841 — held until exit
    if args.answerer != "local":
        from experiments.z2.profiles import metadata
        role = args.answerer.split(":", 1)[1] if args.answerer.startswith("cloud:") else args.answerer
        answerer = Answerer(role)
        cap, workers = OUTPUT_TOKENS.get(role, OUTPUT_TOKENS["muse12"]), WORKERS.get(role, WORKERS["muse12"])
        status = run(args.units, args.out, generate=answerer.generate, meta=dict(metadata(role), output_tokens=cap),
                     reload_env=answerer.refresh, workers=workers, pace=role == "luna", output_tokens=cap,
                     conditions=args.conditions.split(",") if args.conditions else None,
                     checkpoints=set(args.checkpoints.split(",")) if args.checkpoints else None, reuse=args.reuse)
    else:
        answerer = LocalAnswerer()
        try:
            status = run(args.units, args.out, generate=answerer.generate,
                         meta=dict(answerer.metadata(), output_tokens=OUTPUT_TOKENS["local"]),
                         workers=WORKERS["local"], output_tokens=OUTPUT_TOKENS["local"])
        finally:
            answerer.close()
    print(json.dumps(status))
    return 0 if status["finished"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
