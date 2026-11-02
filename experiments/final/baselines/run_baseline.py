"""FINAL's published memory systems on LoCoMo (Z3 D11, D13; pre-registration r2 §4; LoCoMo only
since 2026-10-28 — the benchmark all three report on): Graphiti, Mem0 and A-Mem, open source, run
locally, so only the memory system differs from ICE.

Runs in this folder's own environment (isolated from ICE's). A fresh store per LoCoMo
conversation; every session ingested in date order through the system's own write path, with
FINAL's background model (gemma4:e4b via Ollama, temperature 0) doing its LLM work and the
system's own documented local embedder (Graphiti, Mem0: nomic-embed-text via Ollama; A-Mem:
all-MiniLM-L6-v2 on CPU). LoCoMo is two named people talking; each system gets them as its own
LoCoMo evaluation did: Graphiti and A-Mem read the messages with each speaker's name in the text;
Mem0 keeps one memory per speaker, that speaker's messages as the "user" (it remembers facts
about its user), and each question searches both. Per question the system's own search returns
its top MEMORIES results (Mem0: per speaker, each labelled with whose memory it is), placed in
exactly the evidence position of ICE's bare-vector condition (ICE's recent_only envelope for
the same question) and written as a read unit that experiments/z2/answers.py and grade.py take
unchanged (answered by FINAL's answerer).

Two phases (maintainer, 2026-10-31), so the slow part runs before ICE is frozen:
- **search** — ingest + every question searched; nothing of ICE is read. Each question's hits
  are saved on their own (hits/<ask>.json) as soon as they are found.
- **wrap** — minutes, after FINAL's LoCoMo read: the saved hits placed in ICE's recent_only
  envelope for the same question (the frame the freeze may still change).

Stopping is safe at any moment. A store is written only while it ingests: a conversation stopped
mid-ingest is wiped and ingested again from its first session, never twice into one store. Once
ingested (ingested.json), a stop during search keeps the store and the hits found so far for the
systems whose store lives on disk (Graphiti, Mem0); A-Mem's store lives in memory, so its
conversation is ingested again. A finished conversation (done.json) is never touched again.

    cd experiments/final/baselines
    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --no-sync python run_baseline.py search --system mem0 \
        --history LOCOMO-conv-26 --out <search dir>
    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --no-sync python run_baseline.py wrap --system mem0 \
        --history LOCOMO-conv-26 --search <search dir> --ice-units <its local-read folder> --out <dir>
"""

import argparse
import asyncio
import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLLAMA = "http://localhost:11434"
LLM_MODEL, EMBED_MODEL = "gemma4:e4b", "nomic-embed-text"
MEMORIES = 10
POLICY = "ice-final-baseline-1"
# The evidence pair of ICE's prompt assembler (experiments/z2/read_passes.py CONTEXT_HEADER /
# CONTEXT_ACK) — kept identical; tests/smoke/test_final_baselines.py checks it.
CONTEXT_HEADER = "=== RETRIEVED CONTEXT ==="
CONTEXT_ACK = "Understood — I have the background context. What would you like to know?"


def session_text(msgs):
    """The session as Graphiti reads it: each message already names its speaker."""
    return "\n".join(m["content"] for m in msgs)


def from_side(msgs, speaker):
    """Mem0's view for one speaker: their messages as the user's, the other person's as replies."""
    return [dict(role="user" if m["content"].startswith(f"{speaker}: ") else "assistant", content=m["content"])
            for m in msgs]


def envelope(recent_unit, memories, question):
    """ICE's recent_only envelope with the baseline's memories in the evidence position."""
    messages = [dict(m) for m in recent_unit["stage"]["prompt_messages"]]
    block = "\n".join(f"- {m}" for m in memories)
    evidence = [dict(role="user", content=f"{CONTEXT_HEADER}\n{block}"),
                dict(role="assistant", content=CONTEXT_ACK)] if memories else []
    return messages[:-1] + evidence + messages[-1:]


def wrapped_list(reply, response_model):
    """A bare JSON list where the schema is an object holding one list field, wrapped back into that
    object — the model's answer unchanged, only its envelope restored; None when it is not that case.
    gemma4:e4b on Ollama returns `[...]` for Graphiti's NodeResolutions on LoCoMo conv-26 session 1
    despite the json_schema response format (2026-11-02), and Graphiti's own code then crashes on it."""
    if not isinstance(reply, list) or response_model is None:
        return None
    fields = response_model.model_fields
    if len(fields) != 1:
        return None
    (name, field), = fields.items()
    return {name: reply} if getattr(field.annotation, "__origin__", None) is list else None


def graphiti_client(cfg):
    """Graphiti's own OpenAI-compatible client, with one counted repair: wrapped_list. Every repair is
    warned and counted (stats → done.json), never silent."""
    import sys
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    class Client(OpenAIGenericClient):
        repairs = {}

        async def _generate_response(self, messages, response_model=None, *args, **kwargs):
            reply = await super()._generate_response(messages, response_model, *args, **kwargs)
            fixed = wrapped_list(reply, response_model)
            if fixed is None:
                return reply
            name = response_model.__name__
            self.repairs[name] = self.repairs.get(name, 0) + 1
            print(f"WARNING: graphiti {name}: bare list wrapped into its object "
                  f"({self.repairs[name]} so far)", file=sys.stderr, flush=True)
            return fixed

    return Client(config=cfg)


class GraphitiMemory:
    """Graphiti with its embedded Kuzu graph, an OpenAI-compatible client to Ollama for the LLM
    and embedder, and the LLM itself as reranker (Graphiti's documented local setup)."""

    def __init__(self, store):
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        cfg = LLMConfig(api_key="local", model=LLM_MODEL, small_model=LLM_MODEL, base_url=f"{OLLAMA}/v1")
        llm = graphiti_client(cfg)
        self.g = Graphiti(graph_driver=KuzuDriver(db=str(store / "graph.kuzu")), llm_client=llm,
                          embedder=OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key="local", embedding_model=EMBED_MODEL,
                                                                              base_url=f"{OLLAMA}/v1")),
                          cross_encoder=OpenAIRerankerClient(client=llm, config=cfg))

    async def setup(self):
        """Graphiti 0.31 on Kuzu never creates its own full-text indexes (build_indices_and_constraints
        is a no-op there and setup_schema skips them), so the first keyword search fails; create the
        ones its get_fulltext_indices(KUZU) defines and the store lacks (they persist on disk)."""
        from graphiti_core.driver.driver import GraphProvider
        from graphiti_core.graph_queries import get_fulltext_indices
        await self.g.build_indices_and_constraints()
        rows, _, _ = await self.g.driver.execute_query("CALL SHOW_INDEXES() RETURN *")
        have = {r["index_name"] for r in rows}
        for query in get_fulltext_indices(GraphProvider.KUZU):
            if query.split("'")[3] not in have:
                await self.g.driver.execute_query(query)

    def stats(self):
        """The counted repairs (wrapped_list), per response model."""
        return dict(repairs=dict(self.g.llm_client.repairs))

    async def add(self, when, sid, msgs):
        from graphiti_core.nodes import EpisodeType
        await self.g.add_episode(name=sid, episode_body=session_text(msgs), source=EpisodeType.message,
                                 source_description="chat session", reference_time=when)

    async def search(self, question):
        edges = await self.g.search(question, num_results=MEMORIES)
        return [e.fact + (f" (valid from {e.valid_at:%Y-%m-%d})" if getattr(e, "valid_at", None) else "")
                + (f" (no longer valid from {e.invalid_at:%Y-%m-%d})" if getattr(e, "invalid_at", None) else "")
                for e in edges]


class session_clock:
    """The clock a library's modules read, set to one session's time while it is ingested — as if the
    session were happening now. Only the modules' `datetime` name is swapped; their code is untouched."""

    def __init__(self, when, modules):
        self.when, self.modules, self.saved = when, modules, []

    def __enter__(self):
        when = self.when

        class Fixed(datetime):
            @classmethod
            def now(cls, tz=None):
                return when.astimezone(tz) if tz else when.replace(tzinfo=None)
        self.saved = [(m, m.datetime) for m in self.modules]
        for m in self.modules:
            m.datetime = Fixed
        return self

    def __exit__(self, *exc):
        for m, real in self.saved:
            m.datetime = real
        return False


class Mem0Memory:
    """Mem0's open-source SDK: Ollama LLM and embedder, Qdrant in local on-disk mode.

    Open-source Mem0 takes no session date (its `timestamp` is platform-only and raises), so its
    extraction resolves "yesterday" against the machine's clock — on a replay years later, a 2023
    event is stored under the run day. It assumes it ingests live; each session is ingested with
    the clock its prompt and storage modules read set to that session's time (session_clock), as if
    it ran that day — the Observation Date its own extraction prompt is written for. Mem0's code
    and prompts are unchanged (maintainer, 2026-10-31)."""

    def __init__(self, store):
        from mem0 import Memory
        self.m = Memory.from_config(dict(
            llm=dict(provider="ollama", config=dict(model=LLM_MODEL, ollama_base_url=OLLAMA, temperature=0)),
            embedder=dict(provider="ollama", config=dict(model=EMBED_MODEL, ollama_base_url=OLLAMA)),
            vector_store=dict(provider="qdrant", config=dict(path=str(store / "qdrant"), embedding_model_dims=768,
                                                             on_disk=True)),
            history_db_path=str(store / "history.db")))

    async def setup(self):
        return None

    speakers = ()                                  # set by conversation(): (speaker_a, speaker_b)

    async def add(self, when, sid, msgs):
        import mem0.configs.prompts
        import mem0.memory.main
        with session_clock(when, (mem0.configs.prompts, mem0.memory.main)):
            for speaker in self.speakers:
                self.m.add(from_side(msgs, speaker), user_id=speaker, metadata=dict(session=sid, date=when.isoformat()))

    async def search(self, question):
        out = []
        for speaker in self.speakers:
            found = self.m.search(question, filters=dict(user_id=speaker), top_k=MEMORIES)   # Mem0 2.x
            items = found.get("results", found) if isinstance(found, dict) else found
            out += [f"{speaker}: " + i.get("memory", "")
                    + (f" (from {i['metadata']['date'][:10]})" if i.get("metadata", {}).get("date") else "")
                    for i in items]
        return out


class AMemMemory:
    """A-Mem — the memory layer its paper evaluated (WujiangXu/A-mem @ b31a1dd, MIT, pinned in
    AMEM.json and fetched by vendor.py; the agiresearch/A-mem package skips note construction).
    Every note goes through its own LLM note construction (keywords, context, tags) and evolution
    (links, neighbour updates) on gemma4:e4b via Ollama, its all-MiniLM-L6-v2 embedder (CPU) and
    its embedding retriever. One note per message, timed by its session. Searched with the raw
    question, as Graphiti and Mem0 are (its paper harness first rewrites the question into keywords
    with another model call); its own text block — each hit plus its linked notes — is the
    evidence. Every model call at temperature 0 (its default is 0.7). A failed call becomes an
    empty JSON in its code: those fallbacks are counted and reported (stats), never hidden."""

    def __init__(self, store):
        import importlib.util
        from vendor import fetch
        spec = importlib.util.spec_from_file_location("amem_memory_layer", fetch())
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.mod.FAILURES.clear()
        self.m = self.mod.AgenticMemorySystem(model_name="all-MiniLM-L6-v2", llm_backend="ollama", llm_model=LLM_MODEL)
        llm, self.calls = self.m.llm_controller.llm, 0
        real = llm.get_completion

        def at_zero(prompt, response_format=None, temperature=0.7):
            self.calls += 1
            return real(prompt, response_format, temperature=0)
        llm.get_completion = at_zero

    async def setup(self):
        return None

    async def add(self, when, sid, msgs):
        for m in msgs:                             # one note per message; it names its speaker
            self.m.add_note(m["content"], time=when.strftime("%Y-%m-%d %H:%M"))

    async def search(self, question):
        block = self.m.find_related_memories_raw(question, k=MEMORIES)
        return [block] if block else []

    def stats(self):
        failed = sum(self.mod.FAILURES.values())
        return dict(model_calls=self.calls, fallbacks=dict(self.mod.FAILURES),
                    fallback_rate=round(failed / self.calls, 4) if self.calls else None)


SYSTEMS = {"graphiti": GraphitiMemory, "mem0": Mem0Memory, "amem": AMemMemory}
ON_DISK = {"graphiti", "mem0"}                     # stores that survive the process (A-Mem's lives in memory)


def stats(memory):
    """A system's own call/fallback counts where it keeps them (A-Mem), else None."""
    return memory.stats() if hasattr(memory, "stats") else None


def fresh_store(d, keep_ingested=False):
    """A store a crash left half-ingested is rebuilt from nothing, never ingested twice. With
    keep_ingested, a fully ingested on-disk store (ingested.json) is kept with its saved hits."""
    if d.exists() and not (d / "done.json").exists() and not (keep_ingested and (d / "ingested.json").exists()):
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    """Whole or not at all: a stop mid-write leaves no half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)


def originals_sessions(originals, gap=timedelta(hours=6)):
    """Exported originals (LoCoMo) as [(date, session id, [{role, content}])]: a new session
    wherever the source clock jumps by *gap* or more (experiments/z2/clock.py's rule)."""
    out, last = [], None
    for r in originals:
        when = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
        if last is None or when - last >= gap:
            out.append((when, f"s{len(out) + 1}", []))
        if r["prompt"]:                            # a session the second speaker opens has no user side
            out[-1][2].append(dict(role="user", content=r["prompt"]))
        if r["response"]:
            out[-1][2].append(dict(role="assistant", content=r["response"]))
        last = when
    return out


def speakers(originals):
    """(speaker_a, speaker_b) from the exported originals' name prefixes."""
    name = lambda text: text.split(": ", 1)[0]
    return (name(next(r["prompt"] for r in originals if r["prompt"])),
            name(next(r["response"] for r in originals if r["response"])))


def questions(asks):
    """Every ask searched: the questions and the no-memory controls (FINAL revision 5)."""
    return [a for a in asks if a.get("kind") in ("question", "control")]


async def search(system, history, originals, asks, out):
    """One LoCoMo conversation: one store, every session ingested, then each question searched."""
    d = out / "instances" / history
    if (d / "done.json").exists():
        return
    resume = system in ON_DISK and (d / "ingested.json").exists()
    fresh_store(d, keep_ingested=system in ON_DISK)
    started = time.time()
    memory = SYSTEMS[system](d)
    memory.speakers = speakers(originals)
    await memory.setup()
    groups = originals_sessions(originals)
    if resume:
        ingest = json.loads((d / "ingested.json").read_text())["ingest_seconds"]
        print(json.dumps(dict(history=history, resumed="search", hits=len(list((d / "hits").glob("*.json"))))), flush=True)
    else:
        for i, (when, sid, msgs) in enumerate(groups, 1):
            await memory.add(when, sid, msgs)
            write_json(d / "ingest-progress.json", dict(sessions=i, of=len(groups)))   # the runner's meter
        ingest = time.time() - started
        write_json(d / "ingested.json", dict(policy=POLICY, system=system, ingest_seconds=round(ingest, 1)))
    for ask in questions(asks):
        path = d / "hits" / f"{ask['id']}.json"
        if path.exists():
            continue
        write_json(path, dict(policy=POLICY, history=history, system=system, ask_id=ask["id"],
                              question=ask["question"], memories=await memory.search(ask["question"])))
    write_json(d / "done.json", dict(policy=POLICY, system=system, history=history, ingest_seconds=round(ingest, 1),
                                     sessions=len(groups), questions=len(asks), stats=stats(memory)))


def wrap(system, history, asks, search_out, ice_units, out):
    """The saved hits in ICE's recent_only envelope for the same question, as read units."""
    d = search_out / "instances" / history
    if not (d / "done.json").exists():
        raise SystemExit(f"{system} {history}: search not finished ({d}/done.json missing)")
    for ask in questions(asks):
        hit = json.loads((d / "hits" / f"{ask['id']}.json").read_text())
        recent = json.loads((ice_units / "units" / "recent_only" / f"{ask['id']}.json").read_text())
        memories = hit["memories"]
        write_json(out / "units" / system / f"{ask['id']}.json",
                   dict(policy=POLICY, history=history, condition=system, ask=recent["ask"],
                        checkpoint=recent.get("checkpoint"), question_time=recent.get("question_time"),
                        stage=dict(prompt_messages=envelope(recent, memories, ask["question"]), memories=memories,
                                   selected_tokens=sum(len(m) for m in memories) // 4)))
    write_json(out / "instances" / history / "done.json", json.loads((d / "done.json").read_text()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="phase", required=True)
    for name in ("search", "wrap"):
        p = sub.add_parser(name)
        p.add_argument("--system", choices=sorted(SYSTEMS), required=True)
        p.add_argument("--history", required=True, help="the LoCoMo history key, e.g. LOCOMO-conv-26")
        p.add_argument("--out", type=Path, required=True)
        if name == "wrap":
            p.add_argument("--search", type=Path, required=True, help="the search phase's --out")
            p.add_argument("--ice-units", type=Path, required=True, help="ICE's read folder (recent_only envelopes)")
    args = parser.parse_args()
    base = ROOT / "logs/final-locomo" / args.history.removeprefix("LOCOMO-")
    asks = json.loads((base / "asks.json").read_text())
    if args.phase == "search":
        try:
            asyncio.run(search(args.system, args.history, json.loads((base / "originals.json").read_text()), asks, args.out))
        except KeyboardInterrupt:
            print(json.dumps(dict(history=args.history, stopped=True)), flush=True)
            return 130
    else:
        wrap(args.system, args.history, asks, args.search, args.ice_units, args.out)
    print(json.dumps(dict(history=args.history, phase=args.phase, done=True)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
