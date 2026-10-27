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
unchanged (answered by FINAL's answerer). Durable per conversation (done.json): a store left
half-ingested by a crash is wiped and rebuilt, never ingested twice.

    cd experiments/final/baselines
    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --no-sync python run_baseline.py --system mem0 \
        --history LOCOMO-conv-26 --ice-units <its local-read folder> --out <dir>
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


class GraphitiMemory:
    """Graphiti with its embedded Kuzu graph, an OpenAI-compatible client to Ollama for the LLM
    and embedder, and the LLM itself as reranker (Graphiti's documented local setup)."""

    def __init__(self, store):
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        cfg = LLMConfig(api_key="local", model=LLM_MODEL, small_model=LLM_MODEL, base_url=f"{OLLAMA}/v1")
        llm = OpenAIGenericClient(config=cfg)
        self.g = Graphiti(graph_driver=KuzuDriver(db=str(store / "graph.kuzu")), llm_client=llm,
                          embedder=OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key="local", embedding_model=EMBED_MODEL,
                                                                              base_url=f"{OLLAMA}/v1")),
                          cross_encoder=OpenAIRerankerClient(client=llm, config=cfg))

    async def setup(self):
        await self.g.build_indices_and_constraints()

    async def add(self, when, sid, msgs):
        from graphiti_core.nodes import EpisodeType
        await self.g.add_episode(name=sid, episode_body=session_text(msgs), source=EpisodeType.message,
                                 source_description="chat session", reference_time=when)

    async def search(self, question):
        edges = await self.g.search(question, num_results=MEMORIES)
        return [e.fact + (f" (valid from {e.valid_at:%Y-%m-%d})" if getattr(e, "valid_at", None) else "")
                + (f" (no longer valid from {e.invalid_at:%Y-%m-%d})" if getattr(e, "invalid_at", None) else "")
                for e in edges]


class Mem0Memory:
    """Mem0's open-source SDK: Ollama LLM and embedder, Qdrant in local on-disk mode."""

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
        for speaker in self.speakers:
            self.m.add(from_side(msgs, speaker), user_id=speaker, metadata=dict(session=sid, date=when.isoformat()))

    async def search(self, question):
        out = []
        for speaker in self.speakers:
            found = self.m.search(question, user_id=speaker, limit=MEMORIES)
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


def stats(memory):
    """A system's own call/fallback counts where it keeps them (A-Mem), else None."""
    return memory.stats() if hasattr(memory, "stats") else None


def fresh_store(d):
    """A store a crash left half-ingested is rebuilt from nothing, never ingested twice."""
    if d.exists() and not (d / "done.json").exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


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


async def conversation(system, history, originals, asks, ice_units, out):
    """One LoCoMo conversation: one store, every session ingested, then each question searched."""
    d = out / "instances" / history
    if (d / "done.json").exists():
        return
    fresh_store(d)
    started = time.time()
    memory = SYSTEMS[system](d)
    memory.speakers = speakers(originals)
    await memory.setup()
    groups = originals_sessions(originals)
    for when, sid, msgs in groups:
        await memory.add(when, sid, msgs)
    ingest = time.time() - started
    for ask in (a for a in asks if a.get("kind") == "question"):
        recent = json.loads((ice_units / "units" / "recent_only" / f"{ask['id']}.json").read_text())
        memories = await memory.search(ask["question"])
        unit = dict(policy=POLICY, history=history, condition=system, ask=recent["ask"], checkpoint=recent.get("checkpoint"),
                    question_time=recent.get("question_time"),
                    stage=dict(prompt_messages=envelope(recent, memories, ask["question"]), memories=memories,
                               selected_tokens=sum(len(m) for m in memories) // 4))
        path = out / "units" / system / f"{ask['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(unit, ensure_ascii=False, indent=1) + "\n")
    (d / "done.json").write_text(json.dumps(dict(policy=POLICY, system=system, history=history,
                                                 ingest_seconds=round(ingest, 1), sessions=len(groups),
                                                 questions=len(asks), stats=stats(memory))) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--history", required=True, help="the LoCoMo history key, e.g. LOCOMO-conv-26")
    parser.add_argument("--ice-units", type=Path, required=True, help="ICE's read folder (recent_only envelopes)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = ROOT / "logs/final-locomo" / args.history.removeprefix("LOCOMO-")
    asyncio.run(conversation(args.system, args.history, json.loads((base / "originals.json").read_text()),
                             json.loads((base / "asks.json").read_text()), args.ice_units, args.out))
    print(json.dumps(dict(history=args.history, done=True)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
