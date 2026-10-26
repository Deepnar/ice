"""FINAL memory-system baseline on the LongMemEval-S slice (Z3 D11): Graphiti or Mem0, open
source, run locally, so only the memory system differs from ICE.

Runs in this folder's own environment (isolated from ICE's). Per instance of the pinned slice:
a fresh store; every haystack session ingested in date order through the system's own write
path, with FINAL's background model (gemma4:e4b via Ollama) doing its extraction and the
system's documented local embedder (nomic-embed-text via Ollama); at the question date, the
system's own search returns its top MEMORIES results; they are placed in exactly the evidence
position of ICE's bare-vector condition (ICE's recent_only envelope for the same instance), and
written as a read unit that experiments/z2/answers.py and oracle.py grade take unchanged.
Durable per instance (done.json); a rerun resumes.

    cd experiments/final/baselines
    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --no-sync python run_baseline.py --system graphiti \
        --ice-units ../../../logs/final-run-main/local-lmes --out ../../../logs/final-run-main/baseline-graphiti
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/lme/data/longmemeval_s"
SLICE = ROOT / "experiments/lme/final_slice.json"
OLLAMA = "http://localhost:11434"
LLM_MODEL, EMBED_MODEL = "gemma4:e4b", "nomic-embed-text"
MEMORIES = 10
POLICY = "ice-final-baseline-1"
# The evidence pair of ICE's prompt assembler (experiments/z2/read_passes.py CONTEXT_HEADER /
# CONTEXT_ACK) — kept identical; tests/smoke/test_final_baselines.py checks it.
CONTEXT_HEADER = "=== RETRIEVED CONTEXT ==="
CONTEXT_ACK = "Understood — I have the background context. What would you like to know?"


def parse_date(raw):
    """LongMemEval dates, e.g. '2023/05/20 (Sat) 02:21'."""
    return datetime.strptime(raw.split(" (")[0] + " " + raw.split(") ")[-1], "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)


def sessions(inst):
    """[(date, session id, [{role, content}])] in date order."""
    rows = [(parse_date(d), sid, msgs) for d, sid, msgs in
            zip(inst["haystack_dates"], inst["haystack_session_ids"], inst["haystack_sessions"])]
    return sorted(rows, key=lambda r: r[0])


def session_text(msgs):
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)


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

    async def add(self, when, sid, msgs):
        self.m.add(msgs, user_id="lme", metadata=dict(session=sid, date=when.isoformat()))

    async def search(self, question):
        found = self.m.search(question, user_id="lme", limit=MEMORIES)
        items = found.get("results", found) if isinstance(found, dict) else found
        return [i.get("memory", "") + (f" (from {i['metadata']['date'][:10]})" if i.get("metadata", {}).get("date") else "")
                for i in items]


SYSTEMS = {"graphiti": GraphitiMemory, "mem0": Mem0Memory}


async def one(system, inst, recent_unit, out):
    qid = inst["question_id"]
    d = out / "instances" / qid
    if (d / "done.json").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    started = time.time()
    memory = SYSTEMS[system](d)
    await memory.setup()
    for when, sid, msgs in sessions(inst):
        await memory.add(when, sid, msgs)
    ingest = time.time() - started
    memories = await memory.search(inst["question"])
    messages = envelope(recent_unit, memories, inst["question"])
    unit = dict(policy=POLICY, history="lme-s", condition=system, ask=recent_unit["ask"], checkpoint=None,
                question_time=recent_unit.get("question_time"),
                stage=dict(prompt_messages=messages, memories=memories,
                           selected_tokens=sum(len(m) for m in memories) // 4))
    path = out / "units" / system / f"{qid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(unit, ensure_ascii=False, indent=1) + "\n")
    (d / "done.json").write_text(json.dumps(dict(policy=POLICY, system=system, question_id=qid,
                                                 ingest_seconds=round(ingest, 1),
                                                 sessions=len(inst["haystack_sessions"]),
                                                 memories=len(memories))) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--ice-units", type=Path, required=True, help="ICE's LongMemEval-S read folder (recent_only envelopes)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    ids = json.loads(SLICE.read_text())["question_ids"][:args.limit] if args.limit else json.loads(SLICE.read_text())["question_ids"]
    data = {i["question_id"]: i for i in json.loads(DATA.read_text())}
    for qid in ids:
        recent = json.loads((args.ice_units / "units" / "recent_only" / f"{qid}.json").read_text())
        asyncio.run(one(args.system, data[qid], recent, args.out))
        print(json.dumps(dict(question_id=qid, done=True)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
