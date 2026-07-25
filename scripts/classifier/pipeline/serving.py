"""Local OpenAI-compatible inference server, for the bulk offline jobs.

B1 is 100% local — no cloud calls anywhere in this phase (cloud is reserved
entirely for FINAL). Bulk labeling therefore runs on the user's own 24 GB GPU,
and **not through Ollama**: Ollama is unusably slow for this shape of work.

**SGLang is the preferred backend for this exact workload**, for two reasons that
are specific to labeling rather than general benchmarketing:

* **RadixAttention prefix caching.** The rubric is ~4,000 tokens and byte
  identical across every row in the corpus. That is the maximally prefix-heavy
  case; the shared prefix is computed once and reused for every subsequent
  request. (On all-unique prompts this advantage largely disappears — ours is the
  opposite situation.)
* **Constrained decoding from a JSON schema.** Output is forced to be valid by
  construction, which removes v1's retry-on-invalid-JSON loop entirely.

vLLM is a supported fallback (the v1 labeler already served AWQ weights through
it at ``localhost:8001/v1``). Both speak the same OpenAI API, so ``label.py`` and
``synth.py`` never learn which one is running.

**Backend reality as of 2026-07-25: the fallback is what actually runs.** The
pinned ``sglang 0.3.6.post2`` cannot import against the environment's Triton
(``ImportError: cannot import name 'default_cache_dir' from
'triton.runtime.cache'``), and unpinning SGLang would drag the repo's torch stack
with it — a much larger intervention than the spec's own sanctioned fallback.
vLLM 0.22.0 runs, and with ``--enable-prefix-caching`` it captures the part of
the SGLang argument that mattered here (the ~4k-token rubric is shared by every
row). ``--backend sglang`` stays wired and becomes correct again the day SGLang
is upgraded.

**One model at a time.** 24 GB holds exactly one of these models, so the two
labelers run SEQUENTIALLY — A over the whole corpus, shut down, then B. That is
also why each model gets the *full* GPU rather than being shrunk to co-exist.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# 24 GB budget. Context length is the lever that matters: these models advertise
# 128k–256k windows and the server will happily reserve KV cache for all of it,
# then OOM. The labeling prompt is rubric (~4k) + row (usually <1k) + reasoning
# output (<1k), so 8k is generous and leaves the rest of the card for the KV pool
# that actually gives us throughput.
DEFAULT_CONTEXT_LENGTH = 8192
DEFAULT_PORT = 30000

# Per-model serving profiles. mem_fraction is the share of the 24 GB the server
# may claim for weights + KV pool; max_running is how many rows are decoded
# concurrently. Dense 27B leaves ~7 GB for KV after ~16 GB of weights; the MoE
# models are lighter on weights or cheaper per token, so they can run wider.
#
# quantization stays None on purpose: community requants of the same model
# disagree about their own format (one of these ships AWQ, another ships
# compressed-tensors), and every checkpoint already declares its method in
# config.json. Forcing a value here only creates a mismatch the server refuses.
PROFILES = {
    # labeler A — dense, strongest instruction-following at this size
    "qwen3.6-27b": {
        "model": "mattbucci/Qwen3.6-27B-AWQ",
        "family": "qwen",
        "mem_fraction": 0.88,
        "max_running": 24,
        "quantization": None,
    },
    # labeler B — DIFFERENT FAMILY (this is the point: agreement between two Qwen
    # variants measures a model against itself). MoE, ~4B active ⇒ fast decode.
    "gemma-4-26b-a4b": {
        "model": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "family": "gemma",
        "mem_fraction": 0.88,
        "max_running": 32,
        "quantization": None,
    },
    # tiebreak C — a third family again, only ~200–400 rows
    "mistral-small-24b": {
        "model": "jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym",
        "family": "mistral",
        "mem_fraction": 0.88,
        "max_running": 24,
        "quantization": None,
    },
    # throughput swap for A (MoE, ~3B active) — legitimate only after a ~200-row
    # quality spot-check against the dense 27B (spec 6b).
    "qwen3.6-35b-a3b": {
        "model": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
        "family": "qwen",
        "mem_fraction": 0.92,
        "max_running": 16,
        "quantization": None,
    },
}


def resolve(name_or_path: str) -> dict:
    """Profile by short name, or a bare HF path with defaults."""
    if name_or_path in PROFILES:
        return {"key": name_or_path, **PROFILES[name_or_path]}
    return {"key": name_or_path, "model": name_or_path, "family": "unknown",
            "mem_fraction": 0.88, "max_running": 24, "quantization": None}


def is_up(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def served_model(base_url: str) -> str | None:
    import json
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
            data = json.loads(resp.read())
        return (data.get("data") or [{}])[0].get("id")
    except Exception:
        return None


class LocalServer:
    """Launch, wait for, and shut down one local inference server."""

    def __init__(self, model: str, backend: str = "vllm", port: int = DEFAULT_PORT,
                 context_length: int = DEFAULT_CONTEXT_LENGTH,
                 mem_fraction: float | None = None, max_running: int | None = None,
                 quantization: str | None = None, log_path: str | None = None):
        profile = resolve(model)
        self.key = profile["key"]
        self.model = profile["model"]
        self.family = profile["family"]
        self.backend = backend
        self.port = port
        self.context_length = context_length
        self.mem_fraction = mem_fraction or profile["mem_fraction"]
        self.max_running = max_running or profile["max_running"]
        self.quantization = quantization or profile.get("quantization")
        # Repo-root anchored: these stages are run from their own directory, and
        # a relative path would scatter server logs wherever the shell happened
        # to be.
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self.log_path = log_path or os.path.join(
            root, "logs", f"{backend}_{self.key.replace('/', '_')}.log")
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _command(self) -> list:
        if self.backend == "sglang":
            cmd = [sys.executable, "-m", "sglang.launch_server",
                   "--model-path", self.model,
                   "--port", str(self.port),
                   "--host", "127.0.0.1",
                   "--context-length", str(self.context_length),
                   "--mem-fraction-static", str(self.mem_fraction),
                   "--max-running-requests", str(self.max_running)]
            if self.quantization:
                cmd += ["--quantization", self.quantization]
            return cmd
        if self.backend == "vllm":
            cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                   "--model", self.model,
                   "--port", str(self.port),
                   "--host", "127.0.0.1",
                   "--max-model-len", str(self.context_length),
                   "--gpu-memory-utilization", str(self.mem_fraction),
                   "--max-num-seqs", str(self.max_running),
                   # Prefix caching is the whole reason this workload is cheap.
                   "--enable-prefix-caching"]
            if self.quantization:
                cmd += ["--quantization", self.quantization]
            return cmd
        raise ValueError(f"unknown backend {self.backend!r} (sglang | vllm)")

    def start(self, timeout: int = 1800) -> "LocalServer":
        if is_up(self.base_url):
            raise RuntimeError(
                f"something is already serving on port {self.port} "
                f"({served_model(self.base_url)}). Only ONE model fits in 24 GB — "
                f"stop it first or pass a different --port.")

        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        cmd = self._command()
        print(f"[serving] {self.backend}: {self.model}")
        print(f"[serving] ctx={self.context_length} mem={self.mem_fraction} "
              f"concurrency={self.max_running} → {self.log_path}")
        with open(self.log_path, "w") as log:
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = _tail(self.log_path, 30)
                raise RuntimeError(
                    f"{self.backend} exited with code {self.proc.returncode} during "
                    f"startup.\n--- {self.log_path} (tail) ---\n{tail}")
            if is_up(self.base_url):
                print(f"[serving] ready after {int(timeout - (deadline - time.time()))}s")
                return self
            time.sleep(3)

        self.stop()
        raise TimeoutError(f"{self.backend} did not come up within {timeout}s — "
                           f"see {self.log_path}")

    def stop(self) -> None:
        """Terminate the server and WAIT for the VRAM to actually come back.

        The wait is not politeness: the next labeler cannot load until this one's
        16 GB is released, and starting it early is an OOM.
        """
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                self.proc.terminate()
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    self.proc.kill()
                self.proc.wait(timeout=30)
        self.proc = None
        _wait_for_vram()
        print("[serving] stopped")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def _tail(path: str, lines: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])
    except OSError:
        return "(no log)"


def _wait_for_vram(free_mib: int = 18000, timeout: int = 180) -> None:
    """Block until the GPU has *free_mib* free again (best-effort)."""
    if not shutil.which("nvidia-smi"):
        time.sleep(10)
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
            if out and int(out[0]) >= free_mib:
                return
        except Exception:
            return
        time.sleep(5)
