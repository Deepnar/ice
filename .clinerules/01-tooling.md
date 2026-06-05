# Tooling & Environment

## Package Management
- Use **uv** for all Python operations. Never use pip, pip3, or pip install under any circumstances.
- `uv add <package>` — add a dependency and write to pyproject.toml
- `uv remove <package>` — remove a dependency
- `uv run <script.py>` — run a script inside the managed venv
- `uv run pytest` — run tests
- `uv sync` — sync the environment to pyproject.toml
- The virtual environment is managed by uv automatically. Do not manually create or activate venvs.
- Dependencies live in `pyproject.toml`. There is no requirements.txt.

## Model Serving
- All inference is served via **vLLM** on port **8001**. Do NOT use Ollama or any other inference tool.
- **Labeling model:** `Qwen/Qwen2.5-7B-Instruct-AWQ` — start with the `vllm-label` shell function. Do NOT add `--kv-cache-dtype fp8` for the labeling model; fp8 introduces quantization artifacts that corrupt strict JSON parsing.
- **Coding assistant:** `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` — start with the `vllm-coder` shell function (includes `--kv-cache-dtype fp8`; safe for free-form text generation).
- **Background NLP model** (workers only, never user-facing): `Qwen2.5-1.5B Q8_0` served on the same vLLM instance at a sub-endpoint.
- vLLM exposes an OpenAI-compatible API at `http://localhost:8001/v1`.
- When writing code that calls the inference API, always use `http://localhost:8001/v1` as the base URL and treat it as an OpenAI-compatible endpoint.

## Hardware
- OS: CachyOS (Arch-based Linux), Hyprland WM.
- GPU: NVIDIA RTX 5090 Laptop GPU, 24 GB VRAM.
- PyTorch operations default to CUDA unless explicitly specified otherwise.
- Monitor VRAM with `nvtop`.

## Infrastructure (Docker)
- PostgreSQL 16 with pgvector: started via `docker compose up -d` from `docker/`.
- Redis 7: also started via the same docker compose.
- Connection strings are in `.env` — never hardcode them.
- All infrastructure configuration lives in `docker/docker-compose.yml`.

## Embeddings
- `all-MiniLM-L6-v2` (384-dim) is used for ALL embeddings throughout the project — both the classifier input and all memory store embeddings.
- Never swap this model mid-project. If the embedding model changes, all stored embeddings become invalid and the database must be rebuilt.

## Logging
- Structured JSON logging via `structlog` in all server-side code.
- Always attach a `correlation_id` to every log line within a request or task scope.

## Key Documents (priority order)
1. `PROGRESS.md` — the current project state. Read this FIRST before anything else, every session.
2. `docs/ARCHITECTURE.md` — full system design and invariants. Read **only the section relevant to the current task**, not the whole file.
3. `docs/BLUEPRINT.md` — step-by-step build guide. Treat as a guide, not a contract. The actual implementation may diverge in file structure, tool choices, or approach as long as it remains consistent with ARCHITECTURE.md invariants.