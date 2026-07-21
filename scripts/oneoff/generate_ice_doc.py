#!/usr/bin/env python3
"""
ICE Project Documentation Generator
=====================================
Generates 3 layered markdown docs for the ICE codebase:
  1. ice_raw_extract.md      — folder/file structure + raw code / data previews
  2. ice_llm_summary.md      — LLM-generated per-file technical summaries
  3. ice_full_context.md     — combined: summary + full code per file

Usage: python generate_ice_docs.py
Run from: /home/deepnar/Programs/ice/
Output:   /home/deepnar/Programs/ice/docs/
"""

import os
import sys
import json
import subprocess
import textwrap
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ICE_ROOT        = Path("/home/deepnar/Programs/ice")
DOCS_DIR        = ICE_ROOT / "docs"
RAW_MD          = DOCS_DIR / "RAW.md"
SUMMARY_MD      = DOCS_DIR / "SUMMARY.md"
FULL_CONTEXT_MD = DOCS_DIR / "FULL_CONTEXT.md"

OLLAMA_MODEL    = "qwen3-coder:30b-a3b-q4_K_M"
OLLAMA_URL      = "http://localhost:11434/api/generate"

# Data/config files: only preview first N lines (1 line = schema only)
DATA_EXTENSIONS    = {".json", ".jsonl", ".txt", ".csv", ".tsv", ".log", ".env"}
DATA_PREVIEW_LINES = 1   # one line reveals schema/structure, not content

# Dirs to skip traversal entirely
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", "dist", "build", "eggs",
    ".eggs", "docs",  # skip docs dir itself to avoid recursion
}

# Files to skip entirely (no entry in any doc)
ALWAYS_SKIP_FILES = {
    ".DS_Store", "Thumbs.db",
    "ice_raw_extract.md", "ice_llm_summary.md", "ice_full_context.md",
    "generate_ice_docs.py",
}

# Non-readable extensions: listed in docs with name/path + LLM note, but no raw content
NON_READABLE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".egg",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".bin", ".pkl", ".pt", ".safetensors", ".gguf",
    ".lock",
}

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def is_data_file(path: Path) -> bool:
    return path.suffix.lower() in DATA_EXTENSIONS


def is_non_readable(path: Path) -> bool:
    return path.suffix.lower() in NON_READABLE_EXTENSIONS


def read_file_content(path: Path) -> tuple[str, bool]:
    """
    Returns (content, truncated).
    - Non-readable (binary/weights/images): empty string, truncated=False
    - .json        : parse + pretty-print top 2 levels of keys (30 raw lines fallback)
    - .jsonl       : first 1 line (each line is a full record = full schema)
    - .csv / .tsv  : first 1 line (header row = full schema)
    - .txt/.log/.env: first 1 line (structure sample)
    - code files   : full content
    """
    if is_non_readable(path):
        return "", False

    ext = path.suffix.lower()

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception as e:
        return f"[ERROR reading file: {e}]", False

    # ── .json: parse and show schema (top 2 levels) ──────────────
    if ext == ".json":
        try:
            parsed = json.loads(raw)
            schema = _json_schema_preview(parsed, depth=2)
            return schema, True
        except Exception:
            # fallback: first 30 raw lines
            lines = raw.splitlines(keepends=True)
            if len(lines) > 30:
                return "".join(lines[:30]), True
            return raw, False

    # ── .jsonl: first line = one full record ─────────────────────
    if ext == ".jsonl":
        first = raw.split("\n", 1)[0].strip()
        try:
            obj = json.loads(first)
            return json.dumps(obj, indent=2), True
        except Exception:
            return first, True

    # ── .csv / .tsv: header row only ─────────────────────────────
    if ext in {".csv", ".tsv"}:
        header = raw.split("\n", 1)[0]
        return header, True

    # ── .txt / .log / .env: first line ───────────────────────────
    if ext in {".txt", ".log", ".env"}:
        first = raw.split("\n", 1)[0]
        return first, True

    # ── fallback for anything else in DATA_EXTENSIONS ────────────
    if is_data_file(path):
        lines = raw.splitlines(keepends=True)
        if len(lines) > 1:
            return lines[0], True
        return raw, False

    # ── code files: full content ──────────────────────────────────
    return raw, False


def _json_schema_preview(obj, depth: int, _cur: int = 0) -> str:
    """
    Recursively build a schema-like preview of a JSON object/array.
    Shows keys and value types; truncates arrays to 1 example element.
    """
    indent = "  " * _cur
    if isinstance(obj, dict):
        if _cur >= depth:
            keys = list(obj.keys())
            return f"{{...}}  # keys: {keys}"
        lines = ["{"]
        for k, v in list(obj.items())[:20]:  # cap at 20 keys
            child = _json_schema_preview(v, depth, _cur + 1)
            lines.append(f"{indent}  {json.dumps(k)}: {child},")
        if len(obj) > 20:
            lines.append(f"{indent}  ... ({len(obj)} keys total)")
        lines.append(f"{indent}}}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        if _cur >= depth:
            return f"[...]  # {len(obj)} items, type: {type(obj[0]).__name__}"
        example = _json_schema_preview(obj[0], depth, _cur + 1)
        return f"[\n{indent}  {example},\n{indent}  ... ({len(obj)} items)\n{indent}]"
    else:
        # scalar: show type and a short value preview
        if isinstance(obj, str) and len(obj) > 60:
            return f'"{obj[:57]}..."  # str'
        return f"{json.dumps(obj)}  # {type(obj).__name__}"


def get_code_fence(path: Path) -> str:
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "jsx", ".tsx": "tsx", ".sh": "bash", ".yaml": "yaml",
        ".yml": "yaml", ".toml": "toml", ".json": "json",
        ".jsonl": "json", ".sql": "sql", ".md": "markdown",
        ".html": "html", ".css": "css", ".env": "bash",
        ".txt": "text", ".cfg": "ini", ".ini": "ini",
    }
    return ext_map.get(path.suffix.lower(), "text")


def collect_files(root: Path) -> list[tuple[Path, list[Path]]]:
    """
    Walk root, return list of (dir_path, [file_paths]) in sorted order.
    Includes ALL files except ALWAYS_SKIP_FILES and hidden files.
    Non-readable files (binary etc) are included — handled separately in writers.
    """
    result = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        )
        dp = Path(dirpath)
        files = sorted(
            dp / f for f in filenames
            if f not in ALWAYS_SKIP_FILES
            and not f.startswith(".")
        )
        if files or dp == root:
            result.append((dp, files))
    return result


def rel(path: Path) -> str:
    """Return path relative to ICE_ROOT for display."""
    try:
        return str(path.relative_to(ICE_ROOT))
    except ValueError:
        return str(path)


# ──────────────────────────────────────────────
# LLM CALL
# ──────────────────────────────────────────────

LLM_SYSTEM = """You are a senior backend engineer reviewing source files from a large Python/AI project.
For each file given, produce a dense technical summary in Markdown covering:
- **Purpose**: What this file does in 1-2 sentences.
- **Key Functions / Classes**: For every function and class: name, parameters, return type/value, what it does, side effects.
- **Key Variables / Constants**: Important module-level variables, their types and roles.
- **Imports & Dependencies**: External libs or internal modules imported and why.
- **Data Flow**: How data enters, transforms, and exits this file.
- **Integration Points**: What calls this file and what this file calls.
- **Edge Cases / Notes**: Any gotchas, TODOs, or important design decisions visible in the code.

For binary/non-readable files (images, weights, compiled, archives): describe what this file type likely is,
its probable role in the project given its name and path, and how it is typically used.

Be exhaustive. This summary replaces reading the file. Use bullet points and sub-bullets. No fluff.
"""

def llm_summarize(file_path: Path, content: str, truncated: bool) -> str:
    """Call local Ollama model and return the summary string."""
    if is_non_readable(file_path):
        prompt = (
            f"File: {rel(file_path)}\n"
            f"Extension: {file_path.suffix}\n"
            f"[This is a binary/non-readable file — no content available.]\n\n"
            f"Based on the filename, path, and extension, describe what this file likely is, "
            f"its probable role in this AI/ML project, and how it is typically used or generated."
        )
    elif is_data_file(file_path):
        ext = file_path.suffix.lower()
        if ext == ".json":
            schema_note = "[DATA FILE — schema preview only (top 2 levels of keys/types shown, not full content). Summarize the data structure and its likely role.]\n"
        elif ext == ".jsonl":
            schema_note = "[DATA FILE — first record shown (each line is one full JSON record). Summarize the schema and what each field likely represents.]\n"
        elif ext in {".csv", ".tsv"}:
            schema_note = "[DATA FILE — header row only shown. Summarize the column schema and likely data purpose.]\n"
        else:
            schema_note = "[DATA FILE — first line shown as structure sample. Summarize the format and likely purpose.]\n"
        prompt = f"File: {rel(file_path)}\n{schema_note}\n```{get_code_fence(file_path)}\n{content}\n```\n\nWrite the full technical summary:"
    else:
        prompt = f"File: {rel(file_path)}\n\n```{get_code_fence(file_path)}\n{content}\n```\n\nWrite the full technical summary:"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": LLM_SYSTEM,
        "stream": False,
        "think": False,   # disable CoT — structured extraction, not reasoning
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
        }
    }

    try:
        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "[No response from LLM]").strip()
    except Exception as e:
        return f"[LLM ERROR: {e}]"


# ──────────────────────────────────────────────
# SECTION WRITERS
# ──────────────────────────────────────────────

def write_file_block_raw(f, file_path: Path, content: str, truncated: bool):
    f.write(f"#### 📄 `{file_path.name}`\n")
    f.write(f"**Path:** `{rel(file_path)}`\n\n")

    if is_non_readable(file_path):
        size_str = ""
        try:
            size_kb = file_path.stat().st_size / 1024
            size_str = f" ({size_kb:.1f} KB)"
        except Exception:
            pass
        f.write(f"> 🚫 **Binary / non-readable file**{size_str} — no raw content extracted.\n\n")
    else:
        fence = get_code_fence(file_path)
        if truncated:
            f.write(f"> ⚠️ **Data file — schema preview only (first {DATA_PREVIEW_LINES} line)**\n\n")
        f.write(f"```{fence}\n{content}\n```\n\n")

    f.write("---\n\n")


def write_file_block_summary(f, file_path: Path, summary: str):
    f.write(f"#### 📄 `{file_path.name}`\n")
    f.write(f"**Path:** `{rel(file_path)}`\n\n")
    f.write(summary)
    f.write("\n\n---\n\n")


def write_file_block_full(f, file_path: Path, content: str, truncated: bool, summary: str):
    f.write(f"#### 📄 `{file_path.name}`\n")
    f.write(f"**Path:** `{rel(file_path)}`\n\n")
    f.write("##### 🧠 LLM Summary\n\n")
    f.write(summary)
    f.write("\n\n")

    if is_non_readable(file_path):
        size_str = ""
        try:
            size_kb = file_path.stat().st_size / 1024
            size_str = f" ({size_kb:.1f} KB)"
        except Exception:
            pass
        f.write("##### 📋 Source\n\n")
        f.write(f"> 🚫 **Binary / non-readable file**{size_str} — no raw content available.\n\n")
    else:
        fence = get_code_fence(file_path)
        f.write("##### 📋 Source / Preview\n\n")
        if truncated:
            f.write(f"> ⚠️ **Data file — schema preview only (first {DATA_PREVIEW_LINES} line)**\n\n")
        f.write(f"```{fence}\n{content}\n```\n\n")

    f.write("---\n\n")


def write_dir_header(f, dir_path: Path, level: int = 2):
    rel_dir = rel(dir_path)
    prefix  = "🗂️" if dir_path != ICE_ROOT else "🏠"
    f.write(f"{'#' * level} {prefix} `{dir_path.name}/`\n")
    f.write(f"**Path:** `{rel_dir}/`\n\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📂 ICE Root: {ICE_ROOT}")
    print(f"📁 Output:   {DOCS_DIR}")
    print(f"🤖 Model:    {OLLAMA_MODEL}\n")

    structure = collect_files(ICE_ROOT)

    total_files = sum(len(files) for _, files in structure)
    print(f"Found {total_files} files across {len(structure)} directories.\n")

    # Pre-read all files once, store content
    # file_data: {Path -> (content, truncated)}
    file_data: dict[Path, tuple[str, bool]] = {}
    for _, files in structure:
        for fp in files:
            file_data[fp] = read_file_content(fp)

    # ── PASS 1: RAW EXTRACT ──────────────────────────────────
    print("📝 Writing ice_raw_extract.md ...")
    with open(RAW_MD, "w", encoding="utf-8") as f:
        f.write("# ICE Codebase — Raw Extract\n\n")
        f.write("> Auto-generated by `generate_ice_docs.py`\n")
        f.write("> Code files: full content. Data files (.json/.jsonl/.txt etc): first 1 line (schema only). Binary files: name/path only.\n\n")
        f.write("---\n\n")

        for dir_path, files in structure:
            if not files:
                continue
            write_dir_header(f, dir_path, level=2)
            for fp in files:
                content, truncated = file_data[fp]
                write_file_block_raw(f, fp, content, truncated)

    print(f"  ✅ {RAW_MD.name}")

    # ── PASS 2: LLM SUMMARIES ────────────────────────────────
    print(f"\n🤖 Writing ice_llm_summary.md (calling {OLLAMA_MODEL}) ...")

    # Store summaries for pass 3
    summaries: dict[Path, str] = {}

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# ICE Codebase — LLM Technical Summaries\n\n")
        f.write(f"> Auto-generated by `generate_ice_docs.py` using `{OLLAMA_MODEL}`\n")
        f.write("> Each file summarized: purpose, functions, classes, variables, data flow, integrations.\n\n")
        f.write("---\n\n")

        processed = 0
        for dir_path, files in structure:
            if not files:
                continue
            write_dir_header(f, dir_path, level=2)
            for fp in files:
                processed += 1
                content, truncated = file_data[fp]
                print(f"  [{processed}/{total_files}] Summarizing {rel(fp)} ...", end=" ", flush=True)
                summary = llm_summarize(fp, content, truncated)
                summaries[fp] = summary
                write_file_block_summary(f, fp, summary)
                print("✅")

    print(f"  ✅ {SUMMARY_MD.name}")

    # ── PASS 3: FULL CONTEXT ─────────────────────────────────
    print("\n📚 Writing ice_full_context.md ...")
    with open(FULL_CONTEXT_MD, "w", encoding="utf-8") as f:
        f.write("# ICE Codebase — Full Context (Summary + Source)\n\n")
        f.write("> Auto-generated by `generate_ice_docs.py`\n")
        f.write("> Per file: LLM technical summary followed by full source code.\n\n")
        f.write("---\n\n")

        for dir_path, files in structure:
            if not files:
                continue
            write_dir_header(f, dir_path, level=2)
            for fp in files:
                content, truncated = file_data[fp]
                summary = summaries.get(fp, "[Summary not generated]")
                write_file_block_full(f, fp, content, truncated, summary)

    print(f"  ✅ {FULL_CONTEXT_MD.name}")

    # ── DONE ─────────────────────────────────────────────────
    print("\n" + "="*50)
    print("✅ All 3 docs generated:")
    for p in [RAW_MD, SUMMARY_MD, FULL_CONTEXT_MD]:
        size_kb = p.stat().st_size / 1024
        print(f"   {p.name:<30} {size_kb:>8.1f} KB")
    print("="*50)


if __name__ == "__main__":
    main()