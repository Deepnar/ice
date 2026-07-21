#!/usr/bin/env python3
"""
ICE Project RAW Documentation Generator
========================================

Generates:

    RAW.md

Contains:
    - Folder/file structure
    - Full source code for code files
    - Schema previews for data files
    - Binary file placeholders

Usage:
    python generate_ice_docs.py

Run from:
    /home/deepnar/Programs/ice/

Output:
    /home/deepnar/Programs/ice/docs/RAW.md
"""

import os
import json
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

ICE_ROOT = Path("/home/deepnar/Programs/ice")

DOCS_DIR = ICE_ROOT / "docs"
RAW_MD = DOCS_DIR / "RAW.md"

# Data/config files: only preview first N lines
DATA_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".txt",
    ".csv",
    ".tsv",
    ".log",
    ".env",
}

DATA_PREVIEW_LINES = 1

# Dirs to skip traversal entirely
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "eggs",
    ".eggs",
    "docs",
}

# Files to skip entirely
ALWAYS_SKIP_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "ice_raw_extract.md",
    "ice_llm_summary.md",
    "ice_full_context.md",
    "generate_ice_docs.py",
}

# Binary / non-readable files
NON_READABLE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".egg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".svg",
    ".webp",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".bin",
    ".pkl",
    ".pt",
    ".safetensors",
    ".gguf",
    ".lock",
}

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────


def is_data_file(path: Path) -> bool:
    return path.suffix.lower() in DATA_EXTENSIONS


def is_non_readable(path: Path) -> bool:
    return path.suffix.lower() in NON_READABLE_EXTENSIONS


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ICE_ROOT))
    except ValueError:
        return str(path)


def _json_schema_preview(obj, depth: int, _cur: int = 0) -> str:
    indent = "  " * _cur

    if isinstance(obj, dict):

        if _cur >= depth:
            keys = list(obj.keys())
            return f"{{...}}  # keys: {keys}"

        lines = ["{"]

        for k, v in list(obj.items())[:20]:
            child = _json_schema_preview(v, depth, _cur + 1)
            lines.append(
                f"{indent}  {json.dumps(k)}: {child},"
            )

        if len(obj) > 20:
            lines.append(
                f"{indent}  ... ({len(obj)} keys total)"
            )

        lines.append(f"{indent}}}")

        return "\n".join(lines)

    elif isinstance(obj, list):

        if not obj:
            return "[]"

        if _cur >= depth:
            return (
                f"[...]  # {len(obj)} items, "
                f"type: {type(obj[0]).__name__}"
            )

        example = _json_schema_preview(
            obj[0],
            depth,
            _cur + 1,
        )

        return (
            f"[\n"
            f"{indent}  {example},\n"
            f"{indent}  ... ({len(obj)} items)\n"
            f"{indent}]"
        )

    else:

        if isinstance(obj, str) and len(obj) > 60:
            return f'"{obj[:57]}..."  # str'

        return f"{json.dumps(obj)}  # {type(obj).__name__}"


def read_file_content(path: Path) -> tuple[str, bool]:

    if is_non_readable(path):
        return "", False

    ext = path.suffix.lower()

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:
            raw = f.read()

    except Exception as e:
        return f"[ERROR reading file: {e}]", False

    # JSON
    if ext == ".json":

        try:
            parsed = json.loads(raw)
            schema = _json_schema_preview(parsed, depth=2)
            return schema, True

        except Exception:

            lines = raw.splitlines(keepends=True)

            if len(lines) > 30:
                return "".join(lines[:30]), True

            return raw, False

    # JSONL
    if ext == ".jsonl":

        first = raw.split("\n", 1)[0].strip()

        try:
            obj = json.loads(first)
            return json.dumps(obj, indent=2), True

        except Exception:
            return first, True

    # CSV / TSV
    if ext in {".csv", ".tsv"}:
        header = raw.split("\n", 1)[0]
        return header, True

    # TXT / LOG / ENV
    if ext in {".txt", ".log", ".env"}:
        first = raw.split("\n", 1)[0]
        return first, True

    # Generic data file
    if is_data_file(path):

        lines = raw.splitlines(keepends=True)

        if len(lines) > 1:
            return lines[0], True

        return raw, False

    # Code file
    return raw, False


def get_code_fence(path: Path) -> str:

    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "jsx",
        ".tsx": "tsx",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".json": "json",
        ".jsonl": "json",
        ".sql": "sql",
        ".md": "markdown",
        ".html": "html",
        ".css": "css",
        ".env": "bash",
        ".txt": "text",
        ".cfg": "ini",
        ".ini": "ini",
    }

    return ext_map.get(
        path.suffix.lower(),
        "text",
    )


def collect_files(root: Path):

    result = []

    for dirpath, dirnames, filenames in os.walk(
        root,
        topdown=True,
    ):

        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in SKIP_DIRS
            and not d.startswith(".")
        )

        dp = Path(dirpath)

        files = sorted(
            dp / f
            for f in filenames
            if f not in ALWAYS_SKIP_FILES
            and not f.startswith(".")
        )

        if files or dp == root:
            result.append((dp, files))

    return result


# ──────────────────────────────────────────────
# WRITERS
# ──────────────────────────────────────────────


def write_dir_header(
    f,
    dir_path: Path,
    level: int = 2,
):
    rel_dir = rel(dir_path)

    prefix = (
        "🗂️"
        if dir_path != ICE_ROOT
        else "🏠"
    )

    f.write(
        f"{'#' * level} "
        f"{prefix} `{dir_path.name}/`\n"
    )

    f.write(
        f"**Path:** `{rel_dir}/`\n\n"
    )


def write_file_block_raw(
    f,
    file_path: Path,
    content: str,
    truncated: bool,
):

    f.write(
        f"#### 📄 `{file_path.name}`\n"
    )

    f.write(
        f"**Path:** `{rel(file_path)}`\n\n"
    )

    if is_non_readable(file_path):

        size_str = ""

        try:
            size_kb = (
                file_path.stat().st_size / 1024
            )

            size_str = (
                f" ({size_kb:.1f} KB)"
            )

        except Exception:
            pass

        f.write(
            f"> 🚫 **Binary / non-readable file**"
            f"{size_str} — no raw content extracted.\n\n"
        )

    else:

        fence = get_code_fence(file_path)

        if truncated:
            f.write(
                f"> ⚠️ **Data file — schema preview only "
                f"(first {DATA_PREVIEW_LINES} line)**\n\n"
            )

        f.write(
            f"```{fence}\n"
            f"{content}\n"
            f"```\n\n"
        )

    f.write("---\n\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────


def main():

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"📂 ICE Root: {ICE_ROOT}")
    print(f"📁 Output:   {DOCS_DIR}\n")

    structure = collect_files(
        ICE_ROOT
    )

    total_files = sum(
        len(files)
        for _, files in structure
    )

    print(
        f"Found {total_files} files "
        f"across {len(structure)} directories.\n"
    )

    file_data = {}

    for _, files in structure:
        for fp in files:
            file_data[fp] = (
                read_file_content(fp)
            )

    print(
        "📝 Writing RAW.md ..."
    )

    with open(
        RAW_MD,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# ICE Codebase — Raw Extract\n\n"
        )

        f.write(
            "> Auto-generated by "
            "`generate_ice_docs.py`\n"
        )

        f.write(
            "> Code files: full content. "
            "Data files (.json/.jsonl/.txt etc): "
            "first 1 line (schema only). "
            "Binary files: name/path only.\n\n"
        )

        f.write("---\n\n")

        for dir_path, files in structure:

            if not files:
                continue

            write_dir_header(
                f,
                dir_path,
                level=2,
            )

            for fp in files:

                content, truncated = (
                    file_data[fp]
                )

                write_file_block_raw(
                    f,
                    fp,
                    content,
                    truncated,
                )

    print(
        f"  ✅ {RAW_MD.name}"
    )

    size_kb = (
        RAW_MD.stat().st_size / 1024
    )

    print("\n" + "=" * 50)
    print("✅ RAW document generated:")
    print(
        f"   {RAW_MD.name:<30}"
        f"{size_kb:>8.1f} KB"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()