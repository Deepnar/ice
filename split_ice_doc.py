#!/usr/bin/env python3
"""
ICE Docs Splitter
==================
Splits ice_raw_extract.md and ice_llm_summary.md into multiple files,
each capped at ~TOKEN_LIMIT tokens. Cuts are ONLY made at section
boundaries (--- separators), never mid-code or mid-section.

Output: /home/deepnar/Programs/ice/docs/chunks/
Files:  raw_1.md, raw_2.md, ...  /  summary_1.md, summary_2.md, ...

Usage: python split_ice_docs.py
"""

from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DOCS_DIR    = Path("/home/deepnar/Programs/ice/docs")
CHUNKS_DIR  = DOCS_DIR / "chunks"

RAW_MD      = DOCS_DIR / "RAW.md"

# Target tokens per chunk. 1 token ≈ 4 chars (GPT/Claude tokenizer rough ratio).
# 8000 tokens ≈ 32000 chars — good for most context windows as a single chunk.
TOKEN_LIMIT = 8000
CHARS_PER_TOKEN = 4

# The separator used between file sections in the generated docs
SECTION_SEPARATOR = "\n---\n"

# ──────────────────────────────────────────────
# CORE LOGIC
# ──────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def split_into_sections(content: str) -> list[str]:
    """
    Split markdown content into sections at '---' boundaries.
    Each section includes its trailing '---' so chunks reassemble cleanly.
    The file header (before first ---) is treated as section 0.
    """
    # Split on the separator pattern
    raw_parts = content.split(SECTION_SEPARATOR)

    sections = []
    for i, part in enumerate(raw_parts):
        part = part.strip()
        if not part:
            continue
        # Re-attach separator at end (except for last part if file doesn't end with ---)
        if i < len(raw_parts) - 1:
            sections.append(part + SECTION_SEPARATOR)
        else:
            sections.append(part + "\n")

    return sections


def chunk_sections(sections: list[str], token_limit: int) -> list[list[str]]:
    """
    Greedily bin sections into chunks, each under token_limit tokens.
    If a single section exceeds token_limit on its own, it gets its own chunk
    (we never split within a section).
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for section in sections:
        section_tokens = estimate_tokens(section)

        if current_tokens + section_tokens > token_limit and current:
            # Flush current chunk, start new one
            chunks.append(current)
            current = [section]
            current_tokens = section_tokens
        else:
            current.append(section)
            current_tokens += section_tokens

    if current:
        chunks.append(current)

    return chunks


def write_chunks(chunks: list[list[str]], prefix: str, source_file: Path) -> list[Path]:
    """
    Write chunks to CHUNKS_DIR as {prefix}_1.md, {prefix}_2.md, etc.
    Each file gets a small header noting which part it is.
    Returns list of written paths.
    """
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(chunks)
    written = []

    for i, sections in enumerate(chunks, start=1):
        out_path = CHUNKS_DIR / f"{prefix}_{i}.md"
        content = "".join(sections)
        token_est = estimate_tokens(content)

        header = (
            f"> **{prefix.replace('_', ' ').title()} — Part {i} of {total}**  \n"
            f"> Source: `{source_file.name}` | ~{token_est:,} tokens in this chunk\n\n"
            f"---\n\n"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header + content)

        written.append(out_path)
        print(f"  ✅ {out_path.name:<30} {len(content)//1024:>4} KB   ~{token_est:>6,} tokens   ({len(sections)} sections)")

    return written


def process_file(md_path: Path, prefix: str) -> None:
    if not md_path.exists():
        print(f"  ⚠️  {md_path.name} not found — skipping.")
        return

    print(f"\n📄 Processing {md_path.name} ...")
    content = md_path.read_text(encoding="utf-8")

    total_tokens = estimate_tokens(content)
    print(f"   Total size: {len(content)//1024} KB  ~{total_tokens:,} tokens")

    sections = split_into_sections(content)
    print(f"   Sections found: {len(sections)}")

    chunks = chunk_sections(sections, TOKEN_LIMIT)
    print(f"   Chunks to write: {len(chunks)} (target ≤{TOKEN_LIMIT:,} tokens each)\n")

    write_chunks(chunks, prefix, md_path)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print(f"📂 Docs dir:   {DOCS_DIR}")
    print(f"📁 Output dir: {CHUNKS_DIR}")
    print(f"🎯 Token limit per chunk: {TOKEN_LIMIT:,} (~{TOKEN_LIMIT * CHARS_PER_TOKEN:,} chars)\n")

    process_file(RAW_MD,     prefix="raw")

    # Summary
    all_chunks = list(CHUNKS_DIR.glob("*.md"))
    print(f"\n{'='*55}")
    print(f"✅ Done. {len(all_chunks)} chunk files in {CHUNKS_DIR}/")
    raw_chunks     = sorted(CHUNKS_DIR.glob("raw_*.md"))
    summary_chunks = sorted(CHUNKS_DIR.glob("summary_*.md"))
    print(f"   raw_*.md     : {len(raw_chunks)} files")
    print(f"   summary_*.md : {len(summary_chunks)} files")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()