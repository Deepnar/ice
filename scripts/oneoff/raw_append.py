#!/usr/bin/env python3
"""
Append selected scripts to RAW_SRC.md

Order:
1. scripts/classifier/promt_extraction/
2. scripts/classifier/promt_labeling/VLLM_label_dataset.py
3. scripts/training/
4. scripts/ner/

Appends to:
    docs/RAW_SRC.md
"""

from pathlib import Path
import os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT_ROOT = Path("/home/deepnar/Programs/ice")

OUTPUT_FILE = (
    PROJECT_ROOT
    / "docs"
    / "RAW_SRC.md"
)

TARGETS = [
    PROJECT_ROOT
    / "scripts"
    / "classifier"
    / "promt_extraction",

    PROJECT_ROOT
    / "scripts"
    / "classifier"
    / "promt_labeling"
    / "VLLM_label_dataset.py",

    PROJECT_ROOT
    / "scripts"
    / "training",

    PROJECT_ROOT
    / "scripts"
    / "ner",
]

SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pkl",
    ".pt",
    ".bin",
    ".so",
    ".dll",
    ".pyc",
}

LANGUAGE_MAP = {
    ".py": "python",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sql": "sql",
    ".sh": "bash",
    ".md": "markdown",
    ".txt": "text",
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def is_binary(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXTENSIONS


def get_language(path: Path) -> str:
    return LANGUAGE_MAP.get(
        path.suffix.lower(),
        "text",
    )


def read_file(path: Path) -> str:
    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as e:
        return f"[ERROR READING FILE: {e}]"


def collect_folder_files(folder: Path):

    files = []

    for dirpath, dirnames, filenames in os.walk(
        folder,
        topdown=True,
    ):
        dirnames[:] = [
            d
            for d in sorted(dirnames)
            if d not in SKIP_DIRS
            and not d.startswith(".")
        ]

        for filename in sorted(filenames):

            path = Path(dirpath) / filename

            if is_binary(path):
                continue

            files.append(path)

    return files


def write_file_block(out, file_path: Path):

    relative = file_path.relative_to(
        PROJECT_ROOT
    )

    out.write(
        f"## {relative}\n\n"
    )

    out.write(
        f"Path: `{relative}`\n\n"
    )

    out.write(
        f"```{get_language(file_path)}\n"
    )

    out.write(
        read_file(file_path)
    )

    out.write(
        "\n```\n\n"
    )

    out.write(
        "---\n\n"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8",
    ) as out:

        out.write(
            "\n\n# Additional Script Extraction\n\n"
        )

        for target in TARGETS:

            print(
                f"Processing: {target}"
            )

            if not target.exists():

                out.write(
                    f"## MISSING: {target}\n\n"
                )

                continue

            # --------------------------
            # SINGLE FILE
            # --------------------------

            if target.is_file():

                out.write(
                    f"# FILE: "
                    f"{target.relative_to(PROJECT_ROOT)}\n\n"
                )

                write_file_block(
                    out,
                    target,
                )

                continue

            # --------------------------
            # DIRECTORY
            # --------------------------

            out.write(
                f"# DIRECTORY: "
                f"{target.relative_to(PROJECT_ROOT)}\n\n"
            )

            files = collect_folder_files(
                target
            )

            for file_path in files:
                write_file_block(
                    out,
                    file_path,
                )

    print(
        f"\nDone. Appended to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()