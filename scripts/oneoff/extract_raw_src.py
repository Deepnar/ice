#!/usr/bin/env python3
"""
Generate RAW.md containing the complete contents of every file inside:

    /home/deepnar/Programs/ice/src

Output:
    /home/deepnar/Programs/ice/docs/RAW.md
"""

from pathlib import Path
import os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

SRC_ROOT = Path("/home/deepnar/Programs/ice/src")

OUTPUT_FILE = Path(
    "/home/deepnar/Programs/ice/docs/RAW_SRC.md"
)

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
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sql": "sql",
    ".sh": "bash",
    ".md": "markdown",
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def is_binary(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXTENSIONS


def language(path: Path) -> str:
    return LANGUAGE_MAP.get(
        path.suffix.lower(),
        "text",
    )


def collect_files(root: Path):
    files = []

    for dirpath, dirnames, filenames in os.walk(
        root,
        topdown=True,
    ):
        dirnames[:] = [
            d
            for d in sorted(dirnames)
            if d not in SKIP_DIRS
        ]

        for filename in sorted(filenames):
            path = Path(dirpath) / filename

            if is_binary(path):
                continue

            files.append(path)

    return files


def read_text(path: Path) -> str:
    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as e:
        return f"[ERROR READING FILE: {e}]"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = collect_files(SRC_ROOT)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as out:

        out.write("# ICE SRC RAW EXTRACT\n\n")
        out.write(
            f"Source Root: `{SRC_ROOT}`\n\n"
        )

        out.write(
            f"Total Files: **{len(files)}**\n\n"
        )

        out.write("---\n\n")

        for path in files:

            relative = path.relative_to(
                SRC_ROOT
            )

            out.write(
                f"## {relative}\n\n"
            )

            out.write(
                f"Path: `src/{relative}`\n\n"
            )

            out.write(
                f"```{language(path)}\n"
            )

            out.write(
                read_text(path)
            )

            out.write("\n```\n\n")
            out.write("---\n\n")

    print(
        f"Generated: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()