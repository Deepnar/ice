#!/usr/bin/env python3
"""
Generate a Markdown document containing the
complete directory tree of:

    /home/deepnar/Programs/ice/src

Output:
    /home/deepnar/Programs/ice/docs/SRC_STRUCTURE.md
"""

from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

SRC_ROOT = Path("/home/deepnar/Programs/ice/src")

OUTPUT_FILE = Path(
    "/home/deepnar/Programs/ice/docs/SRC_STRUCTURE.md"
)

SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
}


# --------------------------------------------------
# TREE GENERATOR
# --------------------------------------------------

def build_tree(directory: Path, prefix=""):

    entries = sorted(
        [
            p
            for p in directory.iterdir()
            if p.name not in SKIP_DIRS
            and not p.name.startswith(".")
        ],
        key=lambda p: (
            not p.is_dir(),
            p.name.lower(),
        ),
    )

    lines = []

    for index, entry in enumerate(entries):

        is_last = index == len(entries) - 1

        connector = "└── " if is_last else "├── "

        lines.append(
            prefix + connector + entry.name
        )

        if entry.is_dir():

            extension = (
                "    "
                if is_last
                else "│   "
            )

            lines.extend(
                build_tree(
                    entry,
                    prefix + extension,
                )
            )

    return lines


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tree_lines = [
        SRC_ROOT.name + "/"
    ]

    tree_lines.extend(
        build_tree(SRC_ROOT)
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# ICE Source Structure\n\n"
        )

        f.write(
            f"Root: `{SRC_ROOT}`\n\n"
        )

        f.write("```text\n")

        f.write(
            "\n".join(tree_lines)
        )

        f.write("\n```\n")

    print(
        f"Generated: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()