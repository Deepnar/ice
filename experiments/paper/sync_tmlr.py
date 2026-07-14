"""Keep ICE_paper_tmlr.tex's body in sync with ICE_paper.tex.

The two papers share an identical body (everything from \\begin{abstract}
onward); only the preamble/title/author differ. ICE_paper.tex is the source of
truth for content. After editing it, run this to regenerate the TMLR version:

    uv run python sync_tmlr.py     (or: python3 sync_tmlr.py)

It preserves the EXISTING TMLR preamble (so any TMLR-specific tweaks — author
block, [accepted] toggle — persist) and only re-copies the body from the source.
Idempotent: running it when already in sync changes nothing.
"""
import pathlib

HERE = pathlib.Path(__file__).parent
SRC = HERE / "ICE_paper.tex"
TMLR = HERE / "ICE_paper_tmlr.tex"
MARK = r"\begin{abstract}"


def main():
    src = SRC.read_text()
    tmlr = TMLR.read_text()
    body = src[src.index(MARK):]                 # abstract → end (the shared content)
    tmlr_preamble = tmlr[:tmlr.index(MARK)]      # keep the TMLR wrapper as-is
    new = tmlr_preamble + body
    if new == tmlr:
        print("already in sync — no change")
    else:
        TMLR.write_text(new)
        print(f"synced: TMLR body updated from ICE_paper.tex ({len(new.splitlines())} lines)")


if __name__ == "__main__":
    main()
