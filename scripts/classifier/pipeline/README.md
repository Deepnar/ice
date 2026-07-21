# scripts/classifier/pipeline/ — the B1 (schema v2) classifier pipeline

The active, v2 rewrite of the classifier data+training flow. See the stage table + the
old→new mapping in `../README.md`.

**Status (2026-07-21):** directory created during the pre-B1 cleanup; the stage scripts
(`extract → stitch_icedev → synth → label → build → train → evaluate → promote`) are built
during B1 itself, each rewriting its `../legacy/` predecessor for schema v2, native-1024
input, cloud two-labeler labeling, and more-diverse online pulls. Until then this holds only
this note.
