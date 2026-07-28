# F-track design brief (pre-work for the frontend design conversation)

**This is a BRIEF, not a spec.** Track F's preamble mandates a dedicated design
conversation with the user before anything is built; this document is what that
conversation starts from — the constraints, the owed surfaces, the options with
a recommendation, and the exact questions. Written S1 2026-07-10.

## What the frontend IS (settled by other specs — not up for redesign)

- **A REST adapter over E0's services.** Every operation the UI performs already
  exists as a service function (slots, bookmarks, scoping, clusters, review
  apply-dispatch, registry, graph view/edit/timeline/diff, retrieval preview,
  deletion manifests, import, projects/decisions/tasks). F builds NO business
  logic — if a screen needs an operation that doesn't exist, the fix is a
  service first (E0 discipline).
- **The packaged app boots a SIMPLE stack now** (C7 world): docker postgres +
  one core process (FastAPI + in-process maintenance runtime). No redis, no
  celery, no vLLM by default. "Open the app → stack up" is: ensure docker pg,
  start the core, open the window. E7's headless path coexists via the runtime
  lease — the app and `ice-mcp` can run simultaneously.
- **Telemetry exists to be surfaced:** structlog events already named for F5 —
  `memory_decision` breakdown, `codex_*` attribution events, `timescope_detected`,
  `maintenance_job_*`, `agent_action`/`agent_run_summary`, `mcp_tool_call`,
  `chat_command`, `import_progress`. F5 is largely "promote these to SSE and
  render," plus the G20 note about the dead redis pub/sub being replaced by an
  in-process SSE feed.

## The owed-surfaces ledger (consolidated; every one must appear in the design)

C6: sidebar scope toggles + @-mention + incognito indicator · C9: slot editing
at three tiers · C11: command discoverability (/help, autocomplete) · C12:
document upload + add-doc-to-scope — **expanded 2026-07-28, see the ROADMAP
ledger**: a global document library with a per-conversation **live enable/
disable toggle** (not an attachment), the **doc-vs-transcript choice** at
upload/paste, a promotion indicator (2nd chat to enable a doc makes its
knowledge global, one-way), and the incognito refusal · **F10/F14 import
wizard** (export → adapter pick, or plain `.txt` → F14's amnesia slicer, both
into LSREP replay; dry-run estimate + progress — backend fully shipped) ·
**C12/D10 OCR engine** (backend: `settings.document_extraction_engine`
dispatch, Tika/Docling as opt-in containers, no system binary) · E1/E2: projects UI (register/attach/mode
toggle) · E4: welcome-back block render · E8: architecture-doc view (rendered
`render_architecture_doc`) · G23: export/backup buttons · C10: delete-with-
manifest confirm dialog · F2 review queue (approve/reject over the D6 dispatch)
· F3 graph view (typed nodes, backlinks, negated edges dashed, `entity_diff`
timeline overlay, description-only editing) · F4 full settings (the Z1-prep
knob inventory IS the settings catalog, grouped by stage) · F5 telemetry · F6
select-text→pin · F9 feedback (thumbs → CuratedLabel schema-v2 + B3 rewards) ·
F13 replay/branching (with B6) · F15 hardware advisor.

## Options + recommendation (user-owned — the conversation's main question)

1. **Web app served by the core + Tauri wrapper later (RECOMMENDED).** SvelteKit
   or React SPA talking to the REST adapters + SSE; ships first as
   `localhost:8000` (works today, zero packaging risk), wraps into Tauri for
   the installed-app feel when F matures. Pros: one codebase, packaging
   deferred until the UI is proven, SSE trivially. Cons: two runtimes at
   package time (webview + python core) — accepted; every local-AI app does this.
2. **Electron.** Heavier, same architecture, worse footprint — only if Tauri's
   webview quirks bite.
3. **Native (Qt/pyside).** Best integration, slowest iteration for an
   opinion-heavy consolidation with a graph view — not recommended.

## Questions for the user (the conversation agenda)

1. Stack pick (above) + any strong UI-framework preference.
2. Layout: chat-centered with memory as side panels, or a "memory workbench"
   with chat as one tab? (Recommendation: chat-centered; memory surfaces as a
   right dock — review queue, retrieval attribution, scope — because daily use
   is chat.)
3. Priority order — recommendation: **F1 foundation → F2 review queue (rotting
   proposals become visible) → F5 telemetry (trust through visibility) → F4
   settings → F3 graph view → C12 upload + F10 import UI → F9 feedback → F6 →
   F15 → F13/B6.**
4. The incognito visual language (G16) and how loud constraint warnings (E8)
   should be.
5. Whether F7 (web search) and F8 (deep research) enter this cycle at all —
   both are integrations more than UI (and B1's `Needs_Live_Info` routes
   nowhere until F7).

## Non-goals (already decided elsewhere)

No frontend-hosted coding session (E5 — recorded later option). No second
memory API surface. No UI-side retrieval logic (retrieval_svc previews only).
`./ice`/`stop_ice`/`setup.sh` die at packaging (F preamble; C7 already shrank
them).
