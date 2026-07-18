"""E1b — the code graph: deterministic Python `ast` extraction into the ONE
codex, namespaced (spec D2/D3).

Entities are POINTERS, never bodies: canonical name
``"{project_slug}:{module}.{qualname}"`` (casefolded — the codex convention;
display name in properties), with ``file_path``, ``line_start/end``,
``signature``, and the docstring FIRST LINE only. ``source='static_analysis'``
rows are derived memory: decay-exempt, journal-free (NO CodexEvents — a
re-parse writing thousands of events would bury T-track's signal), excluded
from conversational retrieval unless the query's project matches, and
bulk-rebuild safe. ``aliases`` stay empty so conversational
``get_or_create_entity`` can never re-attach a chat mention to a code row.

Edges: ``imports`` (module → module, project-internal) and ``calls``
(function/method → callee), resolved best-effort statically — same-module
names, imported names, ``self.``-methods, ``module.attr`` on imported project
modules — and tagged via ``extraction_confidence`` (1.0 resolved / 0.6
heuristic; codex_edges has no properties column — spec rev 3). All static
edges share the deterministic per-project batch id (rev 5), which is what
``_codex_scope_sets`` admits for project-scoped retrieval.

Other languages arrive later behind the same ``CodeExtractor`` seam
(tree-sitter candidate); only Python ships now — ICE itself is the first
project.
"""
import ast
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog
from sqlalchemy import text

from src.memory.models import CodexEdge, CodexEntity

logger = structlog.get_logger("ice.coding.code_graph")

SOURCE_STATIC = "static_analysis"
RESOLVED_CONFIDENCE = 1.0
HEURISTIC_CONFIDENCE = 0.6
# Spec §4: single-user laptop reality — bootstrap refuses to walk past this.
MAX_FILES = 20_000
# Spec §4 default ignore set; per-project additions via projects.settings["ignore"].
DEFAULT_IGNORES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", ".eggs", "dist", "build",
    "models", "data", "logs",
}


def code_graph_batch_id(project_id) -> uuid.UUID:
    """The deterministic source_batch for a project's static edges — and the
    'code-graph allowance' _codex_scope_sets grants project-scoped queries."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"ice:code-graph:{project_id}")


def canonical_for(slug: str, module: str, qualname: str = "") -> str:
    base = f"{slug}:{module}" if not qualname else f"{slug}:{module}.{qualname}"
    return base.casefold()


@dataclass
class ParsedSymbol:
    qualname: str            # "" for the module itself; "Cls.method" for methods
    kind: str                # module | class | function
    line_start: int
    line_end: int
    signature: str
    doc_first_line: str


@dataclass
class ParsedFile:
    module: str              # dotted module path relative to its root
    rel_path: str            # path relative to the project root (display/pointer)
    symbols: list = field(default_factory=list)          # [ParsedSymbol]
    import_modules: set = field(default_factory=set)     # dotted module names imported
    imported_names: dict = field(default_factory=dict)   # local name -> "module.qualname" or "module"
    calls: list = field(default_factory=list)            # [(caller_qualname, callee_local_ref)]
    parse_error: Optional[str] = None


def _signature(node) -> str:
    """Compact def/class signature from the AST — no body text ever."""
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(b) for b in node.bases]
        return f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) else ""
    return f"{prefix} {node.name}({args}){ret}"


def _doc_first_line(node) -> str:
    doc = ast.get_docstring(node, clean=True)
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


class _FileVisitor(ast.NodeVisitor):
    """One pass: symbols (classes/functions with pointers), imports, and raw
    call references attributed to the enclosing function/method."""

    def __init__(self, pf: ParsedFile):
        self.pf = pf
        self._stack: list = []   # enclosing (kind, name) frames

    def _qualname(self, name: str) -> str:
        return ".".join([n for _, n in self._stack] + [name])

    def _caller(self) -> Optional[str]:
        """Qualname of the innermost enclosing function/method (a call at
        module or class-body level has no caller entity)."""
        last_fn = None
        for i, (kind, _) in enumerate(self._stack):
            if kind == "function":
                last_fn = i
        if last_fn is None:
            return None
        return ".".join(name for _, name in self._stack[: last_fn + 1])

    def visit_ClassDef(self, node):
        q = self._qualname(node.name)
        self.pf.symbols.append(ParsedSymbol(
            q, "class", node.lineno, node.end_lineno or node.lineno,
            _signature(node), _doc_first_line(node)))
        self._stack.append(("class", node.name))
        self.generic_visit(node)
        self._stack.pop()

    def _visit_fn(self, node):
        q = self._qualname(node.name)
        self.pf.symbols.append(ParsedSymbol(
            q, "function", node.lineno, node.end_lineno or node.lineno,
            _signature(node), _doc_first_line(node)))
        self._stack.append(("function", node.name))
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _visit_fn
    visit_AsyncFunctionDef = _visit_fn

    def visit_Import(self, node):
        for alias in node.names:
            self.pf.import_modules.add(alias.name)
            self.pf.imported_names[alias.asname or alias.name.split(".")[0]] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level:  # relative import → resolve against this module's package
            pkg_parts = self.pf.module.split(".")[: -node.level] if node.level <= len(
                self.pf.module.split(".")) else []
            base = ".".join(pkg_parts + ([node.module] if node.module else []))
        else:
            base = node.module or ""
        if base:
            self.pf.import_modules.add(base)
            for alias in node.names:
                if alias.name != "*":
                    self.pf.imported_names[alias.asname or alias.name] = f"{base}.{alias.name}"
        self.generic_visit(node)

    @staticmethod
    def _dotted(f) -> Optional[str]:
        """A pure Name/Attribute chain as a dotted string; None otherwise."""
        parts = []
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if isinstance(f, ast.Name):
            parts.append(f.id)
            return ".".join(reversed(parts))
        return None

    def visit_Call(self, node):
        caller = self._caller()
        if caller is not None:
            ref = self._dotted(node.func)
            if ref:
                self.pf.calls.append((caller, ref))
        self.generic_visit(node)


class CodeExtractor:
    """Full + incremental parse for one project (the language seam — a future
    tree-sitter extractor implements the same iter/parse/sync surface)."""

    def __init__(self, project):
        self.project = project
        self.slug = project.slug
        self.roots = [Path(r) for r in (project.roots or [])]
        extra = set((project.settings or {}).get("ignore", []))
        self.ignores = DEFAULT_IGNORES | extra
        self.batch_id = code_graph_batch_id(project.id)

    # ── Discovery / parsing ─────────────────────────────────────────────

    def _root_for(self, rel_path: str) -> Optional[Path]:
        for root in self.roots:
            if (root / rel_path).exists():
                return root
        return self.roots[0] if self.roots else None

    def iter_python_files(self) -> list:
        """Relative .py paths across all roots, ignore-filtered, capped."""
        out = []
        for root in self.roots:
            if not root.is_dir():
                continue
            for p in sorted(root.rglob("*.py")):
                rel = p.relative_to(root)
                if any(part in self.ignores for part in rel.parts):
                    continue
                out.append(str(rel))
                if len(out) >= MAX_FILES:
                    logger.warning("code_graph_file_cap", cap=MAX_FILES,
                                   project=self.slug)
                    return out
        return out

    def module_name(self, rel_path: str) -> str:
        parts = list(Path(rel_path).parts)
        parts[-1] = parts[-1][:-3]  # strip .py
        if parts[-1] == "__init__":
            parts = parts[:-1] or ["__init__"]
        return ".".join(parts)

    def parse_file(self, rel_path: str) -> Optional[ParsedFile]:
        """Parse the working tree as-is (the map must match what the agent
        sees). Returns None when the file is gone (treated as a deletion by
        sync); a syntax error returns a ParsedFile with parse_error set."""
        root = self._root_for(rel_path)
        path = root / rel_path if root else None
        if path is None or not path.is_file():
            return None
        pf = ParsedFile(module=self.module_name(rel_path), rel_path=rel_path)
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            pf.parse_error = f"line {exc.lineno}: {exc.msg}"
            return pf
        pf.symbols.append(ParsedSymbol(
            "", "module", 1, len(source.splitlines()) or 1,
            f"module {pf.module}", _doc_first_line(tree)))
        _FileVisitor(pf).visit(tree)
        return pf

    # ── Sync (delete-and-recreate with canonical-stable updates) ─────────

    def _file_entities(self, db, rel_path: str) -> dict:
        rows = db.query(CodexEntity).filter(
            CodexEntity.project_id == self.project.id,
            CodexEntity.source == SOURCE_STATIC,
            CodexEntity.properties["file_path"].astext == rel_path,
        ).all()
        return {e.canonical_name: e for e in rows}

    def _entity_payload(self, sym: ParsedSymbol, rel_path: str) -> str:
        doc = f" — {sym.doc_first_line}" if sym.doc_first_line else ""
        return f"{sym.signature}{doc}  ({rel_path}:{sym.line_start})"

    def _upsert_symbol(self, db, pf: ParsedFile, sym: ParsedSymbol,
                       existing: dict, stats: dict) -> CodexEntity:
        display = f"{pf.module}.{sym.qualname}" if sym.qualname else pf.module
        canonical = canonical_for(self.slug, pf.module, sym.qualname)
        props = {
            "display_name": display,
            "kind": sym.kind,
            "file_path": pf.rel_path,
            "line_start": sym.line_start,
            "line_end": sym.line_end,
            "signature": sym.signature,
            "docstring_summary": sym.doc_first_line,
            "project_slug": self.slug,
        }
        ent = existing.pop(canonical, None)
        if ent is None:
            # A stale row under another file path (file move) — same canonical
            # must not violate the unique constraint.
            ent = db.query(CodexEntity).filter_by(canonical_name=canonical).first()
        if ent is None:
            ent = CodexEntity(
                id=uuid.uuid5(uuid.NAMESPACE_URL, f"ice:code:{canonical}"),
                canonical_name=canonical, aliases=[], tags=[],
                entity_type=sym.kind, description="",
                properties=props,
                context_payload=self._entity_payload(sym, pf.rel_path),
                embedding=None,             # rev 9: code entities skip embeddings
                project_id=self.project.id, source=SOURCE_STATIC,
                last_updated=datetime.now(timezone.utc),
            )
            db.add(ent)
            stats["created"] += 1
        else:
            ent.entity_type = sym.kind
            ent.properties = props
            ent.context_payload = self._entity_payload(sym, pf.rel_path)
            ent.project_id = self.project.id
            ent.source = SOURCE_STATIC
            ent.last_updated = datetime.now(timezone.utc)
            stats["updated"] += 1
        return ent

    def _delete_entities(self, db, entities, stats: dict) -> None:
        for ent in entities:
            db.query(CodexEdge).filter(
                (CodexEdge.source_id == ent.id) | (CodexEdge.target_id == ent.id)
            ).delete(synchronize_session=False)
            db.delete(ent)
            stats["deleted"] += 1

    def sync_files(self, db, rel_paths: list) -> dict:
        """Incremental delete-and-recreate for the given files (spec D5 step 1):
        vanished symbols (and vanished files) are hard-deleted with their
        edges — derived memory is regenerable, not history; surviving
        canonicals update in place so ids (and inbound cross-file edges)
        stay stable. Entities flush BEFORE edges (FK ordering)."""
        stats = {"files": 0, "created": 0, "updated": 0, "deleted": 0,
                 "edges": 0, "resolved": 0, "heuristic": 0, "parse_errors": 0}
        # E11 D4: at most one reparse of a project at a time, cross-process
        # (app + ice-mcp) — a read-freshen racing a commit-reconcile would
        # collide on the deterministic entity ids. Every writer path (freshen,
        # incremental reconcile, bootstrap full_sync) funnels through here;
        # the xact lock releases at this method's commit.
        db.execute(text("SELECT pg_advisory_xact_lock(hashtext(:pid))"),
                   {"pid": str(self.project.id)})
        parsed: list = []
        for rel in dict.fromkeys(rel_paths):        # de-dupe, keep order
            existing = self._file_entities(db, rel)
            pf = self.parse_file(rel)
            if pf is None:                          # file deleted
                self._delete_entities(db, existing.values(), stats)
                continue
            stats["files"] += 1
            if pf.parse_error:
                # Rev 12: keep the last good map (a broken working tree is
                # usually mid-edit); mark the module unit as the parse_error
                # pointer. No symbol deletions, no edge rebuild.
                stats["parse_errors"] += 1
                module_canonical = canonical_for(self.slug, pf.module)
                ent = existing.get(module_canonical)
                if ent is None:
                    ent = self._upsert_symbol(db, pf, ParsedSymbol(
                        "", "module", 1, 1, f"module {pf.module}", ""),
                        existing={}, stats=stats)
                ent.properties = {**(ent.properties or {}),
                                  "parse_error": pf.parse_error}
                ent.last_updated = datetime.now(timezone.utc)
                logger.warning("code_graph_parse_error", file=rel,
                               error=pf.parse_error, project=self.slug)
                continue
            entity_map = {}
            for sym in pf.symbols:
                ent = self._upsert_symbol(db, pf, sym, existing, stats)
                entity_map[sym.qualname] = ent
            # leftovers under this file = symbols that vanished from it
            self._delete_entities(db, existing.values(), stats)
            parsed.append((pf, entity_map))
        db.flush()   # entities BEFORE edges — no relationship() to order FKs

        for pf, entity_map in parsed:
            self._rebuild_edges(db, pf, entity_map, stats)
        db.commit()
        logger.info("code_graph_sync", project=self.slug, **stats)
        return stats

    def full_sync(self, db) -> dict:
        """Bootstrap / force-push fallback: sync every discoverable file, then
        drop derived rows whose file no longer exists."""
        files = self.iter_python_files()
        live = set(files)
        stats = self.sync_files(db, files)
        stale = db.query(CodexEntity).filter(
            CodexEntity.project_id == self.project.id,
            CodexEntity.source == SOURCE_STATIC,
        ).all()
        gone = [e for e in stale
                if (e.properties or {}).get("file_path") not in live]
        if gone:
            self._delete_entities(db, gone, stats)
            db.commit()
        total = stats["resolved"] + stats["heuristic"]
        ratio = (stats["heuristic"] / total) if total else 0.0
        # Empirical deferral (spec rule 2b): heuristic >60% on ICE itself ⇒
        # tighten to imports-only. Logged, not enforced.
        logger.info("code_graph_bootstrap", project=self.slug,
                    heuristic_ratio=round(ratio, 3), **stats)
        return stats

    # ── Edge resolution ──────────────────────────────────────────────────

    def _lookup(self, db, canonical: str) -> Optional[CodexEntity]:
        return db.query(CodexEntity).filter(
            CodexEntity.canonical_name == canonical,
            CodexEntity.source == SOURCE_STATIC,
        ).first()

    def _resolve_call(self, db, pf: ParsedFile, entity_map: dict,
                      caller: str, ref: str):
        """Best-effort static resolution (spec D2 — no type inference).
        Returns (target_entity, confidence) or None."""
        # self.method() → the caller's class
        if ref.startswith("self."):
            method = ref[5:]
            cls = caller.rsplit(".", 1)[0] if "." in caller else None
            if cls:
                target = entity_map.get(f"{cls}.{method}") or self._lookup(
                    db, canonical_for(self.slug, pf.module, f"{cls}.{method}"))
                if target is not None:
                    return target, RESOLVED_CONFIDENCE
            return None
        # same-module qualname (foo, Cls.method)
        target = entity_map.get(ref)
        if target is not None:
            return target, RESOLVED_CONFIDENCE
        if "." in ref:
            # imported prefix + attribute — alias map ("from a import C" /
            # "import a.b as c") or a full "import a.b" module path. Note
            # canonical_for(slug, "a.b", "c") == canonical_for(slug, "a.b.c"),
            # so a resolved prefix concatenates straight into the canonical.
            head, attr = ref.rsplit(".", 1)
            prefix = pf.imported_names.get(head)
            if prefix is None and head in pf.import_modules:
                prefix = head
            if prefix:
                target = self._lookup(
                    db, canonical_for(self.slug, f"{prefix}.{attr}"))
                if target is not None:
                    return target, RESOLVED_CONFIDENCE
            return None
        imported = pf.imported_names.get(ref)
        if imported:
            target = self._lookup(db, canonical_for(self.slug, imported))
            if target is not None:
                return target, RESOLVED_CONFIDENCE
        # heuristic: a unique function of that name anywhere in the project
        # (Postgres LIKE escapes with backslash by default — underscores in
        # identifiers must not act as wildcards)
        pattern = ref.casefold().replace("\\", "\\\\").replace("_", r"\_")
        rows = db.query(CodexEntity).filter(
            CodexEntity.project_id == self.project.id,
            CodexEntity.source == SOURCE_STATIC,
            CodexEntity.entity_type == "function",
            CodexEntity.canonical_name.like(f"%.{pattern}"),
        ).limit(2).all()
        if len(rows) == 1:
            return rows[0], HEURISTIC_CONFIDENCE
        return None

    def _add_edge(self, db, src: CodexEntity, tgt: CodexEntity, relation: str,
                  confidence: float, seen: set, stats: dict) -> None:
        key = (src.id, tgt.id, relation)
        if src.id == tgt.id or key in seen:
            return
        seen.add(key)
        db.add(CodexEdge(
            source_id=src.id, target_id=tgt.id, relation=relation,
            strength=1.0, source_batch=self.batch_id, confidence="active",
            extraction_confidence=confidence, negated=False,
            valid_from=datetime.now(timezone.utc), source=SOURCE_STATIC,
        ))
        stats["edges"] += 1
        stats["resolved" if confidence >= RESOLVED_CONFIDENCE else "heuristic"] += 1

    def _rebuild_edges(self, db, pf: ParsedFile, entity_map: dict,
                       stats: dict) -> None:
        module_ent = entity_map.get("")
        if module_ent is None:
            return
        # outgoing static edges of this file's entities are fully rebuilt
        ids = [e.id for e in entity_map.values()]
        db.query(CodexEdge).filter(
            CodexEdge.source_id.in_(ids),
            CodexEdge.source == SOURCE_STATIC,
        ).delete(synchronize_session=False)
        seen: set = set()
        for mod in sorted(pf.import_modules):
            target = self._lookup(db, canonical_for(self.slug, mod))
            if target is not None:
                self._add_edge(db, module_ent, target, "imports",
                               RESOLVED_CONFIDENCE, seen, stats)
        for caller, ref in pf.calls:
            src = entity_map.get(caller)
            if src is None:
                continue
            resolved = self._resolve_call(db, pf, entity_map, caller, ref)
            if resolved is None:
                continue
            target, conf = resolved
            self._add_edge(db, src, target, "calls", conf, seen, stats)


def where_symbol_rows(db, symbol: str, project_id=None) -> list:
    """ice_where's E1b engine: resolve a symbol against the code graph —
    exact canonical, then qualname-suffix match. Results carry project + path;
    an attached project's rows rank first (spec §4)."""
    sym = (symbol or "").strip().casefold()
    if not sym:
        return []
    like_escaped = sym.replace("%", r"\%").replace("_", r"\_")
    rows = db.execute(text("""
        SELECT id, canonical_name, entity_type, properties, project_id
        FROM codex_entities
        WHERE source = :src
          AND (canonical_name = :sym
               OR canonical_name LIKE :dot_suffix
               OR canonical_name LIKE :colon_suffix)
        ORDER BY length(canonical_name) ASC
        LIMIT 10
    """), {"src": SOURCE_STATIC, "sym": sym,
           "dot_suffix": f"%.{like_escaped}",
           "colon_suffix": f"%:{like_escaped}"}).fetchall()
    out = [
        {
            "canonical_name": r.canonical_name,
            "name": (r.properties or {}).get("display_name", r.canonical_name),
            "kind": (r.properties or {}).get("kind", r.entity_type),
            "project": (r.properties or {}).get("project_slug"),
            "file": (r.properties or {}).get("file_path"),
            "line_start": (r.properties or {}).get("line_start"),
            "line_end": (r.properties or {}).get("line_end"),
            "signature": (r.properties or {}).get("signature"),
            "doc": (r.properties or {}).get("docstring_summary"),
        }
        for r in rows
    ]
    if project_id is not None:
        pid = str(project_id)
        out.sort(key=lambda m: 0 if any(
            str(r.project_id) == pid and r.canonical_name == m["canonical_name"]
            for r in rows) else 1)
    return out
