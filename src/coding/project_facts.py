"""E9 — project facts as derived codex entities, pointer-first (spec D9; E6
settled here: typed knowledge units in our tables ARE the OKF adaptation).

Six deterministic parsers — deps+versions, DB schema, run/test/build
commands, config surface (KEY NAMES, never values), infrastructure, declared
data shapes — each yielding ONE ``codex_entities`` row:
``entity_type='project_fact'``, ``source='derived'``, structured
``properties`` + a human ``description``, file pointers. Re-derived by the
reconciler (D5 step 2) hash-gated: unchanged source files ⇒ no write.
No LLM anywhere; missing files simply yield no unit.
"""
import ast
import hashlib
import json
import re
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from src.memory.models import CodexEntity

logger = structlog.get_logger("ice.coding.project_facts")

SOURCE_DERIVED = "derived"
FACT_TYPE = "project_fact"
_IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
                "models", "data", "logs", "dist", "build"}


def fact_canonical(slug: str, kind: str) -> str:
    return f"{slug}:fact:{kind}".casefold()


# ── Parsers: (data, description, source_files) or None ──────────────────────

def _read(root: Path, rel: str) -> Optional[str]:
    p = root / rel
    try:
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
    except OSError:
        return None


def _parse_dependencies(root: Path):
    data, files = {}, []
    py = _read(root, "pyproject.toml")
    if py is not None:
        files.append("pyproject.toml")
        try:
            doc = tomllib.loads(py)
            deps = doc.get("project", {}).get("dependencies", []) or []
            data["python"] = {"declared": deps}
            lock = _read(root, "uv.lock")
            if lock is not None:
                files.append("uv.lock")
                try:
                    locked = tomllib.loads(lock)
                    tops = {re.split(r"[<>=!~\[; ]", d, 1)[0].strip().lower()
                            for d in deps}
                    versions = {p["name"]: p.get("version")
                                for p in locked.get("package", [])
                                if p.get("name", "").lower() in tops}
                    data["python"]["resolved"] = versions
                except (tomllib.TOMLDecodeError, KeyError, TypeError):
                    pass
        except tomllib.TOMLDecodeError:
            pass
    pkg = _read(root, "package.json")
    if pkg is not None:
        files.append("package.json")
        try:
            doc = json.loads(pkg)
            data["node"] = {"declared": doc.get("dependencies", {})}
        except json.JSONDecodeError:
            pass
    if not data:
        return None
    n = len(data.get("python", {}).get("declared", [])) + \
        len(data.get("node", {}).get("declared", {}))
    return data, f"Declared dependencies ({n} top-level) with resolved versions where a lockfile exists.", files


def _iter_py(root: Path):
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if any(part in _IGNORE_DIRS for part in rel.parts):
            continue
        yield rel, p


def _parse_db_schema(root: Path):
    tables, files = {}, []
    for rel, p in _iter_py(root):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign) and stmt.targets
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == "__tablename__"
                        and isinstance(stmt.value, ast.Constant)):
                    cols = [t.targets[0].id for t in node.body
                            if isinstance(t, ast.Assign) and t.targets
                            and isinstance(t.targets[0], ast.Name)
                            and t.targets[0].id != "__tablename__"]
                    tables[stmt.value.value] = {"model": node.name,
                                                "file": str(rel),
                                                "columns": cols}
                    if str(rel) not in files:
                        files.append(str(rel))
    if not tables:
        return None
    head = _alembic_head(root)
    data = {"tables": tables}
    if head:
        data["alembic_head"] = head
        files.append("alembic/versions")
    return data, (f"ORM schema: {len(tables)} table(s)"
                  + (f"; alembic head {head}" if head else "") + "."), files


def _alembic_head(root: Path) -> Optional[str]:
    versions = root / "alembic" / "versions"
    if not versions.is_dir():
        return None
    revs, downs = {}, set()
    for p in versions.glob("*.py"):
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(r'^revision\s*=\s*["\']([\w]+)["\']', src, re.M)
        d = re.search(r'^down_revision\s*=\s*["\']([\w]+)["\']', src, re.M)
        if m:
            revs[m.group(1)] = True
        if d:
            downs.add(d.group(1))
    heads = [r for r in revs if r not in downs]
    return heads[0] if len(heads) == 1 else (heads and ",".join(sorted(heads)) or None)


def _parse_commands(root: Path):
    data, files = {}, []
    mk = _read(root, "Makefile")
    if mk is not None:
        files.append("Makefile")
        targets = re.findall(r"^([A-Za-z][\w.-]*)\s*:(?!=)", mk, re.M)
        if targets:
            data["make"] = sorted(set(targets))
    pkg = _read(root, "package.json")
    if pkg is not None:
        try:
            scripts = json.loads(pkg).get("scripts", {})
            if scripts:
                data["npm_scripts"] = sorted(scripts)
                files.append("package.json")
        except json.JSONDecodeError:
            pass
    py = _read(root, "pyproject.toml")
    if py is not None:
        try:
            scripts = tomllib.loads(py).get("project", {}).get("scripts", {})
            if scripts:
                data["entry_points"] = dict(scripts)
                if "pyproject.toml" not in files:
                    files.append("pyproject.toml")
        except tomllib.TOMLDecodeError:
            pass
    if not data:
        return None
    return data, "Run/test/build commands: " + ", ".join(
        f"{k}({len(v)})" for k, v in data.items()) + ".", files


def _parse_config_surface(root: Path):
    keys, files = {}, []
    for rel, p in _iter_py(root):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                    isinstance(b, (ast.Name, ast.Attribute))
                    and (getattr(b, "id", None) or getattr(b, "attr", "")) == "BaseSettings"
                    for b in node.bases):
                fields = [s.target.id for s in node.body
                          if isinstance(s, ast.AnnAssign)
                          and isinstance(s.target, ast.Name)]
                if fields:
                    keys.setdefault("settings_fields", []).extend(fields)
                    if str(rel) not in files:
                        files.append(str(rel))
    # .env KEY NAMES only — values are NEVER stored (spec D9)
    for env_name in (".env", ".env.example"):
        env = _read(root, env_name)
        if env is not None:
            names = re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", env, re.M)
            if names:
                keys.setdefault("env_keys", []).extend(names)
                files.append(env_name)
    if not keys:
        return None
    keys = {k: sorted(set(v)) for k, v in keys.items()}
    n = sum(len(v) for v in keys.values())
    return keys, f"Config surface: {n} key name(s) (values never stored).", files


def _parse_infrastructure(root: Path):
    for name in ("docker-compose.yml", "docker/docker-compose.yml",
                 "compose.yml", "docker-compose.yaml"):
        content = _read(root, name)
        if content is None:
            continue
        # No yaml dependency needed for the shapes we care about: top-level
        # services with image/ports lines.
        services = {}
        current = None
        in_services = False
        for line in content.splitlines():
            if re.match(r"^services\s*:", line):
                in_services = True
                continue
            if in_services and re.match(r"^\S", line):
                in_services = False
            if not in_services:
                continue
            m = re.match(r"^  ([\w-]+)\s*:\s*$", line)
            if m:
                current = m.group(1)
                services[current] = {}
                continue
            if current:
                mi = re.match(r"^\s+image\s*:\s*(\S+)", line)
                mp = re.match(r'^\s+-\s*"?([\d.]+:\d+)"?', line)
                if mi:
                    services[current]["image"] = mi.group(1)
                elif mp:
                    services[current].setdefault("ports", []).append(mp.group(1))
        if services:
            return ({"services": services},
                    f"Infrastructure: {len(services)} compose service(s): "
                    + ", ".join(services) + ".", [name])
    return None


def _parse_data_shapes(root: Path):
    shapes, files = {}, []
    for p in sorted(root.rglob("*schema*.json")):
        rel = p.relative_to(root)
        if any(part in _IGNORE_DIRS for part in rel.parts):
            continue
        try:
            if p.stat().st_size > 200_000:
                continue
            doc = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue
        shapes[str(rel)] = sorted(doc.keys()) if isinstance(doc, dict) else type(doc).__name__
        files.append(str(rel))
        if len(shapes) >= 10:
            break
    if not shapes:
        return None
    return ({"shapes": shapes},
            f"Declared data shapes: {len(shapes)} schema file(s).", files)


PARSERS = {
    "dependencies": _parse_dependencies,
    "db_schema": _parse_db_schema,
    "commands": _parse_commands,
    "config_surface": _parse_config_surface,
    "infrastructure": _parse_infrastructure,
    "data_shapes": _parse_data_shapes,
}


def _content_hash(root: Path, rel_files: list) -> str:
    h = hashlib.sha256()
    for rel in sorted(set(rel_files)):
        p = root / rel
        if p.is_file():
            h.update(rel.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def derive_project_facts(db, project, changed_files: Optional[list] = None) -> dict:
    """Run every parser over the project's first root; write/update one
    derived entity per kind. *changed_files* (reconcile path) skips parsers
    whose source files didn't change AND whose unit already exists; the
    content hash gates the final write either way."""
    root = Path(project.roots[0]) if project.roots else None
    stats = {"derived": 0, "unchanged": 0, "skipped": 0, "removed": 0}
    if root is None or not root.is_dir():
        return stats
    changed = set(changed_files or [])
    for kind, parser in PARSERS.items():
        canonical = fact_canonical(project.slug, kind)
        existing = db.query(CodexEntity).filter_by(canonical_name=canonical).first()
        if changed_files is not None and existing is not None:
            prev_files = (existing.properties or {}).get("source_files", [])
            if prev_files and not (set(prev_files) & changed):
                stats["skipped"] += 1
                continue
        try:
            result = parser(root)
        except Exception as exc:
            logger.warning("project_fact_parser_failed", kind=kind,
                           project=project.slug, error=str(exc))
            continue
        if result is None:
            if existing is not None:
                db.delete(existing)
                stats["removed"] += 1
            continue
        data, description, files = result
        digest = _content_hash(root, files)
        if existing is not None and \
                (existing.properties or {}).get("content_hash") == digest:
            stats["unchanged"] += 1
            continue
        props = {"kind": kind, "data": data, "source_files": files,
                 "content_hash": digest, "project_slug": project.slug}
        if existing is None:
            db.add(CodexEntity(
                id=uuid.uuid5(uuid.NAMESPACE_URL, f"ice:fact:{canonical}"),
                canonical_name=canonical, aliases=[], tags=[],
                entity_type=FACT_TYPE, description=description,
                properties=props, context_payload=f"[{kind}] {description}",
                embedding=None, project_id=project.id, source=SOURCE_DERIVED,
                last_updated=datetime.now(timezone.utc)))
        else:
            existing.description = description
            existing.properties = props
            existing.context_payload = f"[{kind}] {description}"
            existing.project_id = project.id
            existing.source = SOURCE_DERIVED
            existing.last_updated = datetime.now(timezone.utc)
        stats["derived"] += 1
    db.commit()
    logger.info("project_facts_derived", project=project.slug, **stats)
    return stats
