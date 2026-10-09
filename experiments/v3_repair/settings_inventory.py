"""Offline candidate consumers, never evidence of a live setting effect."""

import ast
import math
from pathlib import Path

PREFIXES = (
    "context_", "retrieval_", "ltm_", "recent_", "chunk_", "turn_", "summary_",
    "conversation_summary_", "batch_summary_", "procedural_", "codex_", "cluster_",
    "decay_", "strengthen_", "source_support_", "memory_source_", "confidence_",
    "temporal_", "vector_",
)
EXCLUDED_PARTS = {"api", "password", "secret", "credential", "url", "host",
                  "port", "endpoint", "model", "path", "tokenizer"}


def numeric_value(value):
    if value is None or isinstance(value, (bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and numeric_value(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return all(numeric_value(v) for v in value)
    return False


def relevant(name, default, resolved):
    return (name.startswith(PREFIXES) and not EXCLUDED_PARTS.intersection(name.split("_"))
            and numeric_value(default) and numeric_value(resolved))


def candidate_reads(source, path):
    """Recognize direct/import-alias/literal-getattr reads, retaining dynamic gaps."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    settings_names, config_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("config"):
            for item in node.names:
                if item.name == "settings":
                    settings_names.add(item.asname or item.name)
        elif isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "src.api.config":
                    config_names.add(item.asname or "src.api.config")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Several pure production helpers accept Settings explicitly.
            # This is a candidate parameter, not confirmation of its runtime type.
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if arg.arg == "settings" or (arg.annotation is not None
                                             and ast.unparse(arg.annotation) == "Settings"):
                    settings_names.add(arg.arg)

    def settings_object(node):
        if isinstance(node, ast.Name):
            return node.id in settings_names
        return (isinstance(node, ast.Attribute) and node.attr == "settings"
                and ast.unparse(node.value) in config_names)

    def settings_value(node):
        if settings_object(node):
            return True
        if isinstance(node, ast.BoolOp):
            return any(settings_value(value) for value in node.values)
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"model_copy", "copy"}
                and settings_object(node.func.value))

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and settings_value(node.value):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in settings_names:
                        settings_names.add(target.id)
                        changed = True

    reads, dynamic = {}, []
    for node in ast.walk(tree):
        kind, name = None, None
        if isinstance(node, ast.Attribute) and settings_object(node.value):
            # Writes are not evidence that the setting controls a decision.
            parent = parents.get(node)
            if (isinstance(node.ctx, ast.Load)
                    and not (isinstance(parent, ast.Call) and parent.func is node)):
                kind, name = "attribute", node.attr
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "getattr" and len(node.args) >= 2
              and settings_object(node.args[0])):
            if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                kind, name = "literal_getattr", node.args[1].value
            else:
                dynamic.append(dict(path=path, line=node.lineno, kind="dynamic_getattr"))
        if name is not None:
            reads.setdefault(name, []).append(dict(path=path, line=node.lineno, kind=kind))
    return reads, dynamic


def build_inventory(root: Path, settings_class, resolved):
    reads, dynamic = {}, []
    for path in sorted((root / "src").rglob("*.py")):
        found, gaps = candidate_reads(path.read_text(), path.relative_to(root).as_posix())
        dynamic.extend(gaps)
        for name, sites in found.items():
            reads.setdefault(name, []).extend(sites)
    rows = []
    for name, field in sorted(settings_class.model_fields.items()):
        default, value = field.get_default(), getattr(resolved, name)
        if not relevant(name, default, value):
            continue
        sites = reads.get(name, [])
        rows.append(dict(name=name, code_default=default, resolved_value=value,
                         candidate_reads=sites,
                         static_status="candidate_reads" if sites else "unresolved_no_static_read",
                         live_decision_delta=None, cost=None, tuning_verdict="unmeasured",
                         coupled_policy_review_required=True))
    return dict(version="v3", method="static_settings_candidates_v1", settings=rows,
                dynamic_reads=dynamic, behavioral_qualification=False,
                note="Static presence is not a live effect; do not sweep or delete from this alone.")
