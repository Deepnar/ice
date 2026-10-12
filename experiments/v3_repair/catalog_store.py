"""Immutable, lossless catalogs for isolated research traces; no memory writes."""

import gzip
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

from experiments.v3_repair.artifact_inventory import validate

POLICY = "ice-v3-artifact-catalog-reference-1"
DIRECTORY = "artifact-catalogs"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def owners(record):
    """Only native catalog owners; never inspect model message contents."""
    yield record
    if isinstance(record.get("preflight"), dict):
        yield record["preflight"]
    if isinstance(record.get("stages"), dict):
        yield from record["stages"].values()


def persist(root, catalog):
    validate(catalog)
    body = json.dumps(catalog, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    blob = gzip.compress(body, mtime=0)
    reference = dict(version="v3", policy=POLICY, codec="gzip", sha256=sha(body),
        catalog_sha256=catalog["catalog_sha256"], stored_sha256=sha(blob),
        uncompressed_bytes=len(body), stored_bytes=len(blob))
    directory = Path(root) / DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (reference["sha256"] + ".json.gz")
    if path.exists():
        if path.read_bytes() != blob:
            raise ValueError("existing immutable catalog blob changed")
        return reference
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".catalog-", delete=False) as sink:
            temporary = Path(sink.name)
            sink.write(blob)
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != blob:
                raise ValueError("concurrent immutable catalog blob differs")
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return reference


def load(root, reference):
    fields = {"version", "policy", "codec", "sha256", "catalog_sha256", "stored_sha256",
              "uncompressed_bytes", "stored_bytes"}
    if (not isinstance(reference, dict) or set(reference) != fields
            or reference["version"] != "v3" or reference["policy"] != POLICY
            or reference["codec"] != "gzip"):
        raise ValueError("catalog reference policy differs")
    for name in ("sha256", "catalog_sha256", "stored_sha256"):
        value = reference[name]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("catalog reference hash malformed")
    for name in ("uncompressed_bytes", "stored_bytes"):
        if type(reference[name]) is not int or reference[name] <= 0:
            raise ValueError("catalog reference size malformed")
    if root is None:
        raise ValueError("catalog references require their native run root")
    path = Path(root) / DIRECTORY / (reference["sha256"] + ".json.gz")
    blob = path.read_bytes()
    if len(blob) != reference["stored_bytes"] or sha(blob) != reference["stored_sha256"]:
        raise ValueError("stored catalog bytes changed")
    body = gzip.decompress(blob)
    if len(body) != reference["uncompressed_bytes"] or sha(body) != reference["sha256"]:
        raise ValueError("decoded catalog bytes changed")
    catalog = json.loads(body)
    validate(catalog)
    if catalog["catalog_sha256"] != reference["catalog_sha256"]:
        raise ValueError("catalog reference identity differs")
    return catalog


def externalize(root, record):
    """Clone structural owners, keeping caller objects and all measured data."""
    result = dict(record)
    if isinstance(record.get("preflight"), dict):
        result["preflight"] = dict(record["preflight"])
    if isinstance(record.get("stages"), dict):
        result["stages"] = {arm: dict(stage) for arm, stage in record["stages"].items()}
    for owner in owners(result):
        if "artifact_inventory_ref" in owner:
            raise ValueError("new native capture already contains a catalog reference")
        if "artifact_inventory" in owner:
            owner["artifact_inventory_ref"] = persist(root, owner.pop("artifact_inventory"))
    return result


def materialize(root, record):
    """Legacy inline bodies are unchanged; referenced bodies must fully verify."""
    result = dict(record)
    if isinstance(record.get("preflight"), dict):
        result["preflight"] = dict(record["preflight"])
    if isinstance(record.get("stages"), dict):
        result["stages"] = {arm: dict(stage) for arm, stage in record["stages"].items()}
    for owner in owners(result):
        if "artifact_inventory_ref" in owner:
            if "artifact_inventory" in owner:
                raise ValueError("ambiguous inline and referenced native catalog")
            owner["artifact_inventory"] = load(root, owner.pop("artifact_inventory_ref"))
    return result


class Reader:
    """Resolve counted stages without retaining every full catalog in RAM."""

    def __init__(self, root):
        self.root = None if root is None else Path(root)
        self.references = {}
        self.cache = OrderedDict()

    def __call__(self, owner):
        if "artifact_inventory_ref" not in owner:
            return owner.get("artifact_inventory")
        if "artifact_inventory" in owner:
            raise ValueError("ambiguous inline and referenced native catalog")
        reference = owner["artifact_inventory_ref"]
        key = reference.get("sha256") if isinstance(reference, dict) else None
        if key in self.references and self.references[key] != reference:
            raise ValueError("repeated catalog references disagree")
        if key in self.cache:
            value, stamp = self.cache[key]
            path = self.root / DIRECTORY / (key + ".json.gz")
            current = path.stat()
            if stamp == (current.st_size, current.st_mtime_ns):
                self.cache.move_to_end(key)
                return value
        value = load(self.root, reference)
        path = self.root / DIRECTORY / (reference["sha256"] + ".json.gz")
        status = path.stat()
        self.references[reference["sha256"]] = dict(reference)
        self.cache[reference["sha256"]] = (value, (status.st_size, status.st_mtime_ns))
        self.cache.move_to_end(reference["sha256"])
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return value


def verify_trace(root, rows):
    """Validate every committed catalog before completed resume or recount."""
    references = {}
    policy = rows[0].get("identity", {}).get("artifact_catalogs") if rows else None
    for record in rows:
        expected = ([record] if record.get("event") == "source_written" else
                    [record["preflight"]] if record.get("event") == "source_preflight" else
                    list(record["stages"].values()) if record.get("event") == "chronological_probe" else [])
        if policy == POLICY and any("artifact_inventory_ref" not in owner for owner in expected):
            raise ValueError("native catalog reference capture missing")
        for owner in owners(record):
            if "artifact_inventory_ref" not in owner:
                continue
            reference = owner["artifact_inventory_ref"]
            if policy != POLICY:
                raise ValueError("native catalog storage policy unbound")
            if "artifact_inventory" in owner:
                raise ValueError("ambiguous inline and referenced native catalog")
            key = reference["sha256"]
            if key in references and references[key] != reference:
                raise ValueError("repeated catalog references disagree")
            if key not in references:
                load(root, reference)
            references[key] = reference
    return references
