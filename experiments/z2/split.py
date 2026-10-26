"""Z2 S6: Z2 / FINAL split by question family, FINAL part sealed (spec Z2_EVALUATION D5).

60% of families go to Z2, 40% are sealed for FINAL: hash-recorded, never read,
answered or graded before FINAL. The split is decided by a hash of the BASE family
id, so it is deterministic and cannot be tuned; a messy variant always follows its
base family. No-memory control prompts are generic and stay with Z2.

Z2 stages read questions only through `load_z2`, which refuses the sealed file;
`tests/smoke/test_z2_split.py` proves no Z2 loader opens it (§5 item 4).
"""

import hashlib
import json
from pathlib import Path

POLICY = "ice-z2-split-1"
Z2_SHARE = 0.60
SEALED_NAME = "final-sealed.json"


def base_id(family):
    return family.get("base_family") or family["id"]


def goes_to_z2(family):
    digest = hashlib.sha256(f"{POLICY}|{base_id(family)}".encode()).hexdigest()
    return int(digest, 16) % 1000 < Z2_SHARE * 1000


def split(families):
    z2 = [f for f in families if goes_to_z2(f)]
    final = [f for f in families if not goes_to_z2(f)]
    return z2, final


def write_split(directory, history, families, controls=()):
    """Writes `z2-families.json` and the sealed FINAL file; returns the seal record.
    Refuses to overwrite an existing seal (a re-split would expose FINAL)."""
    directory = Path(directory)
    sealed = directory / SEALED_NAME
    if sealed.exists():
        raise RuntimeError(f"{sealed} already sealed; never re-split into the same directory")
    z2, final = split(families)
    body = json.dumps(dict(policy=POLICY, history=history, families=final), ensure_ascii=False,
                      indent=1) + "\n"
    sealed.write_text(body)
    seal = dict(policy=POLICY, history=history, z2_families=len(z2), final_families=len(final),
                final_sha256=hashlib.sha256(body.encode()).hexdigest(), sealed_file=SEALED_NAME)
    (directory / "z2-families.json").write_text(json.dumps(dict(
        seal, families=z2, controls=list(controls)), ensure_ascii=False, indent=1) + "\n")
    (directory / "seal.json").write_text(json.dumps(seal, indent=1) + "\n")
    return seal


def load_z2(path):
    """The only way Z2 stages read questions. Refuses the sealed FINAL file."""
    path = Path(path)
    if path.name == SEALED_NAME:
        raise PermissionError("the FINAL families are sealed until FINAL")
    data = json.loads(path.read_text())
    if any(f.get("id") and not goes_to_z2(f) for f in data.get("families", [])):
        raise ValueError("a FINAL family is present in a Z2 file")
    return data


def load_final(directory):
    """FINAL only — the one reader of the sealed families. Refuses unless the sealed file is
    byte-identical to its recorded hash (FINAL_experiments.md §2.1: B's FINAL asks)."""
    directory = Path(directory)
    if not verify_seal(directory):
        raise ValueError(f"{directory / SEALED_NAME} does not match its seal")
    data = json.loads((directory / SEALED_NAME).read_text())
    if any(goes_to_z2(f) for f in data.get("families", [])):
        raise ValueError("a Z2 family is present in the sealed FINAL file")
    return data


def verify_seal(directory):
    """True when the sealed file is byte-identical to its recorded hash. Hashing
    reads bytes, never parses or shows the families."""
    directory = Path(directory)
    seal = json.loads((directory / "seal.json").read_text())
    actual = hashlib.sha256((directory / SEALED_NAME).read_bytes()).hexdigest()
    return actual == seal["final_sha256"]
