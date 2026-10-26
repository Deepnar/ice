"""Fetch the pinned A-Mem memory layer (AMEM.json) into vendor/amem/ — stdlib only, so ICE's
runner can call it too. Refuses any bytes but the pinned ones.

    python experiments/final/baselines/vendor.py
"""

import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = json.loads((HERE / "AMEM.json").read_text())
TARGET = HERE / "vendor" / "amem" / PIN["file"]


def fetch():
    if TARGET.exists() and hashlib.sha256(TARGET.read_bytes()).hexdigest() == PIN["sha256"]:
        return TARGET
    repo = PIN["repository"].removeprefix("https://github.com/")
    url = f"https://raw.githubusercontent.com/{repo}/{PIN['commit']}/{PIN['file']}"
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != PIN["sha256"]:
        raise ValueError(f"{url}: not the pinned bytes (sha256 {PIN['sha256'][:12]}...)")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(raw)
    return TARGET


if __name__ == "__main__":
    print(fetch())
