"""G23 D3: the one backup wrapper — pg_dump -Fc + trained artifacts + config.

The user-facing entry is scripts/ice_backup.sh (thin exec of this module's
CLI); FINAL's per-checkpoint snapshots call ``snapshot_db()`` directly —
one module, no parallel wrapper. ``model_registry.json`` lives inside
``models/`` and rides along with the copytree.

Every archive carries a RESTORE.md with the exact restore steps and a
backup_info.json (alembic head, git commit, db size) so a future restore
knows what state it is reviving.

Backups are LOCAL AND PRIVATE — they contain the full memory store and the
.env config. ``backups/`` is gitignored; never commit or upload an archive.
"""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import structlog

from src.paths import REPO_ROOT

logger = structlog.get_logger("ice.memory.backup")

# G31: one shared install root (src/paths.py) — this was one of five copies.
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.yml"


def _db_params(database_url: str | None = None) -> dict:
    from src.api.config import settings
    url = database_url or settings.database_url
    parsed = urlparse(url.replace("postgresql+psycopg://", "postgresql://"))
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "user": parsed.username or "ice",
        "password": parsed.password or "",
        "dbname": (parsed.path or "/ice_db").lstrip("/"),
    }


def snapshot_db(out_path: str | Path, database_url: str | None = None) -> Path:
    """``pg_dump -Fc`` of the ICE database to *out_path*.

    Uses the host's pg_dump when installed; otherwise runs pg_dump inside
    the postgres container (guaranteed present, same server version). Fails
    loud on an implausibly small dump — a silent empty backup is worse than
    no backup.
    """
    p = _db_params(database_url)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("pg_dump"):
        cmd = ["pg_dump", "-Fc", "-h", p["host"], "-p", str(p["port"]),
               "-U", p["user"], p["dbname"]]
        env = {**os.environ, "PGPASSWORD": p["password"]}
        with open(out_path, "wb") as fh:
            subprocess.run(cmd, check=True, env=env, stdout=fh)
        via = "host"
    else:
        cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "exec", "-T",
               "postgres", "pg_dump", "-U", p["user"], "-Fc", p["dbname"]]
        with open(out_path, "wb") as fh:
            subprocess.run(cmd, check=True, stdout=fh)
        via = "container"

    size = out_path.stat().st_size
    if size < 4096:
        raise RuntimeError(
            f"pg_dump produced an implausibly small file ({size} bytes) at "
            f"{out_path} — refusing to call this a backup.")
    logger.info("db_snapshot_written", path=str(out_path), bytes=size, via=via)
    return out_path


def _backup_info(database_url: str | None = None) -> dict:
    """Provenance for the archive: alembic head, db size, git commit."""
    from sqlalchemy import create_engine, text

    from src.api.config import settings
    url = database_url or settings.database_url
    info = {"created_at": datetime.now(timezone.utc).isoformat(),
            "database_url_host": _db_params(url)["host"],
            "dbname": _db_params(url)["dbname"]}
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            info["alembic_head"] = conn.execute(
                text("SELECT version_num FROM alembic_version")).scalar()
            info["db_size"] = conn.execute(text(
                "SELECT pg_size_pretty(pg_database_size(current_database()))"
            )).scalar()
    finally:
        engine.dispose()
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True).stdout.strip()
    except Exception:
        info["git_commit"] = None
    return info


_RESTORE_MD = """# Restoring this ICE backup

Contents: `ice_db.dump` (pg_dump -Fc of the full memory store),
`models/` (trained artifacts incl. model_registry.json), `env.snapshot`
(the .env config at backup time), `backup_info.json` (alembic head, git
commit, db size at backup time).

1. Start postgres:
   `docker compose -f docker/docker-compose.yml up -d postgres`
2. Restore the database (drops/recreates objects inside ice_db):
   `docker compose -f docker/docker-compose.yml exec -T postgres pg_restore -U ice -d ice_db --clean --if-exists < ice_db.dump`
   (with host tools: `pg_restore -h localhost -U ice -d ice_db --clean --if-exists ice_db.dump`)
3. Check out the git commit named in backup_info.json (the schema the dump
   carries matches that commit's alembic head — restoring into a newer
   checkout then needs `uv run alembic upgrade head`).
4. Copy `models/` over the repo's `models/`, and `env.snapshot` to `.env`
   if the current config was lost.
"""


def run_backup(out_dir: str | Path = "backups", tag: str | None = None,
               database_url: str | None = None) -> Path:
    """Full backup: DB dump + models/ + .env into one tar.gz archive.

    Returns the archive path. Prints (and logs) the path and size — the
    USER-REQUIRED confirmation for G23/C17 is exactly "archive exists +
    logged size".
    """
    out_dir = Path(out_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"ice_backup_{stamp}" + (f"_{tag}" if tag else "")
    staging = out_dir / name
    staging.mkdir(parents=True, exist_ok=False)

    try:
        snapshot_db(staging / "ice_db.dump", database_url=database_url)

        models_dir = REPO_ROOT / "models"
        if models_dir.is_dir():
            shutil.copytree(models_dir, staging / "models")
        else:
            logger.warning("backup_models_missing", path=str(models_dir))

        env_file = REPO_ROOT / ".env"
        if env_file.is_file():
            shutil.copy2(env_file, staging / "env.snapshot")

        (staging / "backup_info.json").write_text(
            json.dumps(_backup_info(database_url), indent=2))
        (staging / "RESTORE.md").write_text(_RESTORE_MD)

        archive = out_dir / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(staging, arcname=name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    size = archive.stat().st_size
    logger.info("backup_complete", archive=str(archive), bytes=size)
    print(f"Backup archive: {archive}")
    print(f"Archive size:   {size / (1024 * 1024):.1f} MB")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICE full backup: pg_dump + models/ + config snapshot.")
    parser.add_argument("--out-dir", default="backups",
                        help="directory for archives (default: backups/)")
    parser.add_argument("--tag", default=None,
                        help="optional label appended to the archive name")
    args = parser.parse_args()
    run_backup(out_dir=args.out_dir, tag=args.tag)


if __name__ == "__main__":
    main()
