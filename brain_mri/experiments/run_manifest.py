"""Shared helpers for writing reproducible run manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_TIMEOUT_SECONDS = 5
DEFAULT_MAX_MANIFEST_ATTEMPTS = 100
_RESERVATION_MARKER = "__brain_mri_run_manifest_reserved__\n"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_executable() -> str | None:
    git_path = shutil.which("git")
    if git_path is None:
        return None
    return os.path.abspath(git_path)


def git_commit() -> str | None:
    git_path = _git_executable()
    if git_path is None:
        return None

    try:
        out = subprocess.check_output(
            [git_path, "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
        commit = out.strip()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
        UnicodeError,
    ):
        return None
    return commit or None


def git_is_dirty() -> bool | None:
    git_path = _git_executable()
    if git_path is None:
        return None

    try:
        out = subprocess.check_output(
            [git_path, "status", "--porcelain"],
            cwd=_REPO_ROOT,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
        UnicodeError,
    ):
        return None
    return bool(out.strip())


def relativize_command(argv: list[str], repo_root: Path) -> list[str]:
    display: list[str] = []
    resolved_root = repo_root.resolve()

    for raw in argv:
        try:
            path = Path(raw)
        except (TypeError, ValueError):
            display.append(raw)
            continue

        if not path.is_absolute():
            display.append(raw)
            continue

        try:
            display.append(str(path.resolve().relative_to(resolved_root)))
        except (OSError, RuntimeError, ValueError):
            display.append(path.name)

    return display


def capture_pip_freeze() -> list[str] | None:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
        UnicodeError,
    ):
        return None
    return [line for line in out.splitlines() if line.strip()]


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
        fd, temp_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            temp_file.write(payload)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except (OSError, TypeError, ValueError):
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        if _is_manifest_reservation(path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def generate_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_if_exists(path: Path) -> str | None:
    try:
        return sha256_file(path) if path.exists() else None
    except OSError:
        return None


def relative_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except (OSError, RuntimeError, ValueError):
        return str(path)


def manifest_file(path: Path | None, base_dir: Path) -> dict[str, str | None]:
    if path is None:
        return {"path": None, "sha256": None}
    path = Path(path)
    return {"path": relative_path(path, base_dir), "sha256": sha256_if_exists(path)}


def _is_manifest_reservation(path: Path) -> bool:
    try:
        return path.read_text(encoding="utf-8") == _RESERVATION_MARKER
    except (FileNotFoundError, OSError, UnicodeError):
        return False


def _reserve_manifest_path(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    created = False
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        created = True
        os.write(fd, _RESERVATION_MARKER.encode("utf-8"))
        return True
    except FileExistsError:
        return False
    except OSError:
        if created:
            if fd is not None:
                os.close(fd)
                fd = None
            try:
                path.unlink()
            except (FileNotFoundError, OSError):
                pass
        raise
    finally:
        if fd is not None:
            os.close(fd)


def unique_manifest_path(
    manifest_dir: Path,
    timestamp: str,
    *,
    max_attempts: int = DEFAULT_MAX_MANIFEST_ATTEMPTS,
) -> Path:
    if max_attempts < 1:
        raise ValueError("unique_manifest_path requires max_attempts >= 1")

    manifest_dir = Path(manifest_dir)

    manifest_path = manifest_dir / f"{timestamp}.json"
    if _reserve_manifest_path(manifest_path):
        return manifest_path

    candidate = manifest_path
    for counter in range(1, max_attempts + 1):
        candidate = manifest_dir / f"{timestamp}_{counter:02d}.json"
        if _reserve_manifest_path(candidate):
            return candidate

    raise RuntimeError(
        "unique_manifest_path could not find an available manifest path "
        f"for manifest_dir={manifest_dir}, timestamp={timestamp}, "
        f"last_candidate={candidate}, max_attempts={max_attempts}"
    )
