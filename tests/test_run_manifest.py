from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from brain_mri.experiments import run_manifest


def _fake_git_path() -> str:
    return str(Path("git").resolve())


def test_sha256_file_hashes_known_content(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"known manifest content")

    assert run_manifest.sha256_file(sample) == hashlib.sha256(b"known manifest content").hexdigest()


def test_sha256_file_empty_file(tmp_path):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")

    assert run_manifest.sha256_file(empty) == hashlib.sha256(b"").hexdigest()


def test_git_helpers_return_none_when_git_is_unavailable(monkeypatch):
    monkeypatch.setattr(run_manifest.shutil, "which", lambda name: None)

    def fail(*args, **kwargs):
        raise AssertionError("git should not be called when shutil.which fails")

    monkeypatch.setattr(run_manifest.subprocess, "check_output", fail)

    assert run_manifest.git_commit() is None
    assert run_manifest.git_is_dirty() is None


def test_git_helpers_return_none_when_subprocess_fails(monkeypatch):
    git_path = _fake_git_path()

    def fail(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(run_manifest.shutil, "which", lambda name: git_path)
    monkeypatch.setattr(run_manifest.subprocess, "check_output", fail)

    assert run_manifest.git_commit() is None
    assert run_manifest.git_is_dirty() is None


def test_git_helpers_return_none_on_timeout(monkeypatch):
    git_path = _fake_git_path()

    def fail(*args, **kwargs):
        raise run_manifest.subprocess.TimeoutExpired(args[0], kwargs.get("timeout"))

    monkeypatch.setattr(run_manifest.shutil, "which", lambda name: git_path)
    monkeypatch.setattr(run_manifest.subprocess, "check_output", fail)

    assert run_manifest.git_commit() is None
    assert run_manifest.git_is_dirty() is None


def test_git_commit_returns_hash_on_success(monkeypatch):
    fake_hash = "deadbeefcafe1234" * 2 + "deadbeef"
    git_path = _fake_git_path()

    def fake_check_output(cmd, cwd, text, timeout):
        assert cmd == [git_path, "rev-parse", "HEAD"]
        assert cwd == run_manifest._REPO_ROOT
        assert text is True
        assert timeout == run_manifest.GIT_TIMEOUT_SECONDS
        return fake_hash + "\n"

    monkeypatch.setattr(run_manifest.shutil, "which", lambda name: git_path)
    monkeypatch.setattr(run_manifest.subprocess, "check_output", fake_check_output)

    result = run_manifest.git_commit()

    assert result == fake_hash


def test_git_commit_returns_none_for_empty_output(monkeypatch):
    monkeypatch.setattr(run_manifest.shutil, "which", lambda name: _fake_git_path())
    monkeypatch.setattr(
        run_manifest.subprocess, "check_output", lambda *a, **kw: "   \n"
    )

    assert run_manifest.git_commit() is None


def test_git_is_dirty_returns_true_when_output_nonempty(monkeypatch):
    git_path = _fake_git_path()

    def fake_check_output(cmd, cwd, text, timeout):
        assert cmd == [git_path, "status", "--porcelain"]
        assert cwd == run_manifest._REPO_ROOT
        assert text is True
        assert timeout == run_manifest.GIT_TIMEOUT_SECONDS
        return " M brain_mri/foo.py\n"

    monkeypatch.setattr(run_manifest.shutil, "which", lambda name: git_path)
    monkeypatch.setattr(
        run_manifest.subprocess, "check_output", fake_check_output
    )

    assert run_manifest.git_is_dirty() is True


def test_git_is_dirty_returns_false_for_clean_repo(monkeypatch):
    monkeypatch.setattr(run_manifest.shutil, "which", lambda name: _fake_git_path())
    monkeypatch.setattr(
        run_manifest.subprocess, "check_output", lambda *a, **kw: ""
    )

    assert run_manifest.git_is_dirty() is False


def test_relativize_command_converts_absolute_repo_paths(tmp_path):
    script = tmp_path / "scripts" / "run.py"
    script.parent.mkdir()
    script.write_text("print('ok')\n", encoding="utf-8")

    command = run_manifest.relativize_command([str(script), "--seed", "42"], tmp_path)

    assert command == [str(Path("scripts") / "run.py"), "--seed", "42"]


def test_relativize_command_preserves_relative_args(tmp_path):
    command = run_manifest.relativize_command(["run.py", "--foo", "bar", "--seed", "1"], tmp_path)

    assert command == ["run.py", "--foo", "bar", "--seed", "1"]


def test_relativize_command_falls_back_to_basename_for_outside_paths(tmp_path):
    outside = tmp_path / "other" / "tool.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("", encoding="utf-8")

    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    command = run_manifest.relativize_command([str(outside)], repo_root)

    assert command == ["tool.py"]


def test_capture_pip_freeze_returns_list_on_success(monkeypatch):
    fake_output = "numpy==1.24.0\npandas==2.0.1\n\n"

    def fake_check_output(cmd, text, timeout):
        assert cmd == [run_manifest.sys.executable, "-m", "pip", "freeze"]
        assert text is True
        assert timeout == run_manifest.GIT_TIMEOUT_SECONDS
        return fake_output

    monkeypatch.setattr(
        run_manifest.subprocess,
        "check_output",
        fake_check_output,
    )

    result = run_manifest.capture_pip_freeze()

    assert result == ["numpy==1.24.0", "pandas==2.0.1"]


def test_capture_pip_freeze_returns_empty_list_on_success_with_no_output(monkeypatch):
    monkeypatch.setattr(
        run_manifest.subprocess,
        "check_output",
        lambda *a, **kw: "\n",
    )

    assert run_manifest.capture_pip_freeze() == []


def test_capture_pip_freeze_returns_none_on_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("pip unavailable")

    monkeypatch.setattr(run_manifest.subprocess, "check_output", fail)

    assert run_manifest.capture_pip_freeze() is None


def test_capture_pip_freeze_returns_none_on_timeout(monkeypatch):
    def fail(*args, **kwargs):
        raise run_manifest.subprocess.TimeoutExpired(args[0], kwargs.get("timeout"))

    monkeypatch.setattr(run_manifest.subprocess, "check_output", fail)

    assert run_manifest.capture_pip_freeze() is None


def test_generate_timestamp_format():
    ts = run_manifest.generate_timestamp()

    assert re.fullmatch(r"\d{8}_\d{6}", ts), f"Unexpected timestamp format: {ts!r}"


def test_generate_timestamp_uses_utc(monkeypatch):
    class FakeDatetime:
        @classmethod
        def now(cls, tz):
            assert tz is run_manifest.timezone.utc
            return datetime(2025, 1, 2, 3, 4, 5, tzinfo=tz)

    monkeypatch.setattr(run_manifest, "datetime", FakeDatetime)

    assert run_manifest.generate_timestamp() == "20250102_030405"


def test_write_manifest_creates_valid_sorted_json(tmp_path):
    manifest_path = tmp_path / "manifests" / "run.json"

    run_manifest.write_manifest(manifest_path, {"b": 2, "a": {"nested": True}})

    text = manifest_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == {"a": {"nested": True}, "b": 2}
    assert text.index('"a"') < text.index('"b"')


def test_write_manifest_overwrites_existing_file(tmp_path):
    manifest_path = tmp_path / "run.json"
    manifest_path.write_text('{"old": true}\n', encoding="utf-8")

    run_manifest.write_manifest(manifest_path, {"new": 42})

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {"new": 42}


def test_unique_manifest_path_returns_base_when_available(tmp_path):
    result = run_manifest.unique_manifest_path(tmp_path, "20250101_120000")

    assert result == tmp_path / "20250101_120000.json"
    assert result.exists()


def test_unique_manifest_path_returns_first_available_suffix(tmp_path):
    (tmp_path / "20250101_120000.json").write_text("base", encoding="utf-8")
    (tmp_path / "20250101_120000_01.json").write_text("first", encoding="utf-8")

    result = run_manifest.unique_manifest_path(tmp_path, "20250101_120000", max_attempts=2)

    assert result == tmp_path / "20250101_120000_02.json"
    assert result.exists()


def test_unique_manifest_path_raises_after_max_attempts(tmp_path):
    (tmp_path / "20250101_120000.json").write_text("base", encoding="utf-8")
    (tmp_path / "20250101_120000_01.json").write_text("first", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"unique_manifest_path.*manifest_dir=.*timestamp=.*last_candidate="):
        run_manifest.unique_manifest_path(tmp_path, "20250101_120000", max_attempts=1)


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_unique_manifest_path_rejects_non_positive_max_attempts(tmp_path, max_attempts):
    with pytest.raises(ValueError, match="max_attempts >= 1"):
        run_manifest.unique_manifest_path(tmp_path, "20250101_120000", max_attempts=max_attempts)


def test_write_manifest_replaces_reserved_path(tmp_path):
    manifest_path = run_manifest.unique_manifest_path(tmp_path, "20250101_120000")

    run_manifest.write_manifest(manifest_path, {"ok": True})

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {"ok": True}


def test_write_manifest_cleans_reserved_path_on_failure(tmp_path):
    manifest_path = run_manifest.unique_manifest_path(tmp_path, "20250101_120000")

    with pytest.raises(TypeError):
        run_manifest.write_manifest(manifest_path, {"bad": object()})

    assert not manifest_path.exists()
