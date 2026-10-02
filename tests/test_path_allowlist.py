"""Web UI path allow-list (core/state.restrict_paths_to): visitors can't reach other folders."""

import os

import pytest

import core.state as state


@pytest.fixture
def restricted(tmp_path, monkeypatch):
    """UI mode with the working directory and one runs folder allowed."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(state, "_allowed_roots", None)
    state.restrict_paths_to([str(tmp_path), str(runs)])
    yield tmp_path
    monkeypatch.setattr(state, "_allowed_roots", None)


def test_cli_mode_is_not_restricted(monkeypatch):
    monkeypatch.setattr(state, "_allowed_roots", None)
    assert state.validate_path_traversal("/etc/passwd") is None


@pytest.mark.parametrize("path", ["org/model", "runs/exp1", "hf_token_value", "1.0", "model.gguf"])
def test_hub_ids_names_and_relative_paths_stay_allowed(restricted, path):
    assert state.validate_path_traversal(path) is None


def test_absolute_paths_inside_allowed_folders(restricted):
    assert state.validate_path_traversal(str(restricted / "runs" / "exp1")) is None


@pytest.mark.parametrize("path", ["/etc/passwd", "/root/.ssh", "/"])
def test_paths_outside_are_rejected(restricted, path):
    assert "Path outside the folders" in state.validate_path_traversal(path)


def test_prefix_lookalikes_and_symlinks_are_rejected(restricted, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    assert state.validate_path_traversal(str(restricted) + "-evil/x")  # same prefix, other dir
    link = restricted / "link"
    os.symlink(outside, link)
    assert "Path outside" in state.validate_path_traversal("link/secret")  # resolved


def test_ui_handlers_refuse_outside_paths(restricted):
    from export.gguf import on_export_gguf
    from training.kto import train_kto

    status, _ = on_export_gguf("/etc", "q4_k_m")
    assert "Path outside the folders" in status

    class File:
        name = "data.csv"

    assert "Path outside the folders" in train_kto("m", File(), "/root/kto", progress=None)


def test_ui_roots_include_runs_temp_uploads_and_extras(monkeypatch, tmp_path):
    import tempfile

    import main

    monkeypatch.setattr("config.constants.EXTRA_ALLOWED_PATHS", ["/data/models"])
    roots = main._ui_path_roots({"GRADIO_TEMP_DIR": str(tmp_path)})
    assert os.getcwd() in roots and tempfile.gettempdir() in roots
    assert "/data/models" in roots and str(tmp_path) in roots


def test_extra_allowed_paths_setting(monkeypatch):
    import importlib

    import config.constants as constants

    monkeypatch.setenv("LFT_ALLOWED_PATHS", os.pathsep.join(["/data/a", "", "/data/b"]))
    try:
        assert importlib.reload(constants).EXTRA_ALLOWED_PATHS == ["/data/a", "/data/b"]
    finally:
        monkeypatch.delenv("LFT_ALLOWED_PATHS")
        importlib.reload(constants)
