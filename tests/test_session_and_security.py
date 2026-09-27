"""Regression tests for per-session state, adapter safety and launch configuration."""

import importlib
import json
import pathlib
import threading
import zipfile

import pytest

import main
from core.callbacks import StopCallback
from core.state import AppState, SessionState, validate_adapter_dir
from export.utils import on_peft_zip_upload

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ── Session isolation ──────────────────────────────────────────────────────


def test_stop_in_one_session_does_not_stop_another():
    state = AppState()
    state.session("tab-a").stop_event.set()
    assert state.session("tab-a").stop_event.is_set()
    assert not state.session("tab-b").stop_event.is_set()


def test_session_for_without_request_uses_default_session():
    state = AppState()
    assert state.session_for(None) is state.session()


def test_session_for_uses_request_session_hash():
    state = AppState()

    class _Req:
        session_hash = "abc123"

    assert state.session_for(_Req()) is state.session("abc123")


def test_new_run_only_frees_own_session_files(tmp_path):
    state = AppState()
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    state.session("a").track("model_dir", str(a_dir))
    state.session("b").track("model_dir", str(b_dir))

    state.session("b").release("model_dir")

    assert a_dir.exists()
    assert not b_dir.exists()


def test_track_replaces_and_deletes_previous_path(tmp_path):
    session = SessionState()
    old, new = tmp_path / "old.zip", tmp_path / "new.zip"
    old.write_text("x")
    new.write_text("y")
    session.track("zip", str(old))
    session.track("zip", str(new))
    assert not old.exists()
    assert new.exists()
    assert session.tracked("zip") == str(new)


def test_close_session_stops_job_and_deletes_files(tmp_path):
    state = AppState()
    out = tmp_path / "run"
    out.mkdir()
    session = state.session("tab")
    session.track("model_dir", str(out))

    state.close_session("tab")

    assert session.stop_event.is_set()
    assert not out.exists()
    assert state.session("tab") is not session


def test_track_after_close_deletes_immediately(tmp_path):
    session = SessionState()
    session.close()
    late = tmp_path / "late"
    late.mkdir()
    session.track("model_dir", str(late))
    assert not late.exists()


def test_stop_callback_uses_its_own_event():
    own = threading.Event()
    callback = StopCallback(own)

    class _Control:
        should_training_stop = False

    control = callback.on_step_end(None, None, _Control())
    assert control.should_training_stop is False
    own.set()
    control = callback.on_step_end(None, None, _Control())
    assert control.should_training_stop is True


# ── Adapter safety ─────────────────────────────────────────────────────────


def _write_adapter(directory: pathlib.Path, weights_name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    (directory / weights_name).write_bytes(b"\x00")


def test_validate_adapter_dir_accepts_safetensors(tmp_path):
    _write_adapter(tmp_path, "adapter_model.safetensors")
    assert validate_adapter_dir(str(tmp_path)) is None


def test_validate_adapter_dir_rejects_pickle_weights(tmp_path):
    _write_adapter(tmp_path, "adapter_model.bin")
    err = validate_adapter_dir(str(tmp_path))
    assert err and "safetensors" in err


def test_validate_adapter_dir_rejects_non_adapter(tmp_path):
    assert "adapter_config.json" in validate_adapter_dir(str(tmp_path))


class _Upload:
    def __init__(self, name: str) -> None:
        self.name = name


def _zip_dir(src: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    with zipfile.ZipFile(dest, "w") as zf:
        for f in src.rglob("*"):
            zf.write(f, f.relative_to(src.parent))
    return dest


def test_peft_zip_with_pickle_weights_is_rejected_and_removed(tmp_path, monkeypatch):
    import core.state as state_mod

    monkeypatch.setattr(state_mod, "app_state", AppState())
    monkeypatch.setattr("export.utils.app_state", state_mod.app_state)
    _write_adapter(tmp_path / "adapter", "adapter_model.bin")
    archive = _zip_dir(tmp_path / "adapter", tmp_path / "adapter.zip")

    path, status, _ = on_peft_zip_upload(_Upload(str(archive)))

    assert status.startswith("❌") and "pickle" in status
    assert path.strip() == ""
    assert state_mod.app_state.session().tracked("peft_dir") is None


def test_peft_zip_with_safetensors_adapter_is_accepted(tmp_path, monkeypatch):
    import core.state as state_mod

    monkeypatch.setattr(state_mod, "app_state", AppState())
    monkeypatch.setattr("export.utils.app_state", state_mod.app_state)
    _write_adapter(tmp_path / "adapter", "adapter_model.safetensors")
    archive = _zip_dir(tmp_path / "adapter", tmp_path / "adapter.zip")

    path, status, _ = on_peft_zip_upload(_Upload(str(archive)))

    assert status.startswith("✅")
    assert validate_adapter_dir(path) is None


def test_inference_refuses_pickle_adapter_before_loading_model(tmp_path):
    from inference.generate import _load_for_inference

    _write_adapter(tmp_path, "adapter_model.bin")
    with pytest.raises(ValueError, match="safetensors"):
        _load_for_inference("some-model-that-is-never-downloaded", str(tmp_path))


# ── Remote code is opt-in ──────────────────────────────────────────────────


def test_remote_code_disabled_by_default(monkeypatch):
    import config.constants as constants

    monkeypatch.delenv("ALLOW_REMOTE_CODE", raising=False)
    assert importlib.reload(constants).ALLOW_REMOTE_CODE is False
    monkeypatch.setenv("ALLOW_REMOTE_CODE", "true")
    assert importlib.reload(constants).ALLOW_REMOTE_CODE is True
    monkeypatch.delenv("ALLOW_REMOTE_CODE")
    importlib.reload(constants)


def test_no_hardcoded_trust_remote_code_true():
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in REPO_ROOT.rglob("*.py")
        if not {"archive", "tests"} & set(p.relative_to(REPO_ROOT).parts)
        and "trust_remote_code=True" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


# ── Launch configuration ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("alice:pw", [("alice", "pw")]),
        ("alice:pw, bob:p:w", [("alice", "pw"), ("bob", "p:w")]),
    ],
)
def test_parse_auth(raw, expected):
    assert main._parse_auth(raw) == expected


@pytest.mark.parametrize("raw", ["alice", "alice:", ":pw"])
def test_parse_auth_rejects_malformed(raw):
    with pytest.raises(ValueError):
        main._parse_auth(raw)


def test_launch_defaults_to_no_share_and_leaves_host_to_gradio():
    kwargs = main._launch_kwargs({})
    assert kwargs["share"] is False
    assert kwargs["auth"] is None
    assert "server_name" not in kwargs


@pytest.mark.parametrize("var", ["SHARE", "GRADIO_SHARE"])
def test_share_env_vars(var):
    assert main._launch_kwargs({var: "true"})["share"] is True


def test_exposure_warning_only_when_exposed_without_auth():
    local = main._launch_kwargs({})
    assert main._exposure_warning({}, local) is None

    exposed_env = {"GRADIO_SERVER_NAME": "0.0.0.0"}
    assert main._exposure_warning(exposed_env, main._launch_kwargs(exposed_env))

    with_auth = {**exposed_env, "GRADIO_AUTH": "a:b"}
    assert main._exposure_warning(with_auth, main._launch_kwargs(with_auth)) is None
