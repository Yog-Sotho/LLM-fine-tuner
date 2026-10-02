"""Download ZIPs and Hub pushes leave resume checkpoints out; error text is redacted."""

import zipfile

import huggingface_hub
import pytest

TOKEN = "hf_" + "b" * 34


@pytest.fixture
def run_folder(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoint-50").mkdir(parents=True)
    (run / "checkpoint-50" / "optimizer.pt").write_bytes(b"x" * 100)
    (run / "adapter_model.safetensors").write_bytes(b"w")
    (run / "adapter_config.json").write_text("{}")
    (run / "training_args.bin").write_bytes(b"pickle")
    (run / "README.md").write_text("card")
    return run


def test_zip_leaves_checkpoints_and_pickles_out(run_folder):
    from export.utils import create_zip_from_folder

    with zipfile.ZipFile(create_zip_from_folder(str(run_folder))) as zf:
        names = sorted(zf.namelist())
    assert names == ["run/README.md", "run/adapter_config.json", "run/adapter_model.safetensors"]


def test_hub_push_ignores_checkpoints(run_folder, monkeypatch):
    import export.hub as hub

    calls = {}

    class Api:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            pass

        def upload_folder(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(hub, "_complete_card_metadata", lambda api, path: None)
    assert hub.push_to_hub(str(run_folder), "me/model", TOKEN).startswith("✅")
    assert calls["ignore_patterns"] == ["checkpoint-*", "training_args.bin"]


def test_ui_error_text_is_redacted(monkeypatch):
    import inference.generate as generate

    def boom(*args, **kwargs):
        raise RuntimeError(f"401 for token {TOKEN}")

    monkeypatch.setattr(generate, "_load_for_inference", boom)
    message = generate.generate_text("org/model", "", "hi", 4, 0.7, 0.9)
    assert message.startswith("❌ Generation failed") and TOKEN not in message
    assert "[REDACTED]" in message


def test_failed_training_messages_are_redacted(monkeypatch, tmp_path):
    import training.kto as kto

    def boom(file):
        raise ValueError(f"bad data near {TOKEN}")

    monkeypatch.setattr(kto, "load_table_dataset", boom)

    class File:
        name = str(tmp_path / "d.csv")

    status = kto.train_kto("org/model", File(), str(tmp_path / "out"), progress=None)
    assert status.startswith("❌ KTO training failed") and TOKEN not in status
