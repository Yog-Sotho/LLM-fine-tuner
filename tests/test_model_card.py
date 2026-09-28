"""Model cards (written with every run) and the Hub push — no network, no model download."""

import sys
import types

import pytest
import yaml
from datasets import Dataset

import export.hub as hub
from core.model_card import build_model_card, hub_model_id, write_model_card
from core.run_config import dataset_fingerprint, save_run_config
from data.preprocessing import validate_and_clean_dataset

RECORD = {
    "mode": "sft",
    "model": "owner/base",
    "seed": 7,
    "created_at": "2026-01-01T00:00:00+00:00",
    "hyperparams": {"learning_rate": 0.0002, "epochs": 1},
    "peft": {"method": "LoRA", "lora_rank": 8},
    "dataset": {"rows": 3, "sha256": "abc", "hub_id": "owner/data"},
    "libraries": {"trl": "1.0"},
}
TOKEN = "hf_" + "a" * 34


def _header(card) -> dict:
    return yaml.safe_load(str(card).split("---")[1])


@pytest.mark.parametrize(
    ("model", "expected"),
    [("owner/base", "owner/base"), ("gpt2", "gpt2"), ("", None), ("a b", None),
     ("/abs/path", None), ("owner/base/extra", None)],
)  # fmt: skip
def test_hub_model_id(model, expected):
    assert hub_model_id(model) == expected


def test_local_base_model_is_never_base_model(tmp_path):
    # The Hub rejects a card whose base_model is a local path.
    assert hub_model_id(str(tmp_path)) is None
    header = _header(build_model_card({**RECORD, "model": str(tmp_path)}, is_adapter=True))
    assert "base_model" not in header


def test_adapter_card_metadata_and_body():
    card = build_model_card(RECORD, is_adapter=True)
    assert _header(card) == {
        "base_model": "owner/base",
        "datasets": ["owner/data"],
        "library_name": "peft",
        "pipeline_tag": "text-generation",
        "tags": ["llm-fine-tuner", "trl", "sft", "lora"],
    }
    text = str(card)
    assert "AutoPeftModelForCausalLM" in text
    assert "| learning_rate | 0.0002 |" in text and "| seed | 7 |" in text
    assert "[owner/data](https://huggingface.co/datasets/owner/data)" in text


@pytest.mark.parametrize(
    ("record", "is_adapter", "library", "pipeline", "usage"),
    [
        ({**RECORD, "peft": {"method": "Full Fine-tuning"}}, False, "transformers",
         "text-generation", "AutoModelForCausalLM"),
        ({**RECORD, "mode": "reward"}, False, "transformers", "text-classification",
         "AutoModelForSequenceClassification"),
    ],
)  # fmt: skip
def test_full_model_and_reward_cards(record, is_adapter, library, pipeline, usage):
    card = build_model_card(record, is_adapter)
    header = _header(card)
    assert (header["library_name"], header["pipeline_tag"]) == (library, pipeline)
    assert usage in str(card)


def test_local_dataset_has_no_datasets_entry():
    record = {**RECORD, "dataset": {"rows": 3, "sha256": "abc"}}
    assert "datasets" not in _header(build_model_card(record, is_adapter=True))


def test_save_run_config_writes_the_card(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    save_run_config(str(tmp_path), mode="kto", model="owner/base",
                    dataset=Dataset.from_dict({"text": ["a"]}), seed=1)  # fmt: skip
    header = _header((tmp_path / "README.md").read_text())
    assert header["base_model"] == "owner/base" and header["library_name"] == "peft"
    assert "kto" in header["tags"]


def test_write_model_card_detects_adapters(tmp_path):
    write_model_card(str(tmp_path), RECORD)
    assert _header((tmp_path / "README.md").read_text())["library_name"] == "transformers"


def test_hub_dataset_id_survives_cleaning_and_is_fingerprinted():
    ds = Dataset.from_dict({"instruction": ["q", "q2"], "output": ["a", "a2"]})
    ds.info.dataset_name = "owner/data"  # as load_hub_dataset records it
    cleaned, _ = validate_and_clean_dataset(ds)
    assert dataset_fingerprint(cleaned)["hub_id"] == "owner/data"
    plain = Dataset.from_dict({"text": ["a"]})
    assert "hub_id" not in dataset_fingerprint(plain)


# ── Hub push (HfApi replaced by a fake) ────────────────────────────────────


class _HTTPError(Exception):
    pass


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    calls: dict = {}

    class FakeApi:
        def __init__(self, token=None):
            calls["token"] = token

        def model_info(self, repo_id):
            calls["model_info"] = repo_id
            if repo_id == "missing/model":
                raise _HTTPError("404")
            card = sys.modules["huggingface_hub"].ModelCardData(license="apache-2.0")
            return types.SimpleNamespace(id=f"canonical/{repo_id}", card_data=card)

        def create_repo(self, **kwargs):
            calls["create_repo"] = kwargs

        def upload_folder(self, **kwargs):
            calls["upload_folder"] = kwargs

    import huggingface_hub
    import huggingface_hub.errors

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub.errors, "HfHubHTTPError", _HTTPError)
    monkeypatch.setattr(hub, "HAS_HUB", True)
    return calls


def test_push_creates_repo_and_completes_card(fake_hub, tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    write_model_card(str(tmp_path), {**RECORD, "model": "gpt2"})
    status = hub.push_to_hub(str(tmp_path), "me/model", TOKEN)
    assert status == "✅ Pushed to https://huggingface.co/me/model", status
    # upload_folder needs an existing repo: it must be created first (idempotently).
    assert fake_hub["create_repo"] == {"repo_id": "me/model", "repo_type": "model",
                                       "exist_ok": True}  # fmt: skip
    assert fake_hub["upload_folder"]["folder_path"] == str(tmp_path)
    header = _header((tmp_path / "README.md").read_text())
    assert header["base_model"] == "canonical/gpt2" and header["license"] == "apache-2.0"


def test_push_drops_a_base_model_the_hub_does_not_know(fake_hub, tmp_path):
    write_model_card(str(tmp_path), {**RECORD, "model": "missing/model"})
    assert hub.push_to_hub(str(tmp_path), "me/model", TOKEN).startswith("✅")
    assert "base_model" not in _header((tmp_path / "README.md").read_text())
