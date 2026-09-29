"""export/registry.py — versioned uploads to a Hub repo, with the Hub faked."""

import json

import huggingface_hub
import pytest

import export.registry as registry

TOKEN = "hf_" + "a" * 34


class FakeHub:
    """Stands in for HfApi / create_repo; keeps uploaded files in memory."""

    def __init__(self, fail=None):
        self.files: dict[str, bytes] = {}
        self.fail = fail
        self.created = []

    def create_repo(self, repo_id, **kwargs):
        if self.fail == "create":
            raise RuntimeError(f"403 for token {TOKEN}")
        self.created.append(repo_id)

    def upload_folder(self, folder_path, **kwargs):
        if self.fail == "upload":
            raise RuntimeError(f"network down ({TOKEN})")
        self.files["model"] = b"weights"

    def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
        self.files[path_in_repo] = path_or_fileobj

    def list_repo_files(self, **kwargs):
        if self.fail == "list":
            raise RuntimeError(f"not found {TOKEN}")
        return list(self.files)

    def hf_hub_download(self, filename, **kwargs):
        if filename == "metadata_vbroken.json":
            raise OSError("missing")
        path = self.tmp / filename
        path.write_bytes(self.files[filename])
        return str(path)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    fake = FakeHub()
    fake.tmp = tmp_path
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)
    monkeypatch.setattr(huggingface_hub, "create_repo", fake.create_repo)
    return fake


def _model(tmp_path, name="model", config=None, adapter=None):
    folder = tmp_path / name
    folder.mkdir()
    if adapter is not None:
        (folder / "adapter_config.json").write_text(json.dumps(adapter))
    if config is not None:
        (folder / "config.json").write_text(
            config if isinstance(config, str) else json.dumps(config)
        )
    return str(folder)


@pytest.mark.parametrize(
    ("files", "base"),
    [
        ({"adapter": {"base_model_name_or_path": "org/base"}}, "org/base"),
        ({"config": {"_name_or_path": "org/full"}}, "org/full"),
        ({"config": {"base_model_name": "org/other"}}, "org/other"),
        ({}, "unknown"),
        ({"config": "{not json"}, "unknown (error:"),
    ],
)
def test_upload_records_version_and_base_model(hub, tmp_path, files, base):
    path = _model(tmp_path, **files)
    status = registry.on_registry_upload(path, " me/models ", TOKEN, " 1.0 ", "first release")
    assert status.startswith("✅ Version 1.0 uploaded to https://huggingface.co/me/models")
    meta = json.loads(hub.files["metadata_v1.0.json"])
    assert meta["base_model"].startswith(base) and meta["version"] == "1.0"
    assert meta["notes"] == "first release" and meta["trained_with"].startswith("LLM Fine-Tuner v")
    assert hub.created == ["me/models"]


def test_list_versions(hub):
    hub.files["metadata_v1.json"] = json.dumps({"base_model": "org/b", "notes": "n" * 60}).encode()
    hub.files["metadata_vbroken.json"] = b""
    listing = registry.on_registry_list("me/models", TOKEN)
    assert listing.startswith("Versions found:")
    assert "• v1: org/b | " + "n" * 50 + "..." in listing and "• metadata_vbroken.json" in listing
    hub.files.clear()
    assert (
        registry.on_registry_list("me/models", TOKEN)
        == "No versioned uploads found in this repository."
    )


@pytest.mark.parametrize(
    ("fail", "call", "message"),
    [
        ("create", "upload", "❌ Upload failed: Failed to create repo: 403"),
        ("upload", "upload", "❌ Upload failed: network down"),
        ("list", "list", "❌ Could not list versions: not found"),
    ],
)
def test_hub_errors_are_reported_without_the_token(hub, tmp_path, fail, call, message):
    hub.fail = fail
    if call == "upload":
        status = registry.on_registry_upload(_model(tmp_path), "me/models", TOKEN, "2", "")
    else:
        status = registry.on_registry_list("me/models", TOKEN)
    assert status.startswith(message) and TOKEN not in status


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("../m", "me/r", TOKEN, "1", ""), "Path traversal"),
        (("m", "me/r", TOKEN, "", ""), "❌ Please enter a valid version tag."),
        (("m", "no-slash", TOKEN, "1", ""), "❌ Invalid Repo ID"),
        (("m", "me/r", "wrong", "1", ""), "❌ Invalid Hugging Face write token."),
        (("/no/such/dir", "me/r", TOKEN, "1", ""), "❌ No trained model found."),
    ],
)
def test_upload_input_validation(hub, args, message):
    assert message in registry.on_registry_upload(*args)
    assert not hub.files


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("../r", TOKEN), "Path traversal"),
        (("no-slash", TOKEN), "❌ Invalid Repo ID."),
        (("me/r", ""), "❌ Invalid Hugging Face write token."),
    ],
)
def test_list_input_validation(hub, args, message):
    assert message in registry.on_registry_list(*args)


def test_registry_checks_its_own_inputs(hub, tmp_path):
    with pytest.raises(ValueError, match="Path traversal"):
        registry.ModelRegistry("me/../r", TOKEN)
    reg = registry.ModelRegistry("me/r", TOKEN)
    assert "Path traversal" in reg.upload_model("../m", "1", {})
    assert reg.upload_model(str(tmp_path / "missing"), "1", {}) == "❌ Invalid model path."


def test_registry_without_huggingface_hub(monkeypatch):
    monkeypatch.setattr(registry, "HAS_HUB", False)
    assert "huggingface_hub not installed" in registry.on_registry_list("me/r", TOKEN)
