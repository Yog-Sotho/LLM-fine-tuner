"""
export/registry.py
===================
Layer 5 — versioned model registry on HuggingFace Hub.
Imports: config.constants, stdlib, json, datetime.

Classes
-------
ModelRegistry — versioned upload/list operations against a single HF repo

Functions
---------
on_registry_upload — Gradio UI handler for the Registry Upload button
on_registry_list   — Gradio UI handler for the List Versions button
"""

import json
import os
from datetime import datetime

from config.constants import HAS_HUB, HF_TOKEN_PREFIX, HF_TOKEN_MIN_LEN


class ModelRegistry:
    """Versioned model registry backed by a HuggingFace Hub repository.

    Each upload writes both the model files and a JSON metadata sidecar
    (metadata_v{version}.json) so versions can be listed and inspected
    without downloading the full model weights.
    """

    def __init__(self, repo_id: str, token: str):
        if not HAS_HUB:
            raise ImportError("huggingface_hub not installed. Run: pip install huggingface-hub")
        from huggingface_hub import HfApi, create_repo  # lazy

        self._create_repo = create_repo
        self.api     = HfApi()
        self.repo_id = repo_id
        self.token   = token

    def create_repo_if_needed(self) -> None:
        """Ensure the target repository exists (idempotent)."""
        try:
            self._create_repo(
                self.repo_id,
                token=self.token,
                exist_ok=True,
                repo_type="model",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create repo: {e}")

    def upload_model(self, model_path: str, version: str, metadata: dict) -> str:
        """Upload model files and a versioned metadata sidecar to the registry.

        v2.9 Fix F3: base_model is auto-detected from adapter_config.json
        (PEFT adapter dirs) or config.json (merged models) rather than
        requiring the caller to pass it explicitly.

        Returns a status string.
        """
        if not model_path or not os.path.isdir(model_path):
            return "❌ Invalid model path."

        try:
            self.create_repo_if_needed()

            self.api.upload_folder(
                folder_path=model_path,
                repo_id=self.repo_id,
                repo_type="model",
                token=self.token,
                commit_message=f"Upload version {version}",
            )

            # Auto-detect base model name from saved config files.
            base_model_name = "unknown"
            try:
                adapter_config_path = os.path.join(model_path, "adapter_config.json")
                config_path         = os.path.join(model_path, "config.json")
                if os.path.exists(adapter_config_path):
                    with open(adapter_config_path, "r") as f:
                        adapter_cfg = json.load(f)
                    base_model_name = adapter_cfg.get("base_model_name_or_path", "unknown")
                elif os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                    base_model_name = cfg.get(
                        "_name_or_path", cfg.get("base_model_name", "unknown")
                    )
            except Exception as e:
                base_model_name = f"unknown (error: {e})"

            metadata["base_model"]   = base_model_name
            metadata["version"]      = version
            metadata["uploaded_at"]  = datetime.now().isoformat()

            self.api.upload_file(
                path_or_fileobj=json.dumps(metadata, indent=2).encode(),
                path_in_repo=f"metadata_v{version}.json",
                repo_id=self.repo_id,
                repo_type="model",
                token=self.token,
                commit_message=f"Add metadata for version {version}",
            )

            return (
                f"✅ Version {version} uploaded to "
                f"https://huggingface.co/{self.repo_id}\n"
                f"Base Model: {base_model_name}"
            )

        except Exception as e:
            return f"❌ Upload failed: {e}"

    def list_versions(self) -> str:
        """Return a formatted string listing all versioned uploads in the repo."""
        try:
            files = self.api.list_repo_files(
                repo_id=self.repo_id, repo_type="model", token=self.token
            )
            meta_files = [f for f in files if f.startswith("metadata_v")]
            if not meta_files:
                return "No versioned uploads found in this repository."

            versions_info = []
            for meta_file in sorted(meta_files):
                try:
                    content = self.api.hf_hub_download(
                        repo_id=self.repo_id,
                        filename=meta_file,
                        repo_type="model",
                        token=self.token,
                    )
                    with open(content, "r") as f:
                        meta = json.load(f)
                    ver   = meta_file.replace("metadata_v", "").replace(".json", "")
                    base  = meta.get("base_model", "unknown")
                    notes = meta.get("notes", "")
                    notes = (notes[:50] + "...") if len(notes) > 50 else notes
                    versions_info.append(f"• v{ver}: {base} | {notes}")
                except Exception:
                    versions_info.append(f"• {meta_file}")

            return "Versions found:\n" + "\n".join(versions_info)

        except Exception as e:
            return f"❌ Could not list versions: {e}"


# ── Gradio UI handlers ─────────────────────────────────────────────────────

def on_registry_upload(
    model_path_state: str,
    registry_repo_id: str,
    registry_token: str,
    registry_version: str,
    registry_notes: str,
) -> str:
    """Handler for the Registry Upload button in the Share tab."""
    # Sentinel: strip whitespace and validate against path traversal / malformed input.
    registry_repo_id = registry_repo_id.strip() if registry_repo_id else ""
    registry_token   = registry_token.strip()   if registry_token   else ""
    registry_version = registry_version.strip() if registry_version else ""
    model_path_state = model_path_state.strip() if model_path_state else ""

    from core.state import validate_path_traversal, validate_identifier
    if err := (
        validate_path_traversal(model_path_state)
        or validate_path_traversal(registry_repo_id)
        or validate_identifier(registry_version)
    ):
        return err

    if not registry_version:
        return "❌ Please enter a valid version tag."

    if not registry_repo_id or "/" not in registry_repo_id:
        return "❌ Invalid Repo ID. Format: username/model-name"

    # Sentinel: standardized robust token validation.
    if (
        not registry_token
        or not registry_token.startswith(HF_TOKEN_PREFIX)
        or len(registry_token) < HF_TOKEN_MIN_LEN
    ):
        return (
            "❌ Invalid Hugging Face write token.\n"
            f"Tokens start with '{HF_TOKEN_PREFIX}' and are at least {HF_TOKEN_MIN_LEN} characters long."
        )
    if not model_path_state or not os.path.isdir(model_path_state):
        return "❌ No trained model found. Train a model first."

    try:
        reg = ModelRegistry(registry_repo_id, registry_token)
        metadata = {
            "notes": registry_notes or "",
            "trained_with": "LLM Fine-Tuner v3.2",
        }
        return reg.upload_model(model_path_state, registry_version, metadata)
    except Exception as e:
        return f"❌ Registry upload failed: {e}"


def on_registry_list(registry_repo_id: str, registry_token: str) -> str:
    """Handler for the List Versions button in the Share tab."""
    # Sentinel: strip whitespace and validate against path traversal.
    registry_repo_id = registry_repo_id.strip() if registry_repo_id else ""
    registry_token   = registry_token.strip()   if registry_token   else ""

    from core.state import validate_path_traversal
    if err := validate_path_traversal(registry_repo_id):
        return err

    if not registry_repo_id or "/" not in registry_repo_id:
        return "❌ Invalid Repo ID."

    # Sentinel: standardized robust token validation.
    if (
        not registry_token
        or not registry_token.startswith(HF_TOKEN_PREFIX)
        or len(registry_token) < HF_TOKEN_MIN_LEN
    ):
        return (
            "❌ Invalid Hugging Face write token.\n"
            f"Tokens start with '{HF_TOKEN_PREFIX}' and are at least {HF_TOKEN_MIN_LEN} characters long."
        )

    try:
        reg = ModelRegistry(registry_repo_id, registry_token)
        return reg.list_versions()
    except Exception as e:
        return f"❌ {e}"
