"""
export/hub.py
==============
Layer 5 — one-shot model push to HuggingFace Hub.
Imports: config.constants, stdlib.

Functions
---------
push_to_hub — upload an entire model directory to a HF Hub repo

Fix log
-------
  M6 (Medium): Token validation checked `len(token) < 8` — a 9-character
     garbage string passed validation and produced a confusing HuggingFace
     API error with no indication the token itself was malformed. HF write
     tokens are always prefixed with `hf_` and are at least 36 characters
     long. The guard now checks both the prefix and minimum length, giving
     users a clear diagnostic message before any network call is made.
"""

import os

from config.constants import HAS_HUB
from export.utils import validate_hf_token


def push_to_hub(model_path: str, repo_id: str, token: str) -> str:
    """Upload a model directory to HuggingFace Hub.

    Parameters
    ----------
    model_path : local directory containing the saved model
    repo_id    : target HF repo in 'username/model-name' format
    token      : HuggingFace write token (must start with 'hf_', >= 36 chars)

    Returns a status string for display in the UI.
    """
    # Sentinel: strip whitespace and validate against path traversal / malformed input.
    repo_id    = repo_id.strip()    if repo_id    else ""
    token      = token.strip()      if token      else ""
    model_path = model_path.strip() if model_path else ""

    if not model_path or not os.path.isdir(model_path):
        return "❌ No model found. Please train a model first."

    from core.state import validate_path_traversal
    if err := (validate_path_traversal(model_path) or validate_path_traversal(repo_id)):
        return err

    if not repo_id or "/" not in repo_id:
        return "❌ Invalid Repo ID. Format: `username/model-name`"

    # Sentinel: standardized robust token validation.
    if err := validate_hf_token(token):
        return err + "\nGet yours at: https://huggingface.co/settings/tokens"

    if not HAS_HUB:
        return "❌ huggingface_hub not installed. Run: pip install huggingface-hub"

    try:
        from huggingface_hub import HfApi  # lazy

        api = HfApi()
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        return f"✅ Pushed to https://huggingface.co/{repo_id}"
    except Exception as e:
        return f"❌ Push failed: {e}"
