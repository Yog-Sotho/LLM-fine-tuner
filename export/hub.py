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
import re

from config.constants import HAS_HUB, HF_TOKEN_PREFIX, HF_TOKEN_MIN_LEN


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

    # M6 FIX: validate the token format properly — HF tokens are `hf_` + 33 chars.
    # M-BUG08 FIX: Added regex check to reject whitespace and control characters.
    # Previously a token like "hf_" + " " * 33 passed the length check but failed
    # at the HuggingFace API with a cryptic network error instead of a clear message.
    if (
        not token
        or not token.startswith(HF_TOKEN_PREFIX)
        or len(token) < HF_TOKEN_MIN_LEN
    ):
        return (
            "❌ Invalid Hugging Face write token.\n"
            f"Tokens start with '{HF_TOKEN_PREFIX}' and are at least {HF_TOKEN_MIN_LEN} characters long.\n"
            "Get yours at: https://huggingface.co/settings/tokens"
        )
    if not re.fullmatch(r'hf_[A-Za-z0-9]+', token):
        return (
            "❌ Token contains invalid characters.\n"
            "HuggingFace tokens only contain letters and digits after 'hf_'.\n"
            "Get yours at: https://huggingface.co/settings/tokens"
        )
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
