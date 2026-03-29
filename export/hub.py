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

# HuggingFace write tokens always start with this prefix and are >= 36 chars.
_HF_TOKEN_PREFIX: str = "hf_"
_HF_TOKEN_MIN_LEN: int = 36


def push_to_hub(model_path: str, repo_id: str, token: str) -> str:
    """Upload a model directory to HuggingFace Hub.

    Parameters
    ----------
    model_path : local directory containing the saved model
    repo_id    : target HF repo in 'username/model-name' format
    token      : HuggingFace write token (must start with 'hf_', >= 36 chars)

    Returns a status string for display in the UI.
    """
    if not model_path or not os.path.isdir(model_path):
        return "❌ No model found. Please train a model first."
    if not repo_id or "/" not in repo_id:
        return "❌ Invalid Repo ID. Format: `username/model-name`"
    # M6 FIX: validate the token format properly — HF tokens are `hf_` + 33 chars.
    if (
        not token
        or not token.startswith(_HF_TOKEN_PREFIX)
        or len(token) < _HF_TOKEN_MIN_LEN
    ):
        return (
            "❌ Invalid Hugging Face write token.\n"
            "Tokens start with 'hf_' and are at least 36 characters long.\n"
            "Get yours at: https://huggingface.co/settings/tokens"
        )
    if not HAS_HUB:
        return "❌ huggingface_hub not installed. Run: pip install huggingface-hub"

    try:
        from huggingface_hub import HfApi  # lazy

        api = HfApi()
        # L-6 FIX: Distinguish between creating a new repo vs pushing to an existing one.
        repo_existed = True
        try:
            api.repo_info(repo_id=repo_id, token=token, repo_type="model")
        except Exception:
            repo_existed = False

        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        action = "Pushed to" if repo_existed else "Created and pushed to"
        return f"✅ {action} https://huggingface.co/{repo_id}"
    except Exception as e:
        return f"❌ Push failed: {e}"
