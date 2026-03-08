"""
export/hub.py
==============
Layer 5 — one-shot model push to HuggingFace Hub.
Imports: config.constants, stdlib.

Functions
---------
push_to_hub — upload an entire model directory to a HF Hub repo
"""

import os

from config.constants import HAS_HUB


def push_to_hub(model_path: str, repo_id: str, token: str) -> str:
    """Upload a model directory to HuggingFace Hub.

    Parameters
    ----------
    model_path : local directory containing the saved model
    repo_id    : target HF repo in 'username/model-name' format
    token      : HuggingFace write token

    Returns a status string for display in the UI.
    """
    if not model_path or not os.path.isdir(model_path):
        return "❌ No model found. Please train a model first."
    if not repo_id or "/" not in repo_id:
        return "❌ Invalid Repo ID. Format: `username/model-name`"
    if not token or len(token) < 8:
        return "❌ Please provide a valid Hugging Face write token."
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
