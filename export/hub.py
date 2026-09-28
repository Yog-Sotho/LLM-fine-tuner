"""
export/hub.py
==============
Layer 5 — one-shot model push to HuggingFace Hub.
Imports: config.constants, core.state, stdlib, huggingface_hub (lazy).

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

from config.constants import HAS_HUB, HF_TOKEN_MIN_LEN, HF_TOKEN_PREFIX
from core.state import redact_sensitive_info


def _complete_card_metadata(api, model_path: str) -> None:
    """Check the model card's ``base_model`` against the Hub before uploading.

    Uses the canonical id (e.g. ``gpt2`` → ``openai-community/gpt2``) and copies the
    base model's license. A base_model that is not on the Hub (private, deleted,
    typo) is dropped: the Hub rejects cards with an invalid base_model.
    """
    from huggingface_hub import ModelCard  # lazy
    from huggingface_hub.errors import HfHubHTTPError

    path = os.path.join(model_path, "README.md")
    if not os.path.isfile(path):
        return
    card = ModelCard.load(path)
    base = card.data.base_model
    if not isinstance(base, str) or not base:
        return
    try:
        info = api.model_info(base)
    except HfHubHTTPError:
        card.data.base_model = None
    else:
        card.data.base_model = info.id
        base_card = info.card_data
        if base_card is not None and base_card.license and not card.data.license:
            card.data.license = base_card.license
            for key in ("license_name", "license_link"):  # set when license is "other"
                if base_card.get(key):
                    card.data[key] = base_card.get(key)
    card.save(path)


def push_to_hub(model_path: str, repo_id: str, token: str) -> str:
    """Upload a model directory to HuggingFace Hub.

    Parameters
    ----------
    model_path : local directory containing the saved model
    repo_id    : target HF repo in 'username/model-name' format
    token      : HuggingFace write token (must start with 'hf_', >= 36 chars)

    Returns a status string for display in the UI.
    """
    # Strip whitespace and validate against path traversal / malformed input.
    repo_id = repo_id.strip() if repo_id else ""
    token = token.strip() if token else ""
    model_path = model_path.strip() if model_path else ""

    from core.state import validate_path_traversal

    if err := (
        validate_path_traversal(model_path)
        or validate_path_traversal(repo_id)
        or validate_path_traversal(token)
    ):
        return err

    if not model_path or not os.path.isdir(model_path):
        return "❌ No model found. Please train a model first."

    if not repo_id or "/" not in repo_id:
        return "❌ Invalid Repo ID. Format: `username/model-name`"

    # M6 FIX: validate the token format properly — HF tokens are `hf_` + 33 chars.
    if not token or not token.startswith(HF_TOKEN_PREFIX) or len(token) < HF_TOKEN_MIN_LEN:
        return (
            "❌ Invalid Hugging Face write token.\n"
            f"Tokens start with '{HF_TOKEN_PREFIX}' and are at least {HF_TOKEN_MIN_LEN} characters long.\n"
            "Get yours at: https://huggingface.co/settings/tokens"
        )
    if not re.fullmatch(r"hf_[A-Za-z0-9_]+", token):
        return (
            "❌ Token contains invalid characters.\n"
            "Get yours at: https://huggingface.co/settings/tokens"
        )
    if not HAS_HUB:
        return "❌ huggingface_hub not installed. Run: pip install huggingface-hub"

    try:
        from huggingface_hub import HfApi  # lazy

        api = HfApi(token=token)
        _complete_card_metadata(api, model_path)
        # upload_folder needs an existing repo (404 otherwise); visibility follows
        # the account's default for new repos.
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=model_path, repo_id=repo_id, repo_type="model")
        return f"✅ Pushed to https://huggingface.co/{repo_id}"
    except Exception as e:
        err_msg = redact_sensitive_info(str(e))
        return f"❌ Push failed: {err_msg}"
