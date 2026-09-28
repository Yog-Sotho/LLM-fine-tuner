"""
inference/remote.py
===================
Layer 4 — chat with any OpenAI-compatible server (vLLM, llama.cpp's llama-server,
TGI, or a hosted API) through ``/v1/chat/completions``.

The request is made by the machine running this app: in a shared deployment it can
reach hosts the users can't, so only http(s) URLs are accepted.
"""

from urllib.parse import urlparse

import httpx

from core.state import redact_sensitive_info


def api_base(url: str) -> str:
    """Normalise ``http://host:8000`` or ``…/v1/`` to ``http://host:8000/v1``."""
    url = (url or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Endpoint must be an http(s) URL, e.g. http://127.0.0.1:8000")
    return url if url.endswith("/v1") else f"{url}/v1"


def remote_chat(
    url: str,
    prompt: str,
    model: str = "",
    api_key: str = "",
    system_prompt: str = "",
    max_tokens: int = 256,
    temperature: float = 0.7,
    timeout: float = 120.0,
) -> str:
    """Send one chat turn and return the reply text. Raises ValueError with a readable message."""
    if not (prompt or "").strip():
        raise ValueError("Write a prompt.")
    base = api_base(url)
    headers = {"Authorization": f"Bearer {api_key.strip()}"} if (api_key or "").strip() else {}
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
        {"role": "user", "content": prompt}
    ]
    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            if not (model or "").strip():  # the server's first model
                listed = client.get(f"{base}/models")
                listed.raise_for_status()
                models = listed.json().get("data") or []
                if not models:
                    raise ValueError("The server lists no models; enter the model name.")
                model = models[0]["id"]
            reply = client.post(
                f"{base}/chat/completions",
                json={"model": model.strip(), "messages": messages,
                      "max_tokens": int(max_tokens), "temperature": float(temperature)},
            )  # fmt: skip
            reply.raise_for_status()
            return reply.json()["choices"][0]["message"]["content"] or ""
    except httpx.HTTPStatusError as e:
        body = redact_sensitive_info(e.response.text[:300])
        raise ValueError(f"Server answered {e.response.status_code}: {body}") from e
    except httpx.HTTPError as e:
        raise ValueError(f"Cannot reach {base}: {redact_sensitive_info(str(e))}") from e
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError("The server's answer is not an OpenAI chat completion.") from e
