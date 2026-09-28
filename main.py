"""
LLM Fine-Tuner v3.2 — entry point.

Usage
-----
  python main.py                        # Launch the Gradio UI
  python main.py --help                 # Show CLI help
  python main.py train --model gpt2 …  # Headless training

UI launch settings come from the environment:

  GRADIO_SERVER_NAME  bind address (Gradio default: 127.0.0.1 — local only)
  GRADIO_SERVER_PORT  port (Gradio default: first free port from 7860)
  GRADIO_SHARE / SHARE  "true" creates a public share link
  GRADIO_AUTH         "user:password" pairs, comma-separated, to require a login
"""

import os
import sys

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _parse_auth(raw: str | None) -> list[tuple[str, str]] | None:
    """Parse GRADIO_AUTH ("user:pass,user2:pass2") into Gradio's auth list."""
    if not raw or not raw.strip():
        return None
    pairs = []
    for item in raw.split(","):
        user, sep, password = item.strip().partition(":")
        if not sep or not user or not password:
            raise ValueError("GRADIO_AUTH must be 'user:password' pairs separated by commas.")
        pairs.append((user, password))
    return pairs


def _launch_kwargs(environ: dict[str, str]) -> dict:
    """Build demo.launch() kwargs; bind address and port are left to Gradio's own env vars."""
    share_raw = environ.get("GRADIO_SHARE", environ.get("SHARE", "false"))
    kwargs: dict = {
        "share": share_raw.strip().lower() in {"1", "true", "yes"},
        "auth": _parse_auth(environ.get("GRADIO_AUTH")),
        "show_error": True,
    }
    return kwargs


def _exposure_warning(environ: dict[str, str], launch_kwargs: dict) -> str | None:
    """Return a warning when the UI is reachable beyond localhost without a login."""
    host = environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    exposed = launch_kwargs["share"] or host not in _LOOPBACK_HOSTS
    if exposed and not launch_kwargs["auth"]:
        return (
            "⚠️ The UI is reachable from other machines without a login. "
            "Anyone who can reach it can start training jobs on this server. "
            "Set GRADIO_AUTH=user:password to require authentication."
        )
    return None


def main() -> None:
    if len(sys.argv) > 1:
        # Any argument (including --help) goes to the Typer CLI.
        from cli.commands import app as cli_app

        print("\n🧠 LLM Fine-Tuner v3.2 CLI")
        print("=" * 60)
        try:
            cli_app(standalone_mode=True)
        except SystemExit as exc:
            sys.exit(exc.code if exc.code is not None else 0)
        except Exception as exc:  # noqa: BLE001
            print(f"\n❌ Unhandled CLI error: {exc}")
            sys.exit(1)
        return

    if int(os.environ.get("WORLD_SIZE") or 1) > 1:
        # torchrun / accelerate launch start one copy per GPU: that trains data-parallel
        # from the CLI, but would start one web UI per process.
        print(
            "\n❌ Multi-GPU runs use the CLI, e.g. "
            "`accelerate launch main.py train --model ... --data ...` (see docs/10_advanced.md)."
        )
        sys.exit(2)

    import torch

    from ui.app import build_demo, build_theme
    from ui.css import CUSTOM_CSS

    try:
        launch_kwargs = _launch_kwargs(dict(os.environ))
    except ValueError as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)

    print("\n🧠 LLM Fine-Tuner v3.2 — Launching Gradio UI")
    print("=" * 60)
    print(f"✅ Hardware: {'GPU available' if torch.cuda.is_available() else 'CPU mode (slow)'}")
    if warning := _exposure_warning(dict(os.environ), launch_kwargs):
        print(warning)

    demo = build_demo()
    try:
        demo.launch(css=CUSTOM_CSS, theme=build_theme(), **launch_kwargs)
    except KeyboardInterrupt:
        print("\n⚠️ Server stopped by user")
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Launch failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
