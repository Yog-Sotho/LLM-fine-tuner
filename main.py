"""
LLM Fine-Tuner — entry point.

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

LFT_LOG_LEVEL (default INFO) sets how much the app's own modules log.
LFT_ALLOWED_PATHS adds folders the UI may read/write (besides the working directory,
the runs folder and the temp folder).
"""

import logging
import os
import sys

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
# The app's top-level packages; each module logs to logging.getLogger(__name__).
_APP_PACKAGES = ("cli", "config", "core", "data", "export", "inference", "training", "ui")


def _configure_logging(level: str) -> None:
    """Show the app's log records at ``level``; other libraries only log warnings."""
    logging.basicConfig(format="%(levelname)s [%(name)s] %(message)s", level=logging.WARNING)
    if not isinstance(logging.getLevelName(level), int):
        logging.getLogger(__name__).warning("LFT_LOG_LEVEL=%r is not a level; using INFO", level)
        level = "INFO"
    for name in _APP_PACKAGES:
        logging.getLogger(name).setLevel(level)


def _ui_path_roots(environ: dict[str, str]) -> list[str]:
    """Folders the web UI may read and write (paths typed by visitors are checked)."""
    import tempfile

    from config.constants import EXTRA_ALLOWED_PATHS, RUNS_DIR

    roots = [os.getcwd(), RUNS_DIR, tempfile.gettempdir(), *EXTRA_ALLOWED_PATHS]
    if environ.get("GRADIO_TEMP_DIR"):  # where Gradio stores uploads
        roots.append(environ["GRADIO_TEMP_DIR"])
    return roots


def _fail(message: str, code: int) -> None:
    print(f"\n❌ {message}", file=sys.stderr)
    sys.exit(code)


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
    from config.constants import APP_NAME, LOG_LEVEL

    _configure_logging(LOG_LEVEL)
    if len(sys.argv) > 1:
        # Any argument (including --help) goes to the Typer CLI.
        from cli.commands import app as cli_app

        if sys.argv[1] != "--version":
            print(f"\n🧠 {APP_NAME} CLI")
            print("=" * 60)
        try:
            cli_app(standalone_mode=True)
        except SystemExit as exc:
            sys.exit(exc.code if exc.code is not None else 0)
        except Exception as exc:  # noqa: BLE001
            _fail(f"Unhandled CLI error: {exc}", 1)
        return

    if int(os.environ.get("WORLD_SIZE") or 1) > 1:
        # torchrun / accelerate launch start one copy per GPU: that trains data-parallel
        # from the CLI, but would start one web UI per process.
        _fail(
            "Multi-GPU runs use the CLI, e.g. "
            "`accelerate launch main.py train --model ... --data ...` (see docs/10_advanced.md).",
            2,
        )

    import torch

    from core.state import restrict_paths_to
    from ui.app import build_demo, build_theme
    from ui.css import CUSTOM_CSS

    try:
        launch_kwargs = _launch_kwargs(dict(os.environ))
    except ValueError as exc:
        _fail(str(exc), 1)

    print(f"\n🧠 {APP_NAME} — Launching Gradio UI")
    print("=" * 60)
    print(f"✅ Hardware: {'GPU available' if torch.cuda.is_available() else 'CPU mode (slow)'}")
    if warning := _exposure_warning(dict(os.environ), launch_kwargs):
        print(warning, file=sys.stderr)

    restrict_paths_to(_ui_path_roots(dict(os.environ)))
    demo = build_demo()
    try:
        demo.launch(css=CUSTOM_CSS, theme=build_theme(), **launch_kwargs)
    except KeyboardInterrupt:
        print("\n⚠️ Server stopped by user")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Launch failed: {exc}", 1)


if __name__ == "__main__":
    main()
