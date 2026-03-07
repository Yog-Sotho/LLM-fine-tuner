"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🧠 Advanced LLM Fine-Tuner  —  v3.2 (PRODUCTION READY)         ║
║  Entry point: launches Gradio UI (no args) or CLI (any arg).                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
-----
  python main.py                        # Launch Gradio UI on :7860
  python main.py --help                 # Show CLI help
  python main.py train --model gpt2 …  # Headless training
"""

import sys


def main() -> None:
    if len(sys.argv) > 1:
        # ── CLI mode: delegate everything to Typer ──────────────────────────
        # This includes --help, train --help, train --model …, reward …, etc.
        # v3.2 Fix #3: ALL non-zero-argument invocations go to Typer so that
        # `python main.py --help` shows CLI usage instead of launching Gradio.
        from llm_fine_tuner.cli.commands import app as cli_app
        print("\n🧠 LLM Fine-Tuner v3.2 CLI")
        print("=" * 60)
        try:
            cli_app(standalone_mode=True)
        except SystemExit as exc:
            sys.exit(exc.code if exc.code is not None else 0)
        except Exception as exc:  # noqa: BLE001
            print(f"\n❌ Unhandled CLI error: {exc}")
            sys.exit(1)
    else:
        # ── Gradio UI mode ───────────────────────────────────────────────────
        import torch
        from llm_fine_tuner.ui.app import build_demo

        print("\n🧠 LLM Fine-Tuner v3.2 — Launching Gradio UI")
        print("=" * 60)
        print(
            f"✅ Hardware: {'GPU available' if torch.cuda.is_available() else 'CPU mode (slow)'}"
        )
        print(
            "✅ v3.2 Fixes: Small Dataset Guard | PPO Reward Float | "
            "CLI --help | QLoRA Checkbox | CUDA dtype"
        )
        demo = build_demo()
        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                show_error=True,
                prevent_thread_lock=False,
                quiet=False,
            )
            print("\n✅ Server terminated cleanly")
        except KeyboardInterrupt:
            print("\n⚠️ Server stopped by user")
        except Exception as exc:  # noqa: BLE001
            print(f"\n❌ Launch failed: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
