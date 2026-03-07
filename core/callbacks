"""
core/callbacks.py
==================
Layer 1 — HuggingFace Trainer callbacks.
Imports: core.state (+ transformers).
"""

from transformers import TrainerCallback
from core.state import app_state


class StopCallback(TrainerCallback):
    """Signal the Trainer to stop cleanly when the UI stop button is pressed."""

    def on_step_end(self, args, state, control, **kwargs):
        if app_state.stop_event.is_set():
            control.should_training_stop = True
        return control


class LoggingCallback(TrainerCallback):
    """Collect per-step loss records for the live loss chart in the UI."""

    def __init__(self) -> None:
        self.records: list = []

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs and "loss" in logs:
            self.records.append({
                "step":       state.global_step,
                "train_loss": round(logs["loss"], 4),
                "eval_loss":  round(logs.get("eval_loss", float("nan")), 4),
            })
