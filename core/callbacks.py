"""
core/callbacks.py
==================
Layer 1 — HuggingFace Trainer callbacks.
Imports: core.state (+ transformers).

Patch log
---------
  F-2  : LoggingCallback now records wall-clock timing so ETA can be derived
         from any log record. Added on_train_begin() to capture t0.
         New ETAProgressCallback updates the Gradio progress bar description
         with a human-readable ETA on every training step.
"""

import time
from typing import Optional

from transformers import TrainerCallback

from core.state import app_state


class StopCallback(TrainerCallback):
    """Signal the Trainer to stop cleanly when the UI stop button is pressed."""

    def on_step_end(self, args, state, control, **kwargs):
        if app_state.stop_event.is_set():
            control.should_training_stop = True
        return control


class LoggingCallback(TrainerCallback):
    """Collect per-step loss records for the live loss chart in the UI.

    F-2: Records now include ``elapsed_s`` (seconds since training started)
    and ``eta_s`` (estimated seconds remaining) so callers can display
    progress information outside the Gradio progress bar.
    """

    def __init__(self) -> None:
        self.records: list = []
        self._t0: float = 0.0

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """Capture the training start wall-clock time for ETA calculations."""
        self._t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not (logs and "loss" in logs):
            return

        elapsed = time.time() - self._t0
        steps_done = state.global_step
        total_steps = state.max_steps

        # ETA: extrapolate linearly from elapsed / steps_done.
        # Guards: never divide by 0; clamp to 0 when past the total.
        if steps_done > 0 and total_steps > 0:
            eta_s = max(0.0, elapsed / steps_done * (total_steps - steps_done))
        else:
            eta_s = 0.0

        self.records.append({
            "step":       state.global_step,
            "train_loss": round(logs["loss"], 4),
            "eval_loss":  round(logs.get("eval_loss", float("nan")), 4),
            # F-2 additions: timing data for ETA display and telemetry
            "elapsed_s":  round(elapsed, 1),
            "eta_s":      round(eta_s, 1),
        })


class ETAProgressCallback(TrainerCallback):
    """Update the Gradio progress bar with a step-level ETA on every training step.

    F-2: Designed to be added alongside StopCallback and LoggingCallback.
    When the Gradio progress object is None (e.g. CLI mode), the callback
    is a silent no-op so it is safe to always include.

    Parameters
    ----------
    gradio_progress : Gradio Progress object (gr.Progress()) or None.
    progress_start  : Progress fraction at training start (after model load).
                      Default 0.3 — matches the existing progress(0.3, ...) in sft.py.
    progress_end    : Progress fraction at training end (before model save).
                      Default 0.9 — matches existing progress(0.9, ...) in sft.py.
    """

    def __init__(
        self,
        gradio_progress,
        progress_start: float = 0.3,
        progress_end: float = 0.9,
    ) -> None:
        self._progress = gradio_progress
        self._p_start = progress_start
        self._p_end = progress_end
        self._t0: float = 0.0

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        self._t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Compute ETA and push an updated description to the Gradio progress bar."""
        if self._progress is None:
            return control

        steps_done = state.global_step
        total_steps = state.max_steps

        if total_steps <= 0 or steps_done <= 0:
            return control

        elapsed = time.time() - self._t0
        eta_s = max(0.0, elapsed / steps_done * (total_steps - steps_done))

        # Convert eta_s to human-readable string
        if eta_s < 60:
            eta_str = f"{eta_s:.0f}s"
        elif eta_s < 3600:
            m, s = divmod(int(eta_s), 60)
            eta_str = f"{m}m {s:02d}s"
        else:
            h, rem = divmod(int(eta_s), 3600)
            m = rem // 60
            eta_str = f"{h}h {m:02d}m"

        # Interpolate the progress fraction linearly between p_start and p_end
        frac = steps_done / total_steps
        progress_val = self._p_start + frac * (self._p_end - self._p_start)

        # Grab current loss from log_history (last entry, if available)
        loss_str = ""
        if state.log_history:
            last = state.log_history[-1]
            if "loss" in last:
                loss_str = f" | Loss: {last['loss']:.4f}"

        self._progress(
            progress_val,
            desc=f"Step {steps_done}/{total_steps} | ETA: {eta_str}{loss_str}",
        )
        return control
