"""
core/run_config.py
===================
Layer 1 — reproducibility records for training runs.

Every trainer writes ``run_config.yaml`` next to the model it saves: what was
trained (mode, base model, hyperparameters, seed), on which data (row count +
SHA-256 of the rows actually used) and with which library versions. The CLI can
replay an SFT/DPO run from it (``train --config``). The README.md model card is
generated from the same record.
"""

import glob
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from importlib import metadata

import yaml

from config.constants import (
    APP_VERSION,
    HUB_DATASET_ID_PATTERN,
    RUN_CONFIG_FILENAME,
    RUN_NAME_PATTERN,
    RUNS_DIR,
    TRACKING_BACKENDS,
)
from core.model_card import write_model_card

_LIBRARIES = ("torch", "transformers", "trl", "peft", "datasets", "accelerate")
_OPTIONAL_LIBRARIES = ("sentence-transformers",)  # recorded only when installed


def dataset_fingerprint(dataset) -> dict:
    """Row count, columns and a SHA-256 over the rows (independent of `datasets` internals)."""
    digest = hashlib.sha256()
    for batch in dataset.iter(batch_size=1000):
        columns = sorted(batch)
        for row in zip(*(batch[c] for c in columns), strict=True):
            digest.update(json.dumps(dict(zip(columns, row, strict=True)), default=str).encode())
            digest.update(b"\n")
    fingerprint = {
        "rows": len(dataset),
        "columns": sorted(dataset.column_names),
        "sha256": digest.hexdigest(),
    }
    # load_hub_dataset() records the Hub id; it links the model card to the dataset.
    name = getattr(getattr(dataset, "info", None), "dataset_name", None)
    if isinstance(name, str) and re.fullmatch(HUB_DATASET_ID_PATTERN, name):
        fingerprint["hub_id"] = name
    return fingerprint


def library_versions() -> dict[str, str]:
    versions = {"llm-fine-tuner": APP_VERSION}
    for name in _LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    for name in _OPTIONAL_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def save_run_config(output_dir: str, *, mode: str, model: str, dataset, **settings) -> str:
    """Write run_config.yaml and the README.md model card into ``output_dir``.

    Returns the path of run_config.yaml.

    ``settings`` holds everything needed to repeat the run (hyperparameters,
    PEFT options, seed, tracking backend, ...).
    """
    record = {
        "mode": mode,
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **settings,
        "dataset": dataset_fingerprint(dataset) if dataset is not None else None,
        "libraries": library_versions(),
    }
    path = os.path.join(output_dir, RUN_CONFIG_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(record, f, sort_keys=False, allow_unicode=True)
    # Replaces the card TRL writes on save: ours has the dataset, settings and seed,
    # and never puts a local path in base_model (the Hub rejects such cards).
    write_model_card(output_dir, record)
    return path


def load_run_config(path: str) -> dict:
    """Read a run_config.yaml (safe loader: plain data only, never Python objects)."""
    with open(path, encoding="utf-8") as f:
        record = yaml.safe_load(f)
    if not isinstance(record, dict) or "mode" not in record or "model" not in record:
        raise ValueError(f"{path} is not a run config (missing 'mode'/'model').")
    return record


def latest_checkpoint(output_dir: str) -> str | None:
    """Newest ``checkpoint-<step>`` folder in ``output_dir`` (to resume from), or None."""
    ckpts = [
        path
        for path in glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if path.rsplit("-", 1)[-1].isdigit()
    ]
    return max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]), default=None)


def resolve_report_to(value: str | None) -> str:
    """Validate an experiment-tracking choice against the installed backends."""
    choice = (value or "none").strip().lower()
    if choice not in TRACKING_BACKENDS:
        raise ValueError(
            f"Experiment tracking '{choice}' is not available. Installed: {TRACKING_BACKENDS}"
        )
    return choice


def run_dir_for(run_name: str) -> str:
    """Return <RUNS_DIR>/<run_name> after validating the name (no paths, no traversal)."""
    if not re.fullmatch(RUN_NAME_PATTERN, run_name or ""):
        raise ValueError(
            "Run name must be 1–64 characters: letters, digits, '.', '_' or '-', "
            "starting with a letter or digit."
        )
    return os.path.join(RUNS_DIR, run_name)


def new_run_name(mode: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{mode}"
