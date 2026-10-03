"""core/callbacks.py and data/augmentation.py."""

import math
import threading
from types import SimpleNamespace

import pandas as pd
import pytest
from datasets import Dataset

import data.augmentation as aug
from core.callbacks import (
    ETAProgressCallback,
    LoggingCallback,
    StopCallback,
    final_train_loss,
)

# ── Callbacks ──────────────────────────────────────────────────────────────


def _state(step, max_steps=10, log_history=()):
    return SimpleNamespace(global_step=step, max_steps=max_steps, log_history=list(log_history))


def test_stop_callback_stops_only_when_its_event_is_set():
    event = threading.Event()
    cb = StopCallback(event)
    control = SimpleNamespace(should_training_stop=False)
    assert not cb.on_step_end(None, _state(1), control).should_training_stop
    event.set()
    assert cb.on_step_end(None, _state(2), control).should_training_stop


def test_logging_callback_records_train_and_eval_loss():
    cb = LoggingCallback()
    cb.on_train_begin(None, _state(0), None)
    cb.on_log(None, _state(5), None, logs={"loss": 2.345678})
    # Transformers logs evaluation separately from training, at the same step.
    cb.on_log(None, _state(5), None, logs={"eval_loss": 2.1, "eval_runtime": 0.1})
    cb.on_log(None, _state(8), None, logs={"eval_loss": 1.9})  # eval-only step
    cb.on_log(None, _state(10), None, logs={"train_runtime": 3.0})  # ignored
    first, eval_only = cb.records
    assert (first["step"], first["train_loss"], first["eval_loss"]) == (5, 2.3457, 2.1)
    assert first["eta_s"] >= 0 and first["elapsed_s"] >= 0
    assert eval_only["step"] == 8 and math.isnan(eval_only["train_loss"])
    assert eval_only["eval_loss"] == 1.9
    assert final_train_loss(cb.records) == 2.3457


def test_eval_loss_is_not_merged_into_a_warning_note():
    cb = LoggingCallback()
    cb.records.append({"step": 0, "train_loss": 0.0, "eval_loss": float("nan"), "note": "⚠️"})
    cb.on_log(None, _state(0), None, logs={"eval_loss": 3.0})
    assert len(cb.records) == 2 and math.isnan(cb.records[0]["eval_loss"])


def test_final_train_loss_skips_notes_and_eval_only_records():
    assert final_train_loss([]) == "N/A"
    assert final_train_loss([{"step": 0, "train_loss": 0.0, "note": "⚠️ packing skipped"}]) == "N/A"
    assert final_train_loss([{"train_loss": 1.5}, {"train_loss": float("nan")}]) == 1.5


@pytest.mark.parametrize(
    ("elapsed", "text"), [(10.0, "ETA: 40s"), (100.0, "ETA: 6m 40s"), (1000.0, "ETA: 1h 06m")]
)
def test_eta_progress_callback_reports_step_eta_and_loss(monkeypatch, elapsed, text):
    import core.callbacks as callbacks

    calls = []
    cb = ETAProgressCallback(lambda value, desc: calls.append((value, desc)))
    monkeypatch.setattr(callbacks.time, "time", lambda: 1000.0)
    cb.on_train_begin(None, None, None)
    monkeypatch.setattr(callbacks.time, "time", lambda: 1000.0 + elapsed)
    cb.on_step_end(None, _state(2, log_history=[{"loss": 1.23456}]), None)
    ((value, desc),) = calls
    assert value == pytest.approx(0.3 + 0.2 * 0.6)
    assert desc.startswith("Step 2/10 | " + text) and desc.endswith("| Loss: 1.2346")


def test_eta_progress_callback_is_silent_without_progress_or_steps():
    calls = []
    ETAProgressCallback(None).on_step_end(None, _state(1), "c")
    ETAProgressCallback(calls.append).on_step_end(None, _state(0), "c")
    ETAProgressCallback(calls.append).on_step_end(None, _state(1, max_steps=0), "c")
    assert calls == []


# ── Augmentation ───────────────────────────────────────────────────────────


class _File:
    def __init__(self, path):
        self.name = str(path)


class _Augmenter:
    def __init__(self, result=None, error=False):
        self.result, self.error = result, error

    def augment(self, texts):
        if self.error:
            raise RuntimeError("augmenter broke")
        return self.result if self.result is not None else [t.upper() for t in texts]


@pytest.fixture
def fake_nlpaug(monkeypatch):
    """Replace nlpaug's word augmenters (synonyms need WordNet data)."""
    import nlpaug.augmenter.word as naw

    chosen = {}

    def factory(name, augmenter):
        def make(*args, **kwargs):
            chosen["name"] = name
            return augmenter["value"]

        return make

    augmenter = {"value": _Augmenter()}
    for name in ("SynonymAug", "RandomWordAug", "SpellingAug"):
        monkeypatch.setattr(naw, name, factory(name, augmenter))
    return chosen, augmenter


@pytest.mark.parametrize(
    ("aug_type", "cls"),
    [("synonym", "SynonymAug"), ("random_word", "RandomWordAug"), ("spelling", "SpellingAug"),
     ("unknown", "SynonymAug")],
)  # fmt: skip
def test_augmentation_interleaves_copies(fake_nlpaug, aug_type, cls):
    chosen, _ = fake_nlpaug
    ds = Dataset.from_dict({"instruction": ["a b", "c d"], "output": ["x", "y"]})
    out, msg = aug.augment_dataset_v27(ds, augmentation_factor=3, aug_type=aug_type)
    assert chosen["name"] == cls and "Augmented: 6 examples (×3)" in msg
    assert out["instruction"] == ["a b", "A B", "A B", "c d", "C D", "C D"]
    assert out["output"] == ["x", "x", "x", "y", "y", "y"]


def test_augmentation_keeps_row_count_when_the_augmenter_misbehaves(fake_nlpaug):
    _, augmenter = fake_nlpaug
    ds = Dataset.from_dict({"text": ["one", "two", "three"]})
    augmenter["value"] = _Augmenter(result=["ONLY ONE"])  # too short: padded with originals
    assert aug.augment_dataset_v27(ds, 2)[0]["text"][:4] == ["one", "ONLY ONE", "two", "two"]
    augmenter["value"] = _Augmenter(result=["A", "B", "C", "D"])  # too long: truncated
    assert len(aug.augment_dataset_v27(ds, 2)[0]) == 6
    augmenter["value"] = _Augmenter(error=True)  # failure: originals are copied
    assert aug.augment_dataset_v27(ds, 2)[0]["text"] == [
        "one",
        "one",
        "two",
        "two",
        "three",
        "three",
    ]


def test_preference_data_is_not_augmented(fake_nlpaug):
    ds = Dataset.from_dict({"prompt": ["p"], "chosen": ["c"], "rejected": ["r"]})
    out, msg = aug.augment_dataset_v27(ds, 2)
    assert out is ds and msg.startswith("⚠️ Augmentation needs a 'text' or 'instruction' column")


def test_augmentation_edge_cases(fake_nlpaug, monkeypatch):
    ds = Dataset.from_dict({"text": ["one"]})
    assert aug.augment_dataset_v27(ds, 1)[1].startswith("✅ No augmentation")
    monkeypatch.setattr(aug, "HAS_NLPAUG", False)
    same, msg = aug.augment_dataset_v27(ds, 2)
    assert same is ds and "nlpaug not installed" in msg


def test_quality_filter_by_format():
    sft = Dataset.from_dict({"instruction": ["hi", "a" * 40], "output": ["x", "b" * 40]})
    kept, msg = aug.quality_filter_v27(sft, min_length=20, max_length=100)
    assert len(kept) == 1 and "Removed: 1" in msg
    text = Dataset.from_dict({"text": ["short", "long enough text", "x" * 500]})
    assert aug.quality_filter_v27(text, 10, 100)[0]["text"] == ["long enough text"]
    dpo = Dataset.from_dict({"prompt": ["question here", "q"], "chosen": ["good answer", "g"],
                             "rejected": ["bad answer!", "b"]})  # fmt: skip
    assert len(aug.quality_filter_v27(dpo, 5, 100, is_dpo=True)[0]) == 1
    other = Dataset.from_dict({"label": [1, 2]})
    assert len(aug.quality_filter_v27(other, 5, 10)[0]) == 2
    empty = Dataset.from_dict({"text": []})
    assert aug.quality_filter_v27(empty)[1] == "✅ Dataset is already empty."


def test_augment_and_filter_buttons(fake_nlpaug, tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"instruction": ["short", "a longer instruction"],
                  "output": ["ok", "a longer answer"]}).to_csv(path, index=False)  # fmt: skip
    msg, preview, stats, ds = aug.on_augment_click(
        _File(path), "SFT", 2, "synonym", progress=lambda *a, **k: None
    )
    assert msg.startswith("✅ Augmentation complete") and len(ds) == 4 and preview["visible"]
    assert "**Augmented:** 4" in stats["value"]
    msg, preview, stats, ds = aug.on_quality_filter_click(
        _File(path), "SFT", 20, 100, progress=lambda *a, **k: None
    )
    assert len(ds) == 1 and "**After filter:** 1" in stats["value"]


@pytest.mark.parametrize("handler", [aug.on_augment_click, aug.on_quality_filter_click])
def test_augment_and_filter_buttons_report_problems(handler, tmp_path):
    assert handler(None, "SFT", 2, 3)[0] == "❌ Upload a dataset first."
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    msg, _, _, ds = handler(_File(bad), "SFT", 2, 3, progress=lambda *a, **k: None)
    assert msg.startswith("❌") and ds is None


def _noop(*a, **k):
    return None


def test_filter_then_augment_keeps_the_filter(fake_nlpaug, tmp_path):
    """BUG: each button re-read the uploaded file, so augmenting after filtering threw the
    filter away (the docs recommend exactly that order)."""
    path = tmp_path / "d.csv"
    pd.DataFrame({"instruction": ["short", "a longer instruction", "another long instruction"],
                  "output": ["ok", "a longer answer", "another long answer"]}).to_csv(
        path, index=False)  # fmt: skip
    _, _, _, filtered = aug.on_quality_filter_click(_File(path), "SFT", 20, 100, progress=_noop)
    assert len(filtered) == 2
    _, _, stats, augmented = aug.on_augment_click(
        _File(path), "SFT", 2, "synonym", filtered, progress=_noop
    )
    assert len(augmented) == 4 and "**Original:** 2" in stats["value"]
    assert "short" not in augmented["instruction"]


def test_prepared_data_without_a_file_can_be_augmented_and_filtered(fake_nlpaug):
    """Hub loads and data created from documents live only in the prepared state."""
    prepared = Dataset.from_dict({"instruction": ["what is a question here?", "hi"],
                                  "output": ["a sufficiently long answer", "ok"]})  # fmt: skip
    msg, _, _, augmented = aug.on_augment_click(None, "SFT", 2, "synonym", prepared, progress=_noop)
    assert msg.startswith("✅") and len(augmented) == 4
    _, _, _, filtered = aug.on_quality_filter_click(None, "SFT", 20, 100, prepared, progress=_noop)
    assert len(filtered) == 1


def test_augment_and_filter_use_the_column_mapping(fake_nlpaug, tmp_path):
    path = tmp_path / "custom.csv"
    pd.DataFrame({"q": ["a question long enough"], "a": ["an answer long enough"]}).to_csv(
        path, index=False
    )
    msg, _, _, filtered = aug.on_quality_filter_click(
        _File(path), "SFT", 20, 100, None, "q", "a", None, progress=_noop
    )
    assert msg.startswith("✅") and filtered["instruction"] == ["a question long enough"]
    _, _, _, augmented = aug.on_augment_click(
        _File(path), "SFT", 2, "synonym", None, "q", "a", None, progress=_noop
    )
    assert len(augmented) == 2
