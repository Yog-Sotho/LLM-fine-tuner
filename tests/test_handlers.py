"""ui/handlers.py — the Gradio glue, with the heavy calls (training, inference) faked."""

import os

import pandas as pd
import pytest
from datasets import Dataset

import core.run_config as rc
import ui.handlers as handlers
from core.state import app_state


class _File:
    """What Gradio passes for an uploaded file."""

    def __init__(self, path):
        self.name = str(path)


def _csv(tmp_path, name="d.csv", **columns):
    path = tmp_path / name
    pd.DataFrame(columns).to_csv(path, index=False)
    return _File(path)


SFT = {"instruction": ["Say hi", "Say bye", "Count"], "output": ["Hello there", "Bye now", "1 2 3"]}


@pytest.fixture
def runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(rc, "RUNS_DIR", str(runs))
    monkeypatch.setattr(handlers, "RUNS_DIR", str(runs))
    return runs


@pytest.fixture
def fake_train(monkeypatch):
    """Record train_model's arguments and write a tiny 'model' instead of training."""
    calls = []

    def fake(model_name, ds, output_dir, hyperparams, *args, **kwargs):
        calls.append({"model": model_name, "ds": ds, "out": output_dir, "hp": hyperparams,
                      "args": args, "kwargs": kwargs})  # fmt: skip
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            f.write("{}")
        return "✅ trained", [{"step": 1}]

    monkeypatch.setattr(handlers, "train_model", fake)
    return calls


def _train(file, **overrides):
    args = dict(
        file=file, model_choice="gpt2", custom_model="", training_preset="Custom",
        peft_method="LoRA", use_lora=True, lora_rank=8, lora_alpha=16,
        prefix_tuning_num_virtual_tokens=30, prefix_tuning_token_dim=512,
        prefix_tuning_num_layers=2, prompt_tuning_num_virtual_tokens=20,
        lr=2e-4, epochs=1.0, bs=4.0, grad_accum=1.0,
        max_len=128.0, warmup=0.0, early_stop=3, lr_sched="cosine", grad_ckpt=False,
        resume=False, col_inst=None, col_out=None, col_text=None, use_unsloth=False,
        use_chat_template=False, system_prompt="", training_mode="SFT (Supervised Fine-Tuning)",
        dpo_beta=0.1, heretic_mode=False, run_name="exp", progress=None,
    )  # fmt: skip
    args.update(overrides)
    return handlers.on_train_click(**args)


# ── Training ───────────────────────────────────────────────────────────────


def test_train_click_trains_zips_and_casts_numbers(runs, fake_train, tmp_path):
    msg, zip_path, model_dir, records = _train(_csv(tmp_path, **SFT))
    assert msg.startswith("✅ trained") and records == [{"step": 1}]
    assert model_dir == str(runs / "exp") and os.path.isfile(zip_path)
    (call,) = fake_train
    assert call["model"] == "gpt2" and len(call["ds"]) == 3
    # Gradio delivers sliders as floats; the trainer needs ints.
    assert call["hp"]["batch_size"] == 4 and isinstance(call["hp"]["batch_size"], int)
    assert call["kwargs"]["training_mode"] == "sft" and call["kwargs"]["seed"] == 42
    app_state.session_for(None).release("zip")


@pytest.mark.parametrize(
    ("preset", "epochs", "lr"),
    [
        ("Quick (1 epoch)", 1, 5e-4),
        ("Balanced (3 epochs)", 3, 2e-4),
        ("Accurate (5 epochs)", 5, 1e-4),
    ],
)
def test_training_presets_set_epochs_and_learning_rate(
    runs, fake_train, tmp_path, preset, epochs, lr
):
    _train(_csv(tmp_path, **SFT), training_preset=preset)
    hp = fake_train[0]["hp"]
    assert (hp["epochs"], hp["learning_rate"]) == (epochs, lr)


def test_custom_model_overrides_the_choice_and_is_validated(runs, fake_train, tmp_path):
    _train(_csv(tmp_path, **SFT), custom_model="  org/model  ")
    assert fake_train[0]["model"] == "org/model"
    msg, *_ = _train(_csv(tmp_path, **SFT), custom_model="../evil", run_name="exp2")
    assert "Path traversal" in msg and len(fake_train) == 1


def test_column_mapping_is_applied(runs, fake_train, tmp_path):
    file = _csv(tmp_path, q=SFT["instruction"], a=SFT["output"])
    _train(file, col_inst="q", col_out="a")
    assert set(fake_train[0]["ds"].column_names) >= {"instruction", "output"}


def test_dpo_mode_maps_three_columns(runs, fake_train, tmp_path):
    file = _csv(tmp_path, p=["Question one?", "Question two?"], c=["Good answer", "Great one"],
                r=["Bad answer", "Poor one"])  # fmt: skip
    _train(file, training_mode="DPO (Alignment)", col_inst="p", col_out="c", col_text="r")
    call = fake_train[0]
    assert call["kwargs"]["training_mode"] == "dpo"
    assert set(call["ds"].column_names) >= {"prompt", "chosen", "rejected"}


def test_prepared_dataset_is_used_without_a_file(runs, fake_train):
    ds = Dataset.from_dict({"text": ["alpha beta gamma", "delta epsilon"]})
    msg, *_ = _train(None, augmented_ds=ds)
    assert fake_train[0]["ds"] is ds and "prepared in the Data tab" in msg


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({}, "upload a data file"),
        ({"run_name": "../x"}, "Run name"),
        ({"resume": True, "run_name": "missing"}, "Nothing to resume"),
        ({"report_to": "not-a-tracker"}, "❌"),
    ],
)
def test_train_click_rejects_bad_input(runs, fake_train, overrides, message):
    msg, zip_path, model_dir, records = _train(None, **overrides)
    assert message in msg and (zip_path, model_dir, records) == (None, None, [])
    assert not fake_train


def test_existing_run_needs_resume(runs, fake_train, tmp_path):
    (runs / "exp").mkdir(parents=True)
    msg, *_ = _train(_csv(tmp_path, **SFT))
    assert "already exists" in msg and not fake_train


def test_unreadable_or_empty_data_is_reported(runs, fake_train, tmp_path):
    msg, *_ = _train(_csv(tmp_path, instruction=["", ""], output=["", ""]))
    assert msg == "❌ Dataset is empty after cleaning."
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    msg, *_ = _train(_File(bad), run_name="exp3")
    assert msg and not fake_train


def test_failed_run_folder_is_removed_unless_it_has_checkpoints(runs, monkeypatch, tmp_path):
    def boom(model_name, ds, output_dir, *a, **k):
        if "keep" in output_dir:
            os.makedirs(os.path.join(output_dir, "checkpoint-5"))
        raise RuntimeError("out of memory")

    monkeypatch.setattr(handlers, "train_model", boom)
    msg, zip_path, *_ = _train(_csv(tmp_path, **SFT), run_name="gone")
    assert "Training failed: out of memory" in msg and zip_path is None
    assert not (runs / "gone").exists()
    _train(_csv(tmp_path, **SFT), run_name="keep")
    assert (runs / "keep" / "checkpoint-5").is_dir()  # resumable


def test_stop_sets_only_this_sessions_event():
    session = app_state.session_for(None)
    session.stop_event.clear()
    assert "Stop signal" in handlers.on_stop()
    assert session.stop_event.is_set()
    session.stop_event.clear()


# ── Inference and Hub ──────────────────────────────────────────────────────


def test_generate_validates_and_prefers_the_custom_model(monkeypatch):
    seen = []
    monkeypatch.setattr(handlers, "generate_text", lambda *a: seen.append(a) or "out")
    assert handlers.on_generate("hi", "gpt2", " org/m ", " ", 16.0, 0.7, 0.9) == "out"
    assert seen[0][:3] == ("org/m", "", "hi") and seen[0][3] == 16
    assert "Path traversal" in handlers.on_generate("hi", "gpt2", "", "../lora", 16, 0.7, 0.9)


def test_batch_test_tracks_the_result_file(monkeypatch, tmp_path):
    result = tmp_path / "results.csv"
    result.write_text("prompt,response\n")
    monkeypatch.setattr(handlers, "batch_generate", lambda *a: str(result))
    assert handlers.on_batch_test(_File(tmp_path / "p.csv"), "gpt2", "", "") == str(result)
    assert "Path traversal" in handlers.on_batch_test(None, "gpt2", "../m", "")
    assert "Path traversal" in handlers.on_batch_test(_File("../p.csv"), "gpt2", "", "")
    monkeypatch.setattr(handlers, "batch_generate", lambda *a: "❌ no prompts")
    assert handlers.on_batch_test(None, "gpt2", "", "") == "❌ no prompts"
    app_state.session_for(None).release("batch")


def test_push_delegates(monkeypatch):
    monkeypatch.setattr(handlers, "push_to_hub", lambda *a: f"pushed {a}")
    assert handlers.on_push("m", "o/r", "t") == "pushed ('m', 'o/r', 't')"


def test_remote_chat_shows_errors(monkeypatch):
    assert handlers.on_remote_chat("ftp://h", "", "", "", "hi", 16, 0.5).startswith("❌ Endpoint")
    monkeypatch.setattr(handlers, "remote_chat", lambda *a: "reply")
    assert handlers.on_remote_chat("http://h", "", "", "", "hi", 16.0, 0.5) == "reply"


def test_quantize_export_reads_calibration_data(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(handlers, "on_quantize_click", lambda m, f, d, p: seen.append(d) or "ok")
    assert handlers.on_quantize_export("m", "w4a16", _csv(tmp_path, **SFT), None, None) == "ok"
    assert len(seen[0]) == 3
    assert handlers.on_quantize_export("m", "fp8", _csv(tmp_path, **SFT), None, None) == "ok"
    assert seen[1] is None  # FP8 is data-free
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    msg = handlers.on_quantize_export("m", "w4a16", _File(bad), None, None)
    assert msg.startswith("❌ Cannot read the training data")


# ── Data tab ───────────────────────────────────────────────────────────────


def test_upload_of_ready_csv_previews_it(tmp_path):
    status, inst, out, text, preview, stats, raw_df, ftype = handlers.on_file_upload(
        _csv(tmp_path, **SFT), "SFT (Supervised Fine-Tuning)"
    )
    assert status.startswith("✅ Loaded 3 examples") and ftype == "csv"
    assert not inst["visible"] and len(preview) == 3 and len(raw_df) == 3
    assert "**Total examples:** 3" in stats


def test_upload_with_unknown_columns_asks_for_a_mapping(tmp_path):
    status, inst, out, text, *rest = handlers.on_file_upload(
        _csv(tmp_path, q=SFT["instruction"], a=SFT["output"])
    )
    assert status.startswith("⚠️ Map columns") and inst["visible"] and inst["choices"] == ["q", "a"]
    preview, stats, raw_df = rest[0], rest[1], rest[2]
    assert list(preview.columns) == ["q", "a"] and len(raw_df) == 3 and "Rows in file:** 3" in stats


def test_dpo_upload_needs_all_three_columns(tmp_path):
    status, *_ = handlers.on_file_upload(
        _csv(tmp_path, prompt=["Question?"], chosen=["Yes indeed"]), "DPO (Alignment)"
    )
    assert status.startswith("⚠️ Map columns")


def test_jsonl_upload_goes_through_the_loader(tmp_path):
    path = tmp_path / "d.jsonl"
    pd.DataFrame(SFT).to_json(path, orient="records", lines=True)
    status, *_, raw_df, ftype = handlers.on_file_upload(_File(path))
    assert status.startswith("✅ Loaded 3") and raw_df is None and ftype == "jsonl"


@pytest.mark.parametrize(
    ("file", "status"),
    [
        (None, "No file uploaded."),
        (_File("../escape.csv"), "❌ ❌ Path traversal"),
        (_File("notes.docx"), "⚠️ Unsupported file type."),
    ],
)
def test_upload_rejections(file, status):
    result = handlers.on_file_upload(file)
    assert result[0].startswith(status) and result[6] is None and result[7] is None


def test_upload_parse_error_is_reported(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json\n")
    assert handlers.on_file_upload(_File(path))[0].startswith("❌ Error:")


def test_refresh_preview_applies_the_mapping(tmp_path):
    raw = pd.DataFrame({"q": SFT["instruction"], "a": SFT["output"]})
    preview, stats = handlers.on_refresh_preview(_File(tmp_path / "d.csv"), "SFT", "q", "a", None,
                                                 raw, "csv")  # fmt: skip
    assert len(preview) == 3 and "**Total examples:** 3" in stats
    raw = pd.DataFrame({"t": ["some text here", "more text here"]})
    _, stats = handlers.on_refresh_preview(_File("d.csv"), "SFT", None, None, "t", raw, "csv")
    assert "**Total examples:** 2" in stats
    dpo = pd.DataFrame({"p": ["Question?"], "c": ["Good answer"], "r": ["Bad answer"]})
    _, stats = handlers.on_refresh_preview(_File("d.csv"), "DPO", "p", "c", "r", dpo, "csv")
    assert "**Total examples:** 1" in stats


def test_refresh_preview_without_data_or_with_a_bad_mapping():
    assert handlers.on_refresh_preview(None, "SFT", None, None, None, None, None)[1].startswith("⚠️")
    raw = pd.DataFrame({"q": ["x"]})
    _, stats = handlers.on_refresh_preview(_File("d.csv"), "DPO", "q", "q", "q", raw, "csv")
    assert stats.startswith("❌ Preview refresh failed")


def test_hub_load(monkeypatch):
    ds = Dataset.from_dict(SFT)
    monkeypatch.setattr(handlers, "load_hub_dataset", lambda *a, **k: ds)
    status, preview, stats, loaded = handlers.on_hub_load(" org/data ", "", "train", 10)
    assert "Loaded 3 rows from `org/data`" in status and loaded is not None and len(preview) == 3

    def missing(*a, **k):
        raise ValueError("Dataset not found (token hf_abcdefghijklmnopqrstuvwxyz0123)")

    monkeypatch.setattr(handlers, "load_hub_dataset", missing)
    status, _, _, loaded = handlers.on_hub_load("org/none", "", "train", 10)
    assert (
        status.startswith("❌ Dataset not found") and "hf_abcdef" not in status and loaded is None
    )
    monkeypatch.setattr(handlers, "load_hub_dataset", lambda *a, **k: Dataset.from_dict(
        {"instruction": [""], "output": [""]}))  # fmt: skip
    assert (
        handlers.on_hub_load("org/empty", "", "train", 10)[0] == "❌ No usable rows after cleaning."
    )


# ── Loss chart ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(-5, "—"), (0.5, "—"), (45, "45s"), (125, "2m 05s"), (3720, "1h 02m")],
)
def test_eta_format(seconds, text):
    assert handlers._fmt_eta(seconds) == text


def test_loss_chart():
    assert list(handlers.build_loss_chart([]).columns) == ["Step", "Train Loss", "Eval Loss"]
    chart = handlers.build_loss_chart(
        [{"step": 1, "train_loss": 2.0, "eval_loss": float("nan"), "eta_s": 90.0},
         {"step": 2, "train_loss": 1.5, "eval_loss": 1.4}]
    )  # fmt: skip
    assert list(chart["ETA"]) == ["1m 30s", "—"]
    assert chart["Eval Loss"].isna().tolist() == [True, False]
    old = handlers.build_loss_chart([{"step": 1, "train_loss": 1.0, "eval_loss": 1.0}])
    assert "ETA" not in old.columns
