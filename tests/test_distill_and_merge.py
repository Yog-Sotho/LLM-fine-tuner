"""Knowledge distillation (training/distill.py) — data handling, input checks and CLI.

Real distillation runs live in tests/test_smoke_training.py.
"""

import json
import re

import pytest
from datasets import Dataset
from typer.testing import CliRunner

import cli.commands as commands
import training.distill as distill

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class _File:
    def __init__(self, path):
        self.name = str(path)


def _messages(ds):
    return [list(m) for m in ds["messages"]]


def test_chats_are_kept_and_need_a_final_answer():
    from data.preprocessing import chat_dataset

    ds = chat_dataset([
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
        {"messages": [{"role": "user", "content": "No answer"}]},
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": " "}]},
    ])  # fmt: skip
    out = distill.to_distill_dataset(ds)
    assert len(out) == 1 and out[0]["messages"][-1] == {"role": "assistant", "content": "Hello"}


@pytest.mark.parametrize(
    "columns",
    [
        {"instruction": ["What is 2+2?"], "output": ["4"]},
        {"prompt": ["What is 2+2?"], "completion": ["4"]},
    ],
)
def test_pairs_become_two_turn_chats(columns):
    out = distill.to_distill_dataset(Dataset.from_dict(columns))
    assert _messages(out) == [[{"role": "user", "content": "What is 2+2?"},
                               {"role": "assistant", "content": "4"}]]  # fmt: skip


def test_unknown_columns_are_rejected():
    with pytest.raises(ValueError, match="Distillation needs chats"):
        distill.to_distill_dataset(Dataset.from_dict({"text": ["hello"]}))


def test_files_of_each_format_load(tmp_path):
    chats = tmp_path / "chats.jsonl"
    chats.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hey"},
                ]
            }
        )
        + "\n"
    )
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("prompt,completion\nWhat is 2+2?,4\n")  # fmt: skip
    for path in (chats, pairs):
        assert len(distill.to_distill_dataset(distill._load_distill_file(_File(path)))) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"student_model_name": "../s"}, "Path traversal"),
        ({"teacher_model_name": ""}, "Give the student"),
        ({"distill_file": None}, "Please upload a dataset"),
        ({"lmbda": 1.5}, "between 0 and 1"),
    ],
)
def test_input_checks(kwargs, message):
    args = {"student_model_name": "org/small", "teacher_model_name": "org/big",
            "distill_file": _File("d.jsonl"), "output_dir": "out", "progress": None}  # fmt: skip
    assert message in distill.train_distill(**{**args, **kwargs})


def test_missing_trl_gkd_is_reported(monkeypatch):
    monkeypatch.setattr(distill, "HAS_GKD", False)
    status = distill.train_distill("s", "t", _File("d.jsonl"), "out", progress=None)
    assert status.startswith("❌ GKDTrainer not available")


def test_cli_distill_passes_options(monkeypatch, tmp_path):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return "✅ done"

    monkeypatch.setattr(commands, "train_distill", fake)
    data = tmp_path / "d.jsonl"
    data.write_text("{}\n")
    result = CliRunner().invoke(commands.app, [
        "distill", "--student", "org/small", "--teacher", "org/big", "--data", str(data),
        "--lmbda", "1.0", "--beta", "0.1", "--max-new-tokens", "32", "--resume",
    ])  # fmt: skip
    assert result.exit_code == 0 and "✅ done" in result.output
    assert (seen["student_model_name"], seen["teacher_model_name"]) == ("org/small", "org/big")
    assert seen["lmbda"] == 1.0 and seen["beta"] == 0.1 and seen["max_new_tokens"] == 32
    assert seen["resume"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--student", "../s", "--teacher", "t", "--data", "d.jsonl"], "Path traversal"),
        (["--student", "s", "--teacher", "t", "--data", "missing.jsonl"], "Dataset not found"),
    ],
)
def test_cli_distill_rejects_bad_input(args, message):
    result = CliRunner().invoke(commands.app, ["distill", *args])
    assert result.exit_code == 1 and message in _ANSI.sub("", result.output + result.stderr)


def test_cli_distill_reports_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(commands, "train_distill", lambda **k: "❌ vocab mismatch")
    data = tmp_path / "d.jsonl"
    data.write_text("{}\n")
    result = CliRunner().invoke(
        commands.app, ["distill", "--student", "s", "--teacher", "t", "--data", str(data)]
    )
    assert result.exit_code == 1 and "vocab mismatch" in result.stderr
    monkeypatch.setattr(commands, "HAS_GKD", False)
    result = CliRunner().invoke(
        commands.app, ["distill", "--student", "s", "--teacher", "t", "--data", str(data)]
    )
    assert result.exit_code == 1 and "GKD not available" in result.stderr


# ── Merging LoRA adapters (export/merge.py) ────────────────────────────────

import export.merge as merge  # noqa: E402


def _adapter(tmp_path, name, **config):
    folder = tmp_path / name
    folder.mkdir()
    base = {"peft_type": "LORA", "r": 4, "base_model_name_or_path": "org/base"}
    (folder / "adapter_config.json").write_text(json.dumps({**base, **config}))
    (folder / "adapter_model.safetensors").write_bytes(b"")
    return str(folder)


@pytest.mark.parametrize(
    ("setup", "kwargs", "message"),
    [
        (1, {}, "at least two adapter folders"),
        (2, {"method": "average"}, "Method must be one of"),
        (2, {"weights": [1.0]}, "one weight per adapter (2)"),
        (2, {"density": 0.0}, "Density must be above 0"),
        (2, {"output_dir": ""}, "Give an output folder"),
        (2, {"output_dir": "../out"}, "Path traversal"),
    ],
)
def test_merge_input_checks(tmp_path, setup, kwargs, message):
    adapters = [_adapter(tmp_path, f"a{i}") for i in range(setup)]
    args = {"adapter_dirs": adapters, "output_dir": str(tmp_path / "out"), **kwargs}
    assert message in merge.merge_lora_adapters(**args)


def test_merge_checks_each_adapter(tmp_path):
    good = _adapter(tmp_path, "good")
    ia3 = _adapter(tmp_path, "ia3", peft_type="IA3")
    no_base = _adapter(tmp_path, "nobase", base_model_name_or_path="")
    pickle = tmp_path / "pickle"
    pickle.mkdir()
    (pickle / "adapter_config.json").write_text("{}")
    out = str(tmp_path / "out")
    assert "Only LoRA adapters can be merged" in merge.merge_lora_adapters([good, ia3], out)
    assert "same base model" in merge.merge_lora_adapters([good, no_base], out)
    assert "Pickle-based weights" in merge.merge_lora_adapters([good, str(pickle)], out)
    assert "Adapter folder not found" in merge.merge_lora_adapters([good, "nope"], out)
    # density is only checked for the methods that use it
    status = merge.merge_lora_adapters([good, _adapter(tmp_path, "b", r=8)], out, method="cat",
                                       density=0.0)  # fmt: skip
    assert "Density" not in status


def test_merge_button_parses_weights(monkeypatch):
    seen = {}
    monkeypatch.setattr(merge, "merge_lora_adapters",
                        lambda a, o, w, m, d: seen.update(a=a, w=w, m=m, d=d) or "✅")  # fmt: skip
    status = merge.on_merge_adapters_click("runs/a\n\n runs/b \n", "1, 0.5", "ties", 0.4, "o",
                                           None)  # fmt: skip
    assert status == "✅" and seen == {"a": ["runs/a", " runs/b "], "w": [1.0, 0.5], "m": "ties",
                                       "d": 0.4}  # fmt: skip
    merge.on_merge_adapters_click("a\nb", "", "cat", 0.5, "o", None)
    assert seen["w"] is None
    assert merge.on_merge_adapters_click("a\nb", "1, x", "ties", 0.5, "o", None).startswith(
        "❌ Weights must be numbers"
    )


def test_cli_merge_adapters(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        commands,
        "merge_lora_adapters",
        lambda a, o, w, m, d: seen.update(a=a, o=o, w=w, m=m, d=d) or "✅ merged",
    )
    result = CliRunner().invoke(commands.app, [
        "merge-adapters", "--adapter", "a", "--adapter", "b", "--weight", "1", "--weight", "2",
        "--method", "dare_ties", "--density", "0.3", "--output", "out",
    ])  # fmt: skip
    assert result.exit_code == 0 and "✅ merged" in result.output
    assert seen == {"a": ["a", "b"], "o": "out", "w": [1.0, 2.0], "m": "dare_ties", "d": 0.3}
    monkeypatch.setattr(commands, "merge_lora_adapters", lambda *a: "❌ nope")
    result = CliRunner().invoke(commands.app, ["merge-adapters", "--adapter", "a", "--output", "o"])
    assert result.exit_code == 1 and "❌ nope" in result.stderr
