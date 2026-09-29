"""cli/commands.py — reward / orpo / grpo / kto / evaluate / train guards, trainers faked."""

import re

import pandas as pd
import pytest
from typer.testing import CliRunner

import cli.commands as commands

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

PREFS = {"prompt": ["Question one?", "Question two?"], "chosen": ["Good", "Great"],
         "rejected": ["Bad", "Poor"]}  # fmt: skip


def _run(*args):
    result = runner.invoke(commands.app, list(args))
    return result.exit_code, _ANSI.sub("", result.output + (result.stderr or ""))


@pytest.fixture
def prefs(tmp_path):
    path = tmp_path / "prefs.csv"
    pd.DataFrame(PREFS).to_csv(path, index=False)
    return str(path)


def _recorder(monkeypatch, name, result="✅ done"):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(commands, name, fake)
    return calls


# ── reward / orpo ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("command", "trainer", "flag"),
    [
        ("reward", "train_reward_model_v27", "HAS_REWARD_TRAINER"),
        ("orpo", "train_orpo_v27", "HAS_ORPO"),
    ],
)
def test_preference_commands_run_the_trainer(monkeypatch, prefs, tmp_path, command, trainer, flag):
    calls = _recorder(monkeypatch, trainer)
    code, out = _run(command, "--model", "gpt2", "--data", prefs, "--output", str(tmp_path / "o"),
                     "--epochs", "2")  # fmt: skip
    assert code == 0 and "✅ done" in out and "saved to" in out
    assert calls[0]["model_name"] == "gpt2" and calls[0]["output_dir"] == str(tmp_path / "o")


@pytest.mark.parametrize(
    ("command", "trainer"), [("reward", "train_reward_model_v27"), ("orpo", "train_orpo_v27")]
)
def test_preference_commands_report_failures(monkeypatch, prefs, tmp_path, command, trainer):
    _recorder(monkeypatch, trainer, result="❌ out of memory")
    code, out = _run(command, "--model", "gpt2", "--data", prefs)
    assert code == 1 and "❌ out of memory" in out
    _recorder(monkeypatch, trainer, result=RuntimeError("CUDA error"))
    code, out = _run(command, "--model", "gpt2", "--data", prefs)
    assert code == 1 and "training failed: CUDA error" in out
    bad = tmp_path / "data.xyz"
    bad.write_text("x")
    code, out = _run(command, "--model", "gpt2", "--data", str(bad))
    assert code == 1 and "Unsupported format" in out
    code, out = _run(command, "--model", "gpt2", "--data", str(tmp_path / "missing.csv"))
    assert code == 1 and "Dataset not found" in out


def test_orpo_needs_preference_columns(monkeypatch, tmp_path):
    calls = _recorder(monkeypatch, "train_orpo_v27")
    path = tmp_path / "sft.csv"
    pd.DataFrame({"prompt": ["q"], "chosen": ["a"]}).to_csv(path, index=False)
    code, out = _run("orpo", "--model", "gpt2", "--data", str(path))
    assert code == 1 and not calls


@pytest.mark.parametrize(
    ("command", "flag", "extra"),
    [("reward", "HAS_REWARD_TRAINER", ["--model", "m"]), ("orpo", "HAS_ORPO", ["--model", "m"]),
     ("kto", "HAS_KTO", ["--model", "m"]), ("grpo", "HAS_GRPO", ["--policy-model", "m"])],
)  # fmt: skip
def test_commands_explain_a_missing_trainer(monkeypatch, prefs, command, flag, extra):
    monkeypatch.setattr(commands, flag, False)
    code, out = _run(command, *extra, "--data", prefs)
    assert code == 1 and "not available" in out


# ── grpo / kto ─────────────────────────────────────────────────────────────


def test_grpo_passes_every_option(monkeypatch, tmp_path):
    calls = _recorder(monkeypatch, "train_grpo")
    data = tmp_path / "p.jsonl"
    data.write_text('{"prompt": "2+2?", "reference": "4"}\n')
    code, out = _run("grpo", "--policy-model", "gpt2", "--data", str(data), "--reward", "reference",
                     "--reward", "length", "--loss-type", "dr_grpo", "--num-generations", "2",
                     "--use-vllm", "--resume")  # fmt: skip
    assert code == 0 and "✅ done" in out
    call = calls[0]
    assert call["rewards"] == ["reference", "length"] and call["loss_type"] == "dr_grpo"
    assert call["use_vllm"] and call["resume"] and call["num_generations"] == 2


def test_grpo_and_kto_failures(monkeypatch, tmp_path, prefs):
    _recorder(monkeypatch, "train_grpo", result="❌ no reward")
    assert _run("grpo", "--policy-model", "gpt2", "--data", prefs)[0] == 1
    assert "Dataset not found" in _run("grpo", "--policy-model", "g", "--data", "nope.csv")[1]
    assert "Path traversal" in _run("grpo", "--policy-model", "../g", "--data", prefs)[1]
    _recorder(monkeypatch, "train_kto", result="❌ labels missing")
    code, out = _run("kto", "--model", "gpt2", "--data", prefs)
    assert code == 1 and "labels missing" in out
    assert "Dataset not found" in _run("kto", "--model", "g", "--data", "nope.csv")[1]
    assert "Path traversal" in _run("kto", "--model", "g", "--data", "../x.csv")[1]


def test_kto_runs(monkeypatch, prefs):
    calls = _recorder(monkeypatch, "train_kto")
    code, out = _run("kto", "--model", "gpt2", "--data", prefs, "--beta", "0.2", "--resume")
    assert code == 0 and calls[0]["beta"] == 0.2 and calls[0]["resume"]


# ── evaluate ───────────────────────────────────────────────────────────────


@pytest.fixture
def fake_eval(monkeypatch):
    seen = {"bertscore": 0}
    monkeypatch.setattr(commands, "_load_for_inference", lambda m, lora: ("model", "tok"))

    def predict(model, tok, prompts, max_new, base_model=False, batch_size=8):
        seen.setdefault("base", []).append(base_model)
        return [("base " if base_model else "tuned ") + p for p in prompts]

    def bertscore(preds, refs):
        seen["bertscore"] += 1
        return {"bertscore_f1": 0.9}

    monkeypatch.setattr(commands, "generate_predictions", predict)
    monkeypatch.setattr(
        commands,
        "compute_bleu_rouge",
        lambda preds, refs: {"bleu": 0.5 if preds[0].startswith("tuned") else 0.25},
    )
    monkeypatch.setattr(commands, "compute_bertscore_metric", bertscore)  # fmt: skip
    return seen


def test_evaluate_scores_and_saves_predictions(fake_eval, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"prompt": ["a", "b"], "reference": ["x", "y"], "extra": [1, 2]}).to_csv(
        "test.csv", index=False
    )
    code, out = _run("evaluate", "--model", "gpt2", "--data", "test.csv", "--bertscore")
    assert code == 0 and "bleu" in out and "Evaluation complete — 2 examples" in out
    assert fake_eval["bertscore"] == 1
    (saved,) = tmp_path.glob("eval_results_*.csv")
    assert list(pd.read_csv(saved).columns) == ["prompt", "prediction", "reference"]


def test_evaluate_compares_with_the_base_model(fake_eval, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"prompt": ["a"], "reference": ["x"]}).to_json("t.jsonl", orient="records",
                                                                  lines=True)  # fmt: skip
    code, out = _run("evaluate", "--model", "gpt2", "--data", "t.jsonl", "--lora", "adapter",
                     "--compare-base")  # fmt: skip
    assert code == 0 and fake_eval["base"] == [False, True]
    assert re.search(r"bleu\s+0\.5\s+0\.25\s+0\.25", out)
    (saved,) = tmp_path.glob("eval_results_*.csv")
    assert "base_prediction" in pd.read_csv(saved).columns


def test_evaluate_without_references(fake_eval, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"prompt": ["a"]}).to_csv("p.csv", index=False)
    code, out = _run("evaluate", "--model", "gpt2", "--data", "p.csv")
    assert code == 0 and "No reference column" in out


@pytest.mark.parametrize(
    ("setup", "args", "message"),
    [
        (None, ["--compare-base"], "--compare-base needs a LoRA adapter"),
        (None, ["--lora", "../x"], "Path traversal"),
        (None, [], "Dataset not found"),
        ({"text": ["a"]}, [], "requires 'prompt' column"),
    ],
)
def test_evaluate_rejects_bad_input(fake_eval, tmp_path, monkeypatch, setup, args, message):
    monkeypatch.chdir(tmp_path)
    if setup:
        pd.DataFrame(setup).to_csv("d.csv", index=False)
        pd.DataFrame(setup).to_json("d.jsonl", orient="records", lines=True)
        for name in ("d.csv", "d.jsonl"):
            code, out = _run("evaluate", "--model", "gpt2", "--data", name, *args)
            assert code == 1 and message in out
        return
    code, out = _run("evaluate", "--model", "gpt2", "--data", "d.csv", *args)
    assert code == 1 and message in out


def test_evaluate_reports_model_errors(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"prompt": ["a"]}).to_csv("p.csv", index=False)

    def broken(model, lora):
        raise OSError("model not found")

    monkeypatch.setattr(commands, "_load_for_inference", broken)
    code, out = _run("evaluate", "--model", "nope", "--data", "p.csv")
    assert code == 1 and "Evaluation failed: model not found" in out


# ── train guards ───────────────────────────────────────────────────────────


def test_train_config_errors(tmp_path):
    code, out = _run("train", "--config", str(tmp_path / "missing.yaml"))
    assert code == 1 and "Cannot read run config" in out
    assert "Give --model or --config" in _run("train", "--data", "d.csv")[1]
    assert (
        "--lora-variant must be one of"
        in _run("train", "--model", "gpt2", "--data", "d.csv", "--lora-variant", "PiSSA")[1]
    )
    assert "❌" in _run("train", "--model", "gpt2", "--data", "d.csv", "--report-to", "nope")[1]


def test_train_long_sequence_flags(monkeypatch, tmp_path):
    seen = {}

    def fake_train(**kwargs):
        seen.update(kwargs["hyperparams"])
        return "done", []

    monkeypatch.setattr(commands, "train_model", fake_train)
    data = tmp_path / "d.csv"
    pd.DataFrame({"instruction": ["Say hi", "Say bye"], "output": ["Hello", "Bye"]}).to_csv(
        data, index=False
    )
    code, out = _run("train", "--model", "gpt2", "--data", str(data), "--output", str(tmp_path / "o"),
                     "--activation-offloading", "--padding-free")  # fmt: skip
    assert code == 0, out
    assert seen["activation_offloading"] is True and seen["padding_free"] is True
