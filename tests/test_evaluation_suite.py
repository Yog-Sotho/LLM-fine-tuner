"""Unit tests (no downloads) for judge scores, metric tables and the benchmark runner."""

import sys
import types

import numpy as np
import pytest

import inference.benchmarks as benchmarks
from inference.evaluation import _format_metrics, mean_judge_score, parse_judge_score

# ── Judge scores ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reply", "score"),
    [
        ("Score: 7", 7),
        (" 8. Clear and correct.", 8),
        ("score=10 because it is perfect", 10),
        ("9/10 — concise", 9),
        ("The answer is fine. Score : 3", 3),
        ("Score: 11", None),
        ("Score: 0", None),
        ("It covers 2 of the 3 points.", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_judge_score(reply, score):
    assert parse_judge_score(reply) == score


def test_mean_judge_score_ignores_replies_without_score():
    rows = [{"score": 6}, {"score": None}, {"score": 9}]
    assert mean_judge_score(rows) == (7.5, 1)
    assert mean_judge_score([{"score": None}]) == (None, 1)


# ── Metric tables ──────────────────────────────────────────────────────────


def test_format_metrics_single_model():
    assert _format_metrics({"model": {"BLEU-1": 0.5}}) == "**BLEU-1:** 0.5"
    assert "skipped" in _format_metrics({"model": {}})


def test_format_metrics_compares_with_base():
    text = _format_metrics(
        {
            "fine-tuned": {"ROUGE-L": 0.6, "Judge score (1-10)": "n/a"},
            "base": {"ROUGE-L": 0.4, "Judge score (1-10)": 5.0},
        }
    )
    assert "| Metric | Fine-tuned | Base | Δ |" in text
    assert "| ROUGE-L | 0.6 | 0.4 | +0.2000 |" in text
    assert "| Judge score (1-10) | n/a | 5.0 |  |" in text


# ── Benchmarks (lm-eval replaced by a fake module) ─────────────────────────


@pytest.fixture
def fake_lm_eval(monkeypatch):
    """Record HFLM / simple_evaluate calls; results mimic lm-eval 0.4 output."""
    calls: dict = {"models": [], "evaluate": []}

    class FakeHFLM:
        def __init__(self, **kwargs):
            calls["models"].append(kwargs)
            self.peft = kwargs["peft"]

    def simple_evaluate(model, tasks, limit, **kwargs):
        calls["evaluate"].append({"tasks": tasks, "limit": limit, **kwargs})
        acc = 0.75 if model.peft else 0.5
        return {
            "results": {
                task: {
                    "alias": task,
                    "acc,none": np.float32(acc),
                    "acc_stderr,none": 0.1,
                    "sample_len": limit,
                }
                for task in tasks
            }
        }

    lm_eval = types.ModuleType("lm_eval")
    lm_eval.simple_evaluate = simple_evaluate
    models = types.ModuleType("lm_eval.models")
    huggingface = types.ModuleType("lm_eval.models.huggingface")
    huggingface.HFLM = FakeHFLM
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", huggingface)
    monkeypatch.setattr(benchmarks, "HAS_LM_EVAL", True)
    return calls


@pytest.fixture
def adapter_dir(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    return str(tmp_path)


def test_benchmarks_single_model(fake_lm_eval):
    table = benchmarks.run_benchmarks("gpt2", None, ["arc_easy", "piqa"], 20)
    assert table.to_dict("records") == [
        {"task": "arc_easy", "metric": "acc", "score": 0.5},
        {"task": "piqa", "metric": "acc", "score": 0.5},
    ]
    (model_kwargs,) = fake_lm_eval["models"]
    assert model_kwargs["pretrained"] == "gpt2" and model_kwargs["peft"] is None
    assert model_kwargs["trust_remote_code"] is False  # ALLOW_REMOTE_CODE default
    # Exactly these arguments: use_cache would open a pickle-based sqlitedict cache
    # (CVE-2024-35515, waived in CI's pip-audit only because it is never used).
    assert fake_lm_eval["evaluate"] == [{"tasks": ["arc_easy", "piqa"], "limit": 20}]


def test_benchmarks_compare_base(fake_lm_eval, adapter_dir):
    table = benchmarks.run_benchmarks("gpt2", adapter_dir, ["boolq"], 5, compare_base=True)
    assert table.to_dict("records") == [
        {"task": "boolq", "metric": "acc", "fine-tuned": 0.75, "base": 0.5, "Δ": 0.25}
    ]
    assert [m["peft"] for m in fake_lm_eval["models"]] == [adapter_dir, None]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("", None, ["arc_easy"], 10), "Choose a model"),
        (("../model", None, ["arc_easy"], 10), "Path traversal"),
        (("gpt2", "a/../b", ["arc_easy"], 10), "Path traversal"),
        (("gpt2", None, [], 10), "Choose benchmarks"),
        (("gpt2", None, ["arc_easy", "my_task"], 10), "Choose benchmarks"),
        (("gpt2", None, ["arc_easy"], 0), "between 1 and"),
        (("gpt2", None, ["arc_easy"], 10_001), "between 1 and"),
        (("gpt2", "/no/such/adapter", ["arc_easy"], 10), "not found"),
        (("gpt2", None, ["arc_easy"], 10, True), "needs a LoRA adapter"),
    ],
)
def test_benchmark_input_validation(fake_lm_eval, args, message):
    with pytest.raises(ValueError, match=message):
        benchmarks.run_benchmarks(*args)
    assert fake_lm_eval["models"] == []


def test_benchmark_rejects_pickle_adapter(fake_lm_eval, tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="safetensors"):
        benchmarks.run_benchmarks("gpt2", str(tmp_path), ["arc_easy"], 10)


def test_benchmark_without_lm_eval(monkeypatch):
    monkeypatch.setattr(benchmarks, "HAS_LM_EVAL", False)
    with pytest.raises(ImportError, match="pip install"):
        benchmarks.run_benchmarks("gpt2", None, ["arc_easy"], 10)


def test_on_benchmark_click_reports_errors_and_results(fake_lm_eval):
    status, table = benchmarks.on_benchmark_click(
        "gpt2", "", "", [], 10, False, progress=lambda *a, **k: None
    )
    assert status.startswith("❌") and table.empty
    status, table = benchmarks.on_benchmark_click(
        "gpt2", " distilgpt2 ", "", ["hellaswag"], 10, False, progress=lambda *a, **k: None
    )
    assert status.startswith("✅") and len(table) == 1
    assert fake_lm_eval["models"][0]["pretrained"] == "distilgpt2"
