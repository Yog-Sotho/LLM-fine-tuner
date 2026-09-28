"""GRPO built-in rewards and option validation — no model download."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

import training.grpo as grpo


def test_json_reward_accepts_plain_and_fenced_json():
    completions = ['{"a": 1}', "```json\n[1, 2]\n```", "  42 ", "not json", "{'single': 1}"]
    assert grpo.json_reward(completions) == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_json_reward_reads_chat_completions():
    assert grpo.json_reward([[{"role": "assistant", "content": '{"ok": true}'}]]) == [1.0]


def test_regex_reward_matches_the_whole_output():
    reward = grpo.make_regex_reward(r"Answer: \d+")
    assert reward(["Answer: 42", " Answer: 7 ", "Answer: 42 because", "answer: 1"]) == [
        1.0, 1.0, 0.0, 0.0
    ]  # fmt: skip
    assert reward.__name__ == "regex_reward"  # TRL logs rewards by function name


def test_invalid_regex_is_a_clear_error():
    with pytest.raises(ValueError, match="Invalid regular expression"):
        grpo.make_regex_reward("(unclosed")


def test_think_format_reward_on_plain_strings():
    # TRL's function reads completion[0]["content"]; plain strings are wrapped first.
    assert grpo.think_format_reward(["<think>because</think>4", "4"]) == [1.0, 0.0]


def test_math_answer_reward_uses_math_verify():
    rewards = grpo.math_answer_reward(
        ["The answer is $\\boxed{\\frac{1}{2}}$", "$\\boxed{0.5}$", "$\\boxed{3}$"],
        reference=["1/2", "$\\frac{1}{2}$", "2"],
    )
    assert rewards == [1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("rewards", "has_reference", "regex", "message"),
    [
        (["nope"], True, "", "Unknown reward"),
        (["reference"], False, "", "needs a 'reference' column"),
        (["math"], False, "", "needs a 'reference' column"),
        (["regex"], True, " ", "needs a regular expression"),
    ],
)
def test_build_reward_funcs_validation(rewards, has_reference, regex, message):
    with pytest.raises(ValueError, match=message):
        grpo.build_reward_funcs(rewards, has_reference, regex)


def test_build_reward_funcs_order_and_types():
    funcs = grpo.build_reward_funcs(["json", "regex", "reference"], True, "x+")
    assert [f.__name__ for f in funcs] == ["json_reward", "regex_reward", "reference_match_reward"]


# ── Options passed to TRL (model loading and GRPOConfig replaced) ─────────


class _Stop(Exception):
    pass


@pytest.fixture
def grpo_config_spy(monkeypatch, tmp_path):
    import trl

    captured = {}

    def fake_config(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(trl, "GRPOConfig", fake_config)
    monkeypatch.setattr(grpo.AutoTokenizer, "from_pretrained", lambda *a, **k: MagicMock())
    monkeypatch.setattr(grpo.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: MagicMock())
    data = tmp_path / "p.csv"
    pd.DataFrame({"prompt": ["2+2="], "reference": ["4"]}).to_csv(data, index=False)
    upload = MagicMock()
    upload.name = str(data)
    return captured, upload


def test_loss_type_and_vllm_colocate_reach_grpo_config(grpo_config_spy, monkeypatch, tmp_path):
    captured, upload = grpo_config_spy
    monkeypatch.setattr(grpo, "HAS_VLLM", True)
    monkeypatch.setattr(grpo.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(grpo.torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    result = grpo.train_grpo(
        "some/model", "", upload, str(tmp_path / "out"), loss_type="dr_grpo",
        use_vllm=True, progress=None,
    )  # fmt: skip
    assert result.startswith("❌ GRPO training failed")  # stopped by the spy
    assert captured["loss_type"] == "dr_grpo"
    # vllm_mode is explicit: TRL 0.29 defaults to "server", 1.x to "colocate".
    assert (captured["use_vllm"], captured["vllm_mode"]) == (True, "colocate")


def test_vllm_is_off_by_default(grpo_config_spy, tmp_path):
    captured, upload = grpo_config_spy
    grpo.train_grpo("some/model", "", upload, str(tmp_path / "out"), progress=None)
    assert "use_vllm" not in captured and captured["loss_type"] == "dapo"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"loss_type": "ppo"}, "Loss type must be one of"),
        ({"lora_variant": "PiSSA"}, "Unknown LoRA variant"),
        ({"use_vllm": True}, "vLLM generation needs a CUDA GPU"),
        ({"rewards": ["regex"]}, "needs a regular expression"),
    ],
)
def test_grpo_option_errors(grpo_config_spy, monkeypatch, tmp_path, kwargs, message):
    _, upload = grpo_config_spy
    monkeypatch.setattr(grpo.torch.cuda, "is_available", lambda: False)
    result = grpo.train_grpo("some/model", "", upload, str(tmp_path), progress=None, **kwargs)
    assert result.startswith("❌") and message in result
