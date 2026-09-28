from inference.vllm_runner import on_merge_adapter_click
from training.grpo import train_grpo
from training.kto import train_kto
from training.orpo import train_orpo_v27
from training.reward import train_reward_model_v27


def test_reward_path_traversal():
    result = train_reward_model_v27(model_name="../unsafe", reward_file=None, output_dir="./ok")
    assert "❌ Path traversal attempt detected." in result

    result = train_reward_model_v27(model_name="ok", reward_file=None, output_dir="../unsafe")
    assert "❌ Path traversal attempt detected." in result

    result = train_reward_model_v27(model_name="ok\\unsafe", reward_file=None, output_dir="./ok")
    assert "❌ Path traversal attempt detected." in result


def test_grpo_path_traversal():
    for policy, reward, out in [
        ("../unsafe", "./ok", "./ok"),
        ("ok", "../unsafe", "./ok"),
        ("ok", "./ok", "../unsafe"),
    ]:
        result = train_grpo(
            policy_model_name=policy, reward_model_path=reward, prompts_file=None, output_dir=out
        )
        assert "❌ Path traversal attempt detected." in result


def test_kto_path_traversal():
    for model, out in [("../unsafe", "./ok"), ("ok", "../unsafe"), ("ok\\unsafe", "./ok")]:
        result = train_kto(model_name=model, kto_file=None, output_dir=out)
        assert "❌ Path traversal attempt detected." in result


def test_orpo_path_traversal():
    result = train_orpo_v27(model_name="../unsafe", orpo_file=None, output_dir="./ok")
    assert "❌ Path traversal attempt detected." in result

    result = train_orpo_v27(model_name="ok", orpo_file=None, output_dir="../unsafe")
    assert "❌ Path traversal attempt detected." in result


def test_vllm_merge_path_traversal():
    result, update = on_merge_adapter_click(
        base_model_name="../unsafe", adapter_path="./ok", model_path_state="./ok"
    )
    assert "❌ Path traversal attempt detected." in result

    result, update = on_merge_adapter_click(
        base_model_name="ok", adapter_path="../unsafe", model_path_state="./ok"
    )
    assert "❌ Path traversal attempt detected." in result

    result, update = on_merge_adapter_click(
        base_model_name="ok", adapter_path="ok\\unsafe", model_path_state="./ok"
    )
    assert "❌ Path traversal attempt detected." in result
