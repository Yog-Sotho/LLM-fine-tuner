
import pytest
from unittest.mock import MagicMock
import gradio as gr
from training.reward import train_reward_model_v27
from training.ppo import run_ppo_v27
from training.orpo import train_orpo_v27
from inference.vllm_runner import on_merge_adapter_click

def test_reward_path_traversal():
    result = train_reward_model_v27(model_name="../unsafe", reward_file=None, output_dir="./ok")
    assert "❌ Path traversal attempt detected." in result

    result = train_reward_model_v27(model_name="ok", reward_file=None, output_dir="../unsafe")
    assert "❌ Path traversal attempt detected." in result

    result = train_reward_model_v27(model_name="ok\\unsafe", reward_file=None, output_dir="./ok")
    assert "❌ Path traversal attempt detected." in result

def test_ppo_path_traversal():
    result = run_ppo_v27(policy_model_name="../unsafe", reward_model_path="./ok", ppo_file=None, output_dir="./ok")
    assert "❌ Path traversal attempt detected." in result

    result = run_ppo_v27(policy_model_name="ok", reward_model_path="../unsafe", ppo_file=None, output_dir="./ok")
    assert "❌ Path traversal attempt detected." in result

    result = run_ppo_v27(policy_model_name="ok", reward_model_path="./ok", ppo_file=None, output_dir="../unsafe")
    assert "❌ Path traversal attempt detected." in result

def test_orpo_path_traversal():
    result = train_orpo_v27(model_name="../unsafe", orpo_file=None, output_dir="./ok")
    assert "❌ Path traversal attempt detected." in result

    result = train_orpo_v27(model_name="ok", orpo_file=None, output_dir="../unsafe")
    assert "❌ Path traversal attempt detected." in result

def test_vllm_merge_path_traversal():
    result, update = on_merge_adapter_click(base_model_name="../unsafe", adapter_path="./ok", model_path_state="./ok")
    assert "❌ Path traversal attempt detected." in result

    result, update = on_merge_adapter_click(base_model_name="ok", adapter_path="../unsafe", model_path_state="./ok")
    assert "❌ Path traversal attempt detected." in result

    result, update = on_merge_adapter_click(base_model_name="ok", adapter_path="ok\\unsafe", model_path_state="./ok")
    assert "❌ Path traversal attempt detected." in result
