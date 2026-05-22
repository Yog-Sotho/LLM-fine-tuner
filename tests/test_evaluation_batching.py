"""
tests/test_evaluation_batching.py
==================================
Unit tests for Bolt optimization: Batched LLM-as-Judge and simplified stripping.

Verifies:
  - llm_judge_evaluate processes prompts in batches.
  - Correct number of generate calls are made (ceil(n / batch_size)).
  - Responses are accurately extracted using the simplified offset.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from inference.evaluation import llm_judge_evaluate

def test_llm_judge_evaluate_batching():
    """Verify that llm_judge_evaluate batches calls to model.generate."""
    # Setup mocks
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Mock tokenizer(eval_texts, ...)
    # It returns a dict with 'input_ids' and 'attention_mask'
    def mock_tokenizer_call(texts, **kwargs):
        batch_size = len(texts)
        # Assume some dummy input_ids of length 10
        input_ids = torch.zeros((batch_size, 10), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    mock_tokenizer.side_effect = mock_tokenizer_call
    mock_tokenizer.batch_decode.return_value = ["Excellent response."] * 8 # return enough for first batch
    mock_tokenizer.eos_token_id = 50256

    # Mock model.generate(...)
    # It should return a tensor of shape (batch_size, input_len + new_tokens)
    def mock_model_generate(**kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        input_len = input_ids.shape[1]
        # Return input_ids + some new tokens
        return torch.cat([input_ids, torch.ones((batch_size, 5), dtype=torch.long)], dim=1)

    mock_model.generate.side_effect = mock_model_generate

    prompts = [f"Prompt {i}" for i in range(10)]
    responses = [f"Response {i}" for i in range(10)]
    criteria = "helpfulness"

    with patch("inference.evaluation._load_for_inference", return_value=(mock_model, mock_tokenizer)):
        results = llm_judge_evaluate(
            prompts=prompts,
            responses=responses,
            criteria=criteria,
            judge_model_name="mock-judge"
        )

    # Verify results
    assert len(results) == 10
    for i, res in enumerate(results):
        assert res["prompt"] == prompts[i]
        assert res["response"] == responses[i]
        assert res["judgment"] == "Excellent response."

    # Verify batching: batch_size is 8. For 10 prompts, we expect 2 batches.
    assert mock_model.generate.call_count == 2
    # First batch should have 8
    args, kwargs = mock_model.generate.call_args_list[0]
    assert kwargs["input_ids"].shape[0] == 8
    # Second batch should have 2
    args, kwargs = mock_model.generate.call_args_list[1]
    assert kwargs["input_ids"].shape[0] == 2
    # Verify batch_decode
    assert mock_tokenizer.batch_decode.call_count == 2

def test_prompt_stripping_offset():
    """Verify that stripping logic uses the correct offset (input_ids.shape[1])."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Input length 15
    input_len = 15
    def mock_tokenizer_call(texts, **kwargs):
        batch_size = len(texts)
        input_ids = torch.zeros((batch_size, input_len), dtype=torch.long)
        return {"input_ids": input_ids}

    mock_tokenizer.side_effect = mock_tokenizer_call
    mock_tokenizer.eos_token_id = 50256

    # Return 5 new tokens (total length 20)
    def mock_model_generate(**kwargs):
        input_ids = kwargs["input_ids"]
        return torch.cat([input_ids, torch.arange(5).repeat(input_ids.shape[0], 1)], dim=1)

    mock_model.generate.side_effect = mock_model_generate

    # We want to check what slice is passed to tokenizer.batch_decode
    # The code does: tokenizer.batch_decode(outputs[:, input_len:], ...)

    mock_tokenizer.batch_decode.return_value = ["Decoded Response"]

    with patch("inference.evaluation._load_for_inference", return_value=(mock_model, mock_tokenizer)):
        llm_judge_evaluate(
            prompts=["P1"],
            responses=["R1"],
            criteria="C1",
            judge_model_name="mock-judge"
        )

    # Check mock_tokenizer.batch_decode call
    args, kwargs = mock_tokenizer.batch_decode.call_args
    passed_tensor = args[0]
    # passed_tensor is (batch_size, new_tokens) = (1, 5)
    assert passed_tensor.shape == (1, 5)
    # The values should be 0, 1, 2, 3, 4 (the new tokens we added)
    assert torch.equal(passed_tensor, torch.arange(5).unsqueeze(0))

def test_on_evaluate_click_batching():
    """Verify that on_evaluate_click batches calls to model.generate with size 8."""
    from inference.evaluation import on_evaluate_click
    import pandas as pd
    import tempfile
    import os

    # Setup mocks
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    def mock_tokenizer_call(texts, **kwargs):
        batch_size = len(texts)
        input_ids = torch.zeros((batch_size, 10), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    mock_tokenizer.side_effect = mock_tokenizer_call
    # on_evaluate_click uses batch_decode
    def mock_batch_decode(outputs, **kwargs):
        return ["Response"] * outputs.shape[0]
    mock_tokenizer.batch_decode.side_effect = mock_batch_decode
    mock_tokenizer.eos_token_id = 50256

    def mock_model_generate(**kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        return torch.cat([input_ids, torch.ones((batch_size, 5), dtype=torch.long)], dim=1)

    mock_model.generate.side_effect = mock_model_generate

    # Create a dummy CSV file with 10 prompts
    df = pd.DataFrame({"prompt": [f"P{i}" for i in range(10)], "reference": [f"R{i}" for i in range(10)]})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        with patch("inference.evaluation._load_for_inference", return_value=(mock_model, mock_tokenizer)):
            # Mock gr.Progress
            mock_progress = MagicMock()

            mock_file = MagicMock()
            mock_file.name = tmp_path

            result = on_evaluate_click(
                eval_model_name="mock-model",
                eval_custom_model="",
                eval_lora_path="",
                eval_file=mock_file,
                eval_run_bertscore=False,
                eval_use_judge=False,
                judge_model_name="",
                judge_criteria="",
                progress=mock_progress
            )

            # If it failed, result[0] will contain the error message
            if isinstance(result[0], str) and "❌" in result[0]:
                print(f"on_evaluate_click failed: {result[0]}")

        # Verify batching: batch_size is now 8. For 10 prompts, we expect 2 batches.
        assert mock_model.generate.call_count == 2
        # First batch should have 8
        args, kwargs = mock_model.generate.call_args_list[0]
        assert kwargs["input_ids"].shape[0] == 8
        # Second batch should have 2
        args, kwargs = mock_model.generate.call_args_list[1]
        assert kwargs["input_ids"].shape[0] == 2

        # Verify batch_decode was called
        assert mock_tokenizer.batch_decode.call_count == 2
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_batch_generate_optimization():
    """Verify that batch_generate uses batch_decode and correct slicing."""
    from inference.generate import batch_generate
    import pandas as pd
    import tempfile
    import os

    # Setup mocks
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    def mock_tokenizer_call(texts, **kwargs):
        batch_size = len(texts)
        input_ids = torch.zeros((batch_size, 10), dtype=torch.long)
        return {"input_ids": input_ids}

    mock_tokenizer.side_effect = mock_tokenizer_call
    def mock_batch_decode(outputs, **kwargs):
        return ["Response"] * outputs.shape[0]
    mock_tokenizer.batch_decode.side_effect = mock_batch_decode
    mock_tokenizer.eos_token_id = 50256
    mock_tokenizer.pad_token = " "

    def mock_model_generate(**kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        return torch.cat([input_ids, torch.ones((batch_size, 5), dtype=torch.long)], dim=1)

    mock_model.generate.side_effect = mock_model_generate

    # Create a dummy CSV file with 5 prompts
    df = pd.DataFrame({"prompt": [f"P{i}" for i in range(5)]})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        with patch("inference.generate._load_for_inference", return_value=(mock_model, mock_tokenizer)):
            mock_file = MagicMock()
            mock_file.name = tmp_path

            result_path = batch_generate(
                model_name="mock-model",
                lora_path=None,
                prompts_file=mock_file
            )

            assert os.path.exists(result_path)
            res_df = pd.read_csv(result_path)
            assert len(res_df) == 5
            assert all(res_df["response"] == "Response")

        # Verify batch_decode was called
        assert mock_tokenizer.batch_decode.call_count == 1
        # Check slicing: outputs[:, input_len:]
        args, kwargs = mock_tokenizer.batch_decode.call_args
        passed_tensor = args[0]
        assert passed_tensor.shape == (5, 5) # 5 prompts in batch, 5 new tokens

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
