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
    mock_tokenizer.decode.return_value = "Excellent response."
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

    # We want to check what slice is passed to tokenizer.decode
    # The code does: tokenizer.decode(outputs[idx, input_len:], ...)

    with patch("inference.evaluation._load_for_inference", return_value=(mock_model, mock_tokenizer)):
        llm_judge_evaluate(
            prompts=["P1"],
            responses=["R1"],
            criteria="C1",
            judge_model_name="mock-judge"
        )

    # Check mock_tokenizer.decode call
    args, kwargs = mock_tokenizer.decode.call_args
    passed_tensor = args[0]
    assert passed_tensor.shape[0] == 5
    # The values should be 0, 1, 2, 3, 4 (the new tokens we added)
    assert torch.equal(passed_tensor, torch.arange(5))
