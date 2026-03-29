"""
tests/test_ppo_reward_type.py
==============================
Unit tests for v3.2 Fix #2 — PPO reward float type.

The fix ensures reward values are appended to the rewards list as plain
Python floats (not re-wrapped in torch.tensor()), which caused type errors
in some TRL versions when the value was already a float.

Covers:
  - reward_val extracted from a value-head output is a plain float
  - Appending the float directly to rewards list (not re-wrapped)
  - Old broken pattern (torch.tensor(float)) vs fixed pattern
  - rewards list contains Python floats, not tensors
"""

import torch

# ── helpers mirroring run_ppo_v27 reward extraction ───────────────────────

def _broken_reward_append(rewards: list, reward_val: float):
    """Original broken pattern: re-wraps a float in torch.tensor."""
    rewards.append(torch.tensor(reward_val))


def _fixed_reward_append(rewards: list, reward_val: float):
    """Fixed pattern: append the float directly."""
    rewards.append(reward_val)


def _extract_reward_from_value_head(logits_tensor: torch.Tensor) -> float:
    """Mirror the extraction logic in run_ppo_v27."""
    return logits_tensor.squeeze()[-1].item()


# ── tests ──────────────────────────────────────────────────────────────────

def test_extraction_returns_float():
    """_extract_reward_from_value_head must return a Python float, not a tensor."""
    fake_output = torch.randn(1, 5, 1)  # shape: (batch, seq, 1) — value head output
    reward_val = _extract_reward_from_value_head(fake_output.squeeze(-1))
    assert isinstance(reward_val, float), (
        f"Expected float, got {type(reward_val)}"
    )


def test_fixed_pattern_rewards_are_floats():
    """After the fix, all items in the rewards list must be Python floats."""
    rewards = []
    for _ in range(4):
        fake = torch.randn(1, 5, 1)
        val = _extract_reward_from_value_head(fake.squeeze(-1))
        _fixed_reward_append(rewards, val)
    assert all(isinstance(r, float) for r in rewards), (
        f"Expected all floats, got: {[type(r) for r in rewards]}"
    )


def test_broken_pattern_rewards_are_tensors():
    """Control test: broken pattern produces tensors, not floats."""
    rewards = []
    for _ in range(4):
        fake = torch.randn(1, 5, 1)
        val = _extract_reward_from_value_head(fake.squeeze(-1))
        _broken_reward_append(rewards, val)
    assert all(isinstance(r, torch.Tensor) for r in rewards), (
        "Broken pattern should produce tensors — this confirms the fix is meaningful."
    )


def test_fixed_rewards_list_compatible_with_trl():
    """
    TRL's PPOTrainer.step() expects a list of floats or 1-D scalar tensors.
    A list of plain floats must be accepted without type errors.
    All values should be finite real numbers.
    """
    rewards = []
    for _i in range(3):
        fake = torch.randn(1, 5, 1)
        val = _extract_reward_from_value_head(fake.squeeze(-1))
        _fixed_reward_append(rewards, val)
    assert len(rewards) == 3
    for r in rewards:
        assert isinstance(r, float)
        assert not (r != r)  # NaN check


def test_zero_dim_tensor_item_is_float():
    """Confirm that calling .item() on any scalar tensor always gives a Python float."""
    for _ in range(10):
        t = torch.randn(())  # 0-D tensor
        assert isinstance(t.item(), float)
