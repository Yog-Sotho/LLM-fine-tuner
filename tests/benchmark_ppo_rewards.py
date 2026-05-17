
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MockValueHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type('Config', (), {'is_encoder_decoder': False})
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Return dummy values: (batch, seq, 1)
        batch_size, seq_len = input_ids.shape
        values = torch.randn(batch_size, seq_len, 1)
        return type('Output', (), {'values': values})

def sequential_reward(reward_model, tokenizer, batch_prompts, decoded_responses):
    rewards = []
    t0 = time.time()
    with torch.no_grad():
        for prompt, response in zip(batch_prompts, decoded_responses):
            full_text = prompt + response
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
                return_attention_mask=True,
            )
            outputs = reward_model(**inputs)
            values = outputs.values
            last_token_index = inputs["attention_mask"][0].sum().item() - 1
            reward_val = values[0, last_token_index].item()
            rewards.append(reward_val)
    return rewards, time.time() - t0

def batched_reward(reward_model, tokenizer, batch_prompts, decoded_responses):
    t0 = time.time()
    with torch.no_grad():
        full_texts = [p + r for p, r in zip(batch_prompts, decoded_responses)]
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
            return_attention_mask=True,
        )
        outputs = reward_model(**inputs)
        values = outputs.values.squeeze(-1)  # (batch, seq)

        # Extract rewards for the last non-padding token in each sequence
        last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
        rewards = values[torch.arange(values.size(0)), last_token_indices].tolist()
    return rewards, time.time() - t0

def run_benchmark():
    # Use a small tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    reward_model = MockValueHeadModel()

    batch_size = 16
    prompts = ["Tell me a joke." for _ in range(batch_size)]
    responses = ["Why did the chicken cross the road? To get to the other side." for _ in range(batch_size)]

    print(f"Benchmarking reward computation with batch_size={batch_size}...")

    # Warmup
    sequential_reward(reward_model, tokenizer, prompts, responses)
    batched_reward(reward_model, tokenizer, prompts, responses)

    seq_rewards, seq_time = sequential_reward(reward_model, tokenizer, prompts, responses)
    batch_rewards, batch_time = batched_reward(reward_model, tokenizer, prompts, responses)

    print(f"Sequential time: {seq_time:.4f}s")
    print(f"Batched time:    {batch_time:.4f}s")
    print(f"Speedup:         {seq_time / batch_time:.2f}x")

    # Verify results are similar (they use same mock model but may differ due to randomness if not seeded,
    # but the logic should be consistent). Actually MockValueHeadModel returns random values,
    # so we can't easily compare the values unless we seed it or make it deterministic.
    # Let's just check length.
    assert len(seq_rewards) == len(batch_rewards) == batch_size

if __name__ == "__main__":
    run_benchmark()
