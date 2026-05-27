
import time
import torch
from transformers import AutoTokenizer

def sequential_tokenization(tokenizer, batch_prompts):
    t0 = time.time()
    query_tensors = [
        tokenizer.encode(p, return_tensors="pt", truncation=True, max_length=512).squeeze(0)
        for p in batch_prompts
    ]
    return query_tensors, time.time() - t0

def batched_tokenization(tokenizer, batch_prompts):
    t0 = time.time()
    # BOLT OPTIMIZATION: Use batched tokenization
    inputs = tokenizer(
        batch_prompts,
        truncation=True,
        max_length=512,
    )
    query_tensors = [torch.tensor(ids) for ids in inputs['input_ids']]
    return query_tensors, time.time() - t0

def run_benchmark():
    # Use a small tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 128 # Increase batch size to see more impact
    prompts = ["Tell me a joke about a chicken crossing the road to get to the other side. " * 5 for _ in range(batch_size)]

    print(f"Benchmarking PPO tokenization with batch_size={batch_size}...")

    # Warmup
    sequential_tokenization(tokenizer, prompts)
    batched_tokenization(tokenizer, prompts)

    # Run multiple times to get a better average
    seq_times = []
    batch_times = []
    for _ in range(10):
        _, t = sequential_tokenization(tokenizer, prompts)
        seq_times.append(t)
        _, t = batched_tokenization(tokenizer, prompts)
        batch_times.append(t)

    seq_time = sum(seq_times) / len(seq_times)
    batch_time = sum(batch_times) / len(batch_times)

    print(f"Sequential time: {seq_time:.4f}s")
    print(f"Batched time:    {batch_time:.4f}s")
    print(f"Speedup:         {seq_time / batch_time:.2f}x")

    # Verify results
    seq_tensors, _ = sequential_tokenization(tokenizer, prompts)
    batch_tensors, _ = batched_tokenization(tokenizer, prompts)
    assert len(seq_tensors) == len(batch_tensors) == batch_size
    for s, b in zip(seq_tensors, batch_tensors):
        assert torch.equal(s, b)

if __name__ == "__main__":
    run_benchmark()
