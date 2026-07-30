import time
from transformers import AutoTokenizer

def benchmark():
    model_name = "gpt2"
    print(f"Loading fast tokenizer (use_fast=True) for {model_name}...")
    t0 = time.time()
    tokenizer_fast = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer_fast.pad_token = tokenizer_fast.eos_token
    fast_load = time.time() - t0
    print(f"Fast load: {fast_load:.4f}s")

    print(f"Loading slow/default tokenizer (use_fast=False) for {model_name}...")
    t0 = time.time()
    tokenizer_slow = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer_slow.pad_token = tokenizer_slow.eos_token
    slow_load = time.time() - t0
    print(f"Slow load: {slow_load:.4f}s")

    # Generate some dummy prompts
    prompts = [f"This is sample sentence number {i} to test tokenizer performance." for i in range(10000)]

    print(f"\nBenchmarking encoding of {len(prompts)} prompts...")

    t0 = time.time()
    encoded_fast = tokenizer_fast(prompts, padding=True, truncation=True, max_length=512)
    fast_enc = time.time() - t0
    print(f"Fast encoding: {fast_enc:.4f}s")

    t0 = time.time()
    encoded_slow = tokenizer_slow(prompts, padding=True, truncation=True, max_length=512)
    slow_enc = time.time() - t0
    print(f"Slow encoding: {slow_enc:.4f}s")

    print(f"Encoding Speedup: {slow_enc / fast_enc:.2f}x")

    # Benchmarking decoding
    input_ids_list = encoded_fast["input_ids"]

    print(f"\nBenchmarking decoding of {len(input_ids_list)} tokenized sequences...")

    t0 = time.time()
    decoded_fast = tokenizer_fast.batch_decode(input_ids_list, skip_special_tokens=True)
    fast_dec = time.time() - t0
    print(f"Fast decoding: {fast_dec:.4f}s")

    t0 = time.time()
    decoded_slow = tokenizer_slow.batch_decode(input_ids_list, skip_special_tokens=True)
    slow_dec = time.time() - t0
    print(f"Slow decoding: {slow_dec:.4f}s")

    print(f"Decoding Speedup: {slow_dec / fast_dec:.2f}x")

if __name__ == "__main__":
    benchmark()
