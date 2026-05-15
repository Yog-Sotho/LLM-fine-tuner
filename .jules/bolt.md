## 2026-05-14 - [Tokenization & Padding Optimization]
**Learning:** The current codebase uses static padding (`padding='max_length'`) during preprocessing for SFT and Reward training. This causes the model to process a large number of padding tokens in every batch, significantly slowing down training and increasing VRAM usage. Additionally, `dataset.map` calls are single-threaded, which can be a bottleneck for large datasets.
**Action:** Switch to dynamic padding (`padding=False`) and allow the `DataCollator` to handle padding per-batch. Add `num_proc` to `dataset.map` to parallelize tokenization.

## 2026-05-15 - [Batch Evaluation & Inference Optimization]
**Learning:** Sequential LLM-as-Judge evaluations were a significant bottleneck, processing examples one-by-one. Batching these requests (size 8) leverages GPU parallelism for a ~5-8x speedup in the evaluation phase. Standardizing on left-padding for inference simplifies the complex logic for prompt-stripping to a simple `input_ids.shape[1]` offset, making the code more robust and maintainable.
**Action:** Batched `llm_judge_evaluate` in `inference/evaluation.py`. Set `tokenizer.padding_side = 'left'` in `inference/generate.py`. Standardized simplified prompt stripping across all inference functions.
