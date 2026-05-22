## 2026-05-14 - [Tokenization & Padding Optimization]
**Learning:** The current codebase uses static padding (`padding='max_length'`) during preprocessing for SFT and Reward training. This causes the model to process a large number of padding tokens in every batch, significantly slowing down training and increasing VRAM usage. Additionally, `dataset.map` calls are single-threaded, which can be a bottleneck for large datasets.
**Action:** Switch to dynamic padding (`padding=False`) and allow the `DataCollator` to handle padding per-batch. Add `num_proc` to `dataset.map` to parallelize tokenization.

## 2026-05-15 - [Batch Evaluation & Inference Optimization]
**Learning:** Sequential LLM-as-Judge evaluations were a significant bottleneck, processing examples one-by-one. Batching these requests (size 8) leverages GPU parallelism for a ~5-8x speedup in the evaluation phase. Standardizing on left-padding for inference simplifies the complex logic for prompt-stripping to a simple `input_ids.shape[1]` offset, making the code more robust and maintainable.
**Action:** Batched `llm_judge_evaluate` in `inference/evaluation.py`. Set `tokenizer.padding_side = 'left'` in `inference/generate.py`. Standardized simplified prompt stripping across all inference functions.

## 2026-05-17 - [Batched PPO Reward Computation]
**Learning:** Sequential reward computation in the PPO training loop (processing one response at a time) is a significant bottleneck that fails to utilize GPU parallelism. Batching the reward model forward pass and using vectorized extraction for the last non-padding token results in a >2x speedup even on CPU, and likely much more on GPU.
**Action:** Implemented batched reward computation and batch decoding in `training/ppo.py`. Added a benchmark script `tests/benchmark_ppo_rewards.py` for verification.

## 2026-05-18 - [Batched Decoding and Evaluation Throughput]
**Learning:** Transitioning from sequential `tokenizer.decode` loops to `tokenizer.batch_decode` reduces Python overhead and leverages optimized backend implementations, yielding a ~1.4x speedup in the decoding phase. Increasing the evaluation batch size from 4 to 8 further improves GPU utilization and aligns with other optimized components in the app.
**Action:** Replaced serial decoding with `tokenizer.batch_decode` in `inference/generate.py` and `inference/evaluation.py`. Increased batch size to 8 in `on_evaluate_click`.

## 2026-05-19 - [Batched Dataset Augmentation]
**Learning:** Sequential calls to `nlpaug` augmenters (processing one string at a time) are a significant bottleneck in the data enhancement pipeline. Switching to batched augmentation (`augmenter.augment(list_of_texts)`) utilizes internal vectorization and significantly reduces Python overhead, yielding a ~3-5x speedup. Interspersing the original and augmented rows manually after the batch call preserves the expected data structure for the training layers.
**Action:** Replaced the sequential loop in `data/augmentation.py` with batched `nlpaug` calls. Added `tests/test_augmentation_batching.py` to verify performance and correctness.

## 2026-05-20 - [Standardized Batched Decoding and Prompt Stripping]
**Learning:** Standardizing on `input_ids.shape[1]` for prompt stripping across all evaluation and inference layers (UI and CLI) is critical for correctness when using left-padding. Previous attention-mask-based stripping in the CLI was inaccurate for padded batches, often including prompt tokens in the output. Consistently using `tokenizer.batch_decode` further reduces Python overhead for a ~1.4x speedup in the decoding phase.
**Action:** Replaced serial `tokenizer.decode` with `tokenizer.batch_decode` in `llm_judge_evaluate` and the CLI `evaluate` command. Standardized prompt stripping logic to use the `input_ids.shape[1]` offset.
