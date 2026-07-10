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

## 2026-05-21 - [Standardized Batched Decoding in CLI and Judge]
**Learning:** Manual serial decoding loops in CLI commands and LLM-as-Judge evaluations were inconsistent and slower than necessary. Transitioning to `tokenizer.batch_decode` across all evaluation paths (UI and CLI) reduces Python overhead and standardizes prompt-stripping logic. Increasing the CLI default batch size to 8 aligns it with the optimized UI evaluation throughput.
**Action:** Replaced serial loops with `tokenizer.batch_decode` in `inference/evaluation.py` and `cli/commands.py`. Increased default batch size to 8 in the CLI `evaluate` command. Updated `tests/test_evaluation_batching.py` to verify the optimized logic.

## 2026-05-23 - [Vectorized Dataset Preprocessing]
**Learning:** Sequential Python loops and list comprehensions in `validate_and_clean_dataset` are a major bottleneck for large datasets (100k+ rows). Transitioning to vectorized Pandas operations for string stripping, filtering, and deduplication yields a ~36x speedup (0.5s vs 18.2s for 100k rows) while maintaining exact logical parity.
**Action:** Refactored `validate_and_clean_dataset` in `data/preprocessing.py` to use Pandas `to_pandas()` followed by vectorized masking and `drop_duplicates()`. Added `tests/benchmark_preprocessing.py` for verification.

## 2026-05-28 - [PPO Batched Tokenization]
**Learning:** Sequential tokenization of prompts inside the PPO training loop is a major bottleneck, especially when repeated across multiple epochs. Pre-tokenizing the entire dataset using a single batched 'tokenizer()' call outside the loop avoids redundant work and leverages optimized backend implementations, providing a ~4.5x speedup in the preprocessing phase.
**Action:** Implemented pre-tokenization and batched query tensor storage in 'training/ppo.py'. Verified logic with a mock-based unit test 'tests/test_ppo_pre_tokenization.py'.

## 2026-06-05 - [Vectorized Stats & Parallel Trainer Tokenization]
**Learning:** Manual list comprehension loops for dataset statistics (average length) in UI handlers are a major bottleneck for 100k+ row datasets. Converting to vectorized Pandas operations via `to_pandas()` and `.str.len().mean()` yields a ~240x-450x speedup. Furthermore, TRL's `DPOTrainer` and `ORPOTrainer` (and HF's `Trainer`) can parallelize internal tokenization if `dataset_num_proc` is passed, which significantly reduces training startup time.
**Action:** Implemented `get_dataset_stats` in `data/preprocessing.py`. Updated `ui/handlers.py` to use it with a safety fallback. Added `dataset_num_proc=os.cpu_count()` to SFT, DPO, and ORPO trainer constructors with `inspect.signature` guards for backward compatibility.

## 2026-06-12 - [Optimized Dataset Indexing for Previews]
**Learning:** In Hugging Face Datasets, accessing a column first and then slicing (`dataset[COL][:N]`) loads the entire column into memory as a Python list before taking the slice. This is extremely inefficient for large datasets (e.g., 1M+ rows). Slicing the dataset first and then accessing the column (`dataset[:N][COL]`) only retrieves the requested rows, providing a verified ~6x speedup and significantly lower memory overhead.
**Action:** Optimized `preview_dataset` in `data/preprocessing.py` to use the `dataset[:N][COL]` pattern. Verified the improvement with `tests/benchmark_indexing_bolt.py`.
## 2026-06-10 - [Vectorized Augmentation Reconstruction]
**Learning:** While the augmentation process was already batched, the subsequent dataset reconstruction (interleaving original and augmented rows) used a row-wise Python loop that was a major bottleneck for large datasets. Replacing this with vectorized Pandas operations (`pd.concat` and `sort_index(kind='stable')`) yields a ~17x speedup in the reconstruction phase. Additionally, switching from row-wise list comprehensions to direct columnar access (`dataset[COL]`) for extracting texts to augment provides an even more significant speedup (~10,000x for that specific step).
**Action:** Optimized `augment_dataset_v27` in `data/augmentation.py` with vectorized reconstruction and direct column access. Added `tests/benchmark_augmentation_bolt.py` for verification.
## 2026-06-10 - [Efficient Dataset Column Access and Slicing]
**Learning:** In Hugging Face Datasets, row-wise iteration (e.g., `[x[COL] for x in dataset]`) is extremely slow because it converts every row into a dictionary. Direct column access (e.g., `dataset[COL]`) is up to 50,000x faster. Additionally, slicing a column after extraction (e.g., `dataset[COL][:N]`) loads the entire column into memory, which is inefficient for large datasets. Slicing before access (e.g., `dataset[:N][COL]`) only loads the required rows.
**Action:** Replace row-wise iteration with direct column access and use `dataset[:N][COL]` for efficient data previews. Verified with benchmarks showing 57,000x speedup for extraction and 5x speedup for slicing on 1M rows.

## 2026-07-10 - [In-Memory Dataset Refresh Optimization]
**Learning:** Re-loading datasets for UI previews by writing Pandas DataFrames to temporary files and reading them back into HuggingFace Datasets (the previous pattern in `on_refresh_preview`) introduces unnecessary disk I/O latency. Bypassing the disk and converting directly from DataFrame to Dataset using `Dataset.from_pandas` with proper column mapping and type consistency (`fillna("")`, `astype(str)`) yields a verified ~3.9x speedup.
**Action:** Implemented `load_dataset_from_dataframe` in `data/loader.py` and refactored `on_refresh_preview` in `ui/handlers.py` to use it. Added `tests/benchmark_refresh_bolt.py` for verification.
