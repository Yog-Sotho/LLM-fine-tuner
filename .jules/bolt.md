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

## 2026-06-12 - [Optimized Dataset Preview Indexing]
**Learning:** In HuggingFace Datasets, the indexing pattern `dataset[COL][:N]` is significantly less efficient than `dataset[:N][COL]` for large datasets. The former loads the entire column into memory before slicing, while the latter performs a row-wise slice first, returning a small dictionary. This avoids massive memory overhead and provides a verified ~4x-14x speedup for previews.
**Action:** Refactored `preview_dataset` in `data/preprocessing.py` to use the `dataset[:N][COL]` pattern across all dataset types (SFT and DPO).
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

## 2026-07-25 - [Optimized JSON/JSONL Loading]
**Learning:** Manual Python loops for parsing JSONL files and using `json.load` followed by `Dataset.from_list` are less efficient than the native `Dataset.from_json` method. The native method leverages the optimized Arrow-backed C++ implementation of the `datasets` library, providing a verified ~1.26x speedup. Furthermore, experiments with `num_proc` for simple operations like filtering revealed that multiprocessing overhead can actually lead to a significant slowdown (up to 5x) compared to serial processing on datasets with 1M rows.
**Action:** Prefer native `Dataset.from_json`/`from_csv` methods over manual parsing. Be cautious with `num_proc` parallelism for lightweight operations where the overhead of process spawning and data serialization may outweigh the computation gains.

## 2026-07-12 - [Memory-Safe Vectorized Filtering]
**Learning:** Forcing a full dataset conversion to Pandas via `to_pandas()` provides maximum speed but introduces a critical risk of OOM errors on very large datasets. Implementing vectorized Pandas logic *inside* a batched `dataset.filter(batched=True)` call preserves memory mapping/chunking benefits while still providing a ~4.5x speedup over standard Python-loop batch filtering. A larger `batch_size` (e.g., 10,000+) is necessary to amortize the overhead of per-batch Pandas Series creation.
**Action:** Use batched vectorized filtering (Pandas within `dataset.filter`) instead of full DataFrame conversion for memory-safe performance on large datasets.

## 2026-08-01 - [Inference Mode Optimization]
**Learning:** Replacing `torch.no_grad()` with `torch.inference_mode()` in pure inference paths (generation, evaluation, and PPO reward computation) provides a verified performance gain by disabling view tracking. Benchmarks on CPU showed a ~3% speedup for view-intensive operations. While the improvement for simple matrix multiplications is negligible on CPU, it is a best practice for modern PyTorch (1.9+) that yields better performance and safety by preventing accidental gradient computation in inference blocks.
**Action:** Replaced `torch.no_grad()` with `torch.inference_mode()` in `inference/generate.py`, `inference/evaluation.py`, `training/ppo.py`, and `cli/commands.py`.

## 2026-08-05 - [UI Dataset Preview Slicing Optimization]
**Learning:** Performing multiple slicing operations (such as calling `dataset[:5]` twice) on a HuggingFace `Dataset` introduces redundant slicing overhead. Additionally, retrieving elements via multiple duplicate `.get()` calls or unused variables maps redundant operations. Slicing exactly once and using direct dictionary lookup conditional on the column names being present avoids multiple lookup/slicing cycles, yielding up to ~1.16x speedup for UI dataset previews.
**Action:** Optimized `preview_dataset` in `data/preprocessing.py` to slice once and perform direct dictionary lookup based on column names.

## 2026-08-10 - [Arrow Compute for Dataset Statistics]
**Learning:** Converting HuggingFace Datasets to Pandas via `.to_pandas()` solely to compute string statistics (like string length) introduces unnecessary memory overhead and object translation latency. Performing string and mathematical operations directly on the underlying PyArrow Table via `pyarrow.compute` (e.g. `utf8_length`, `fill_null`, `add`, `mean`) operates entirely in optimized C++ on Arrow's native memory layout. This avoids full DataFrame copies and yields a verified ~1.3x to 2.3x speedup on large datasets.
**Action:** Prefer `pyarrow.compute` functions for simple element-wise transformations or aggregations (like lengths, fills, means, additions) directly on `dataset.data` instead of converting to Pandas.

## 2026-08-15 - [Vectorized DPO Dataset Deduplication]
**Learning:** Duplicate preference pairs in DPO datasets cause redundant training iterations and waste valuable CPU/GPU resources during alignment steps (DPO, ORPO, Reward Model, PPO). Applying vectorized Pandas `drop_duplicates` on the `COL_PROMPT`, `COL_CHOSEN`, and `COL_REJECTED` columns achieves extremely fast O(N) deduplication (~0.15 seconds on 100k rows) while ensuring correct input structure and preventing redundant training.
**Action:** Use vectorized subset-based deduplication with `drop_duplicates` for preference-based datasets before tokenization.

## 2026-08-20 - [Single-Pass CSV & Excel Parsing in File Upload]
**Learning:** In the file-upload pipeline, CSV and Excel files were previously read and parsed from disk twice: first inside `load_dataset_from_file` to construct a Hugging Face Dataset, and then immediately after inside `on_file_upload` to extract metadata and store the raw DataFrame. Reading and parsing the exact same file twice introduces significant redundant disk I/O and CPU parsing overhead. Bypassing the duplicate load by reading once into Pandas and then using `load_dataset_from_dataframe` directly yields a verified ~1.6x to 2.8x speedup.
**Action:** Always verify if a file being loaded for metadata extraction is also being parsed for dataset creation, and consolidate them into a single-pass in-memory loader to avoid redundant disk and CPU cycles.

## 2026-08-25 - [Column-Selective CSV Loading]
**Learning:** During batch generation and evaluation, loading full multi-column CSV datasets into memory via `pd.read_csv` when only `prompt` and `reference` are needed introduces severe and unnecessary I/O and memory overhead. By inspecting headers first (`nrows=0`) and loading only the required columns with `usecols`, we avoid loading other heavy or unused columns entirely, achieving a verified ~2.4x speedup and major VRAM/RAM savings.
**Action:** Use header-first column checking and `usecols` whenever reading specific columns from CSV datasets for inference or evaluation.

## 2026-08-30 - [Post-Cleaning Sequence Length Computation]
**Learning:** In dataset cleaning pipelines, calculating character lengths for truncation warnings prior to row filtering and deduplication leads to redundant string computations on dropped rows. Deferring length calculation until *after* filtering and deduplication eliminates wasted CPU cycles and avoids expensive pandas index realignment (`.loc[df.index]`).
**Action:** Compute sequence lengths only on the final, unique, cleaned DataFrame rows.

## 2026-08-30 - [Explicit Fast Tokenizer Force for Inference]
**Learning:** Hugging Face `AutoTokenizer.from_pretrained` might load the slow Python-only tokenizer unless `use_fast=True` is explicitly specified. Forcing the fast Rust-backed tokenizer reduces CPU tokenization and batch decoding overhead during inference and automated evaluation. However, wrapping `BatchEncoding` to copy keys to devices via custom dict comprehension is slightly faster than native `.to()` on CPU due to the overhead of HF input validation wrapper functions.
**Action:** Always explicitly specify `use_fast=True` for AutoTokenizers. Keep simple dictionary comprehensions for moving tokenized batch tensors to device on CPU.

## 2026-09-02 - [In-Place Dataset Column Pre-Stripping]
**Learning:** In `validate_and_clean_dataset`, string columns were repeatedly cast and stripped (once for filtering, and then again during sequence length warnings). Applying casting and stripping (`df[COL] = df[COL].astype(str).str.strip()`) in-place during the initial pass eliminates redundant memory copies, duplicate `.astype(str)` allocations, and extra `.str.strip()` operations, yielding a verified ~11% speedup and guaranteeing training dataset hygiene.
**Action:** Perform string casting and stripping in-place on DataFrame columns during validation to reuse the clean columns in subsequent steps.

## 2026-09-10 - [Multiprocessing Fork Overhead for Large Libraries]
**Learning:** Parallelizing simple CPU-bound metrics (like BLEU or ROUGE) in an application loaded with massive libraries (PyTorch, Transformers, etc.) can actually be slower than sequential execution if workers are spawned. This is because the default `spawn` method re-imports all dependencies for every worker process. Utilizing the `fork` start method on Unix/Linux preserves the memory state and avoids re-import overhead, enabling true CPU parallelism and speeding up scoring by ~25x on large datasets.
**Action:** Prioritize the high-performance `fork` start method on Unix/Linux systems for process pools, and always implement a robust fallback to sequential execution if spawning or execution fails.
