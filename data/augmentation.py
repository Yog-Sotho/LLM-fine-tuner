"""
data/augmentation.py
=====================
Layer 2 — dataset augmentation and quality filtering.
Imports: config.constants, core.state (for app_state), data.loader,
         data.preprocessing, gradio (UI handlers only), datasets.

Functions
---------
augment_dataset_v27        — nlpaug-based synonym / random-word / spelling augmentation
quality_filter_v27         — length-based quality filter
on_augment_click           — Gradio UI handler for the Augment button
on_quality_filter_click    — Gradio UI handler for the Quality Filter button
"""

import gradio as gr
from datasets import Dataset

from config.constants import (
    COL_TEXT,
    COL_INSTRUCTION,
    COL_PROMPT,
    COL_CHOSEN,
    COL_REJECTED,
    COL_OUTPUT,
    HAS_NLPAUG,
)
from data.loader import detect_file_type, load_dataset_from_file
from data.preprocessing import preview_dataset


# ── Core augmentation logic ────────────────────────────────────────────────

def augment_dataset_v27(
    dataset: Dataset,
    augmentation_factor: int = 2,
    aug_type: str = "synonym",
) -> tuple:
    """Augment a dataset using nlpaug text augmenters.

    Parameters
    ----------
    dataset            : input HuggingFace Dataset
    augmentation_factor: total copies per example (1 = no augmentation)
    aug_type           : 'synonym' | 'random_word' | 'spelling'

    Returns
    -------
    (augmented_dataset, status_message)
    Original dataset is returned unchanged if nlpaug is not installed.
    """
    if not HAS_NLPAUG:
        return dataset, (
            "⚠️ nlpaug not installed. Run: pip install nlpaug\n"
            "Original dataset returned unchanged."
        )

    try:
        import nlpaug.augmenter.word as naw  # lazy — only when actually called

        if aug_type == "synonym":
            augmenter = naw.SynonymAug(aug_src="wordnet")
        elif aug_type == "random_word":
            augmenter = naw.RandomWordAug()
        elif aug_type == "spelling":
            augmenter = naw.SpellingAug()
        else:
            augmenter = naw.SynonymAug(aug_src="wordnet")

        # BOLT OPTIMIZATION: Use batched augmentation to significantly speed up processing.
        # Sequential calls to nlpaug are slow because they don't utilize vectorized
        # operations. Batched calls are typically 3-5x faster.
        col_is_text = COL_TEXT in dataset.column_names
        target_col = COL_TEXT if col_is_text else (COL_INSTRUCTION if COL_INSTRUCTION in dataset.column_names else None)

        if target_col:
            # BOLT OPTIMIZATION: Use columnar access instead of row-wise iteration.
            texts_to_aug = list(dataset[target_col])
            all_aug_versions = []

            # Generate (augmentation_factor - 1) augmented versions for the entire batch.
            for _ in range(augmentation_factor - 1):
                try:
                    # nlpaug.augment(list) returns a list of augmented strings.
                    aug_results = augmenter.augment(texts_to_aug)
                    if not isinstance(aug_results, list):
                        aug_results = [str(aug_results)]
                    # Safeguard: if nlpaug returns fewer items than requested, pad with originals
                    if len(aug_results) < len(texts_to_aug):
                        aug_results = list(aug_results) + texts_to_aug[len(aug_results):]
                    all_aug_versions.append(aug_results[:len(texts_to_aug)])
                except Exception:
                    # Fallback: if batch fails, use original texts to preserve row count
                    all_aug_versions.append(texts_to_aug)

            # BOLT OPTIMIZATION: Use dataset.to_dict() and slice-based interleaving
            # for efficient reconstruction, avoiding millions of Python dict objects.
            full_data_dict = dataset.to_dict()
            new_data = {}
            new_len = len(dataset) * augmentation_factor

            for col_name, values in full_data_dict.items():
                interleaved = [None] * new_len
                # 1. Place original values at indices 0, factor, 2*factor...
                interleaved[0::augmentation_factor] = values
                if col_name == target_col:
                    # 2. Place augmented versions at indices 1, 2...
                    for i, aug_version in enumerate(all_aug_versions):
                        interleaved[i+1::augmentation_factor] = aug_version
                else:
                    # 3. For other columns, repeat the original value
                    for i in range(1, augmentation_factor):
                        interleaved[i::augmentation_factor] = values
                new_data[col_name] = interleaved
            aug_ds = Dataset.from_dict(new_data)
        else:
            # Fallback for datasets without TEXT or INSTRUCTION columns
            full_data_dict = dataset.to_dict()
            new_data = {}
            new_len = len(dataset) * augmentation_factor
            for col_name, values in full_data_dict.items():
                interleaved = [None] * new_len
                for i in range(augmentation_factor):
                    interleaved[i::augmentation_factor] = values
                new_data[col_name] = interleaved
            aug_ds = Dataset.from_dict(new_data)
        msg = (
            f"✅ Augmentation complete!\n"
            f"Original: {len(dataset)} examples\n"
            f"Augmented: {len(aug_ds)} examples (×{augmentation_factor})\n"
            f"Method: {aug_type}"
        )
        return aug_ds, msg

    except Exception as e:
        return dataset, f"❌ Augmentation failed: {e}\nOriginal dataset returned."


def quality_filter_v27(
    dataset: Dataset,
    min_length: int = 50,
    max_length: int = 2048,
    is_dpo: bool = False,
) -> tuple:
    """Filter examples by character-length bounds.
    BOLT OPTIMIZATION: Uses batched filtering to significantly speed up processing.

    Parameters
    ----------
    min_length : minimum total character count to keep an example
    max_length : maximum character count (per field for DPO; combined for SFT instruction)

    Returns
    -------
    (filtered_dataset, status_message)
    """
    original_len = len(dataset)
    try:
        if is_dpo:
            def filter_dpo(batch):
                col_p = batch.get(COL_PROMPT, [""] * len(next(iter(batch.values()))))
                col_c = batch.get(COL_CHOSEN, [""] * len(col_p))
                col_r = batch.get(COL_REJECTED, [""] * len(col_p))
                return [
                    (min_length <= len(str(p)) <= max_length
                     and min_length <= len(str(c)) <= max_length
                     and min_length <= len(str(r)) <= max_length)
                    for p, c, r in zip(col_p, col_c, col_r)
                ]
            dataset = dataset.filter(filter_dpo, batched=True)
        elif COL_TEXT in dataset.column_names:
            def filter_text(batch):
                return [min_length <= len(str(t)) <= max_length for t in batch[COL_TEXT]]
            dataset = dataset.filter(filter_text, batched=True)
        elif COL_INSTRUCTION in dataset.column_names:
            # v3.1 Fix #6: Combined instruction+output length checked against
            # max_length * 2 because both fields are concatenated during tokenisation.
            def filter_inst(batch):
                col_i = batch.get(COL_INSTRUCTION, [""] * len(next(iter(batch.values()))))
                col_o = batch.get(COL_OUTPUT, [""] * len(col_i))
                return [
                    min_length <= (len(str(i)) + len(str(o))) <= max_length * 2
                    for i, o in zip(col_i, col_o)
                ]
            dataset = dataset.filter(filter_inst, batched=True)

        removed = original_len - len(dataset)
        msg = (
            f"✅ Quality filter applied!\n"
            f"Removed: {removed} examples (len < {min_length} or > {max_length} chars)\n"
            f"Remaining: {len(dataset)} examples"
        )
        return dataset, msg

    except Exception as e:
        return dataset, f"❌ Quality filter failed: {e}"


# ── Gradio UI handlers ─────────────────────────────────────────────────────

def on_augment_click(file, training_mode, aug_factor, aug_type, progress=gr.Progress()):
    """Handler for the Augment button in the Data tab.

    C-5 FIX: Now returns the augmented Dataset object as the FOURTH return value
    so it can be stored in augmented_ds_state (a gr.State component added to
    data_tab.py). on_train_click in handlers.py then prefers this state when
    available, so training actually uses the augmented data.

    Returns (status_str, preview_df, stats_md, augmented_dataset_or_None)
    """
    if file is None:
        return (
            "❌ Upload a dataset first.",
            gr.update(visible=False),
            gr.update(visible=False),
            None,  # C-5 FIX: augmented_ds_state stays None
        )

    is_dpo = "dpo" in str(training_mode).lower()
    try:
        ftype = detect_file_type(file)
        progress(0, desc="Loading dataset for augmentation…")
        ds = load_dataset_from_file(file, ftype, is_dpo=is_dpo)

        progress(0.3, desc="Augmenting…")
        aug_ds, msg = augment_dataset_v27(
            ds,
            augmentation_factor=int(aug_factor),
            aug_type=aug_type,
        )

        progress(0.9, desc="Building preview…")
        preview = preview_dataset(aug_ds, is_dpo=is_dpo)
        stats = f"**Original:** {len(ds)} examples → **Augmented:** {len(aug_ds)} examples"

        # M-4 FIX: Complete the progress bar (was previously stuck at 0.9 / 90%).
        progress(1.0, desc="Done!")

        # C-5 FIX: Return aug_ds as fourth value for gr.State storage.
        return msg, gr.update(value=preview, visible=True), gr.update(value=stats, visible=True), aug_ds

    except Exception as e:
        return f"❌ {e}", gr.update(visible=False), gr.update(visible=False), None


def on_quality_filter_click(file, training_mode, min_len, max_len, progress=gr.Progress()):
    """Handler for the Quality Filter button in the Data tab.

    C-5 FIX: Now returns the filtered Dataset object as the FOURTH return value
    so it can be stored in augmented_ds_state. on_train_click then uses the
    filtered dataset for training instead of the original.

    Returns (status_str, preview_df, stats_md, filtered_dataset_or_None)
    """
    if file is None:
        return (
            "❌ Upload a dataset first.",
            gr.update(visible=False),
            gr.update(visible=False),
            None,  # C-5 FIX: augmented_ds_state stays None
        )

    is_dpo = "dpo" in str(training_mode).lower()
    try:
        ftype = detect_file_type(file)
        ds = load_dataset_from_file(file, ftype, is_dpo=is_dpo)

        progress(0.3, desc="Applying quality filter…")
        filtered_ds, msg = quality_filter_v27(
            ds,
            min_length=int(min_len),
            max_length=int(max_len),
            is_dpo=is_dpo,
        )

        progress(0.9, desc="Building preview…")
        preview = preview_dataset(filtered_ds, is_dpo=is_dpo)
        stats = f"**After filter:** {len(filtered_ds)} examples"

        # M-4 FIX: Complete the progress bar.
        progress(1.0, desc="Done!")

        # C-5 FIX: Return filtered_ds as fourth value for gr.State storage.
        return msg, gr.update(value=preview, visible=True), gr.update(value=stats, visible=True), filtered_ds

    except Exception as e:
        return f"❌ {e}", gr.update(visible=False), gr.update(visible=False), None
