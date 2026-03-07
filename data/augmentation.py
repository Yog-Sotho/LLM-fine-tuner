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

        augmented_rows = []
        col_is_text = COL_TEXT in dataset.column_names

        for example in dataset:
            augmented_rows.append(dict(example))
            for _ in range(augmentation_factor - 1):
                new_example = dict(example)
                try:
                    if col_is_text:
                        aug_result = augmenter.augment(str(example[COL_TEXT]))
                        new_example[COL_TEXT] = (
                            aug_result[0] if isinstance(aug_result, list) else str(aug_result)
                        )
                    elif COL_INSTRUCTION in example:
                        aug_result = augmenter.augment(str(example[COL_INSTRUCTION]))
                        new_example[COL_INSTRUCTION] = (
                            aug_result[0] if isinstance(aug_result, list) else str(aug_result)
                        )
                    augmented_rows.append(new_example)
                except Exception:
                    # On per-example failure, keep original rather than losing the row
                    augmented_rows.append(dict(example))

        aug_ds = Dataset.from_list(augmented_rows)
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
            dataset = dataset.filter(
                lambda x: (
                    min_length <= len(str(x.get(COL_PROMPT,   ""))) <= max_length
                    and min_length <= len(str(x.get(COL_CHOSEN,  ""))) <= max_length
                    and min_length <= len(str(x.get(COL_REJECTED,""))) <= max_length
                )
            )
        elif COL_TEXT in dataset.column_names:
            dataset = dataset.filter(
                lambda x: min_length <= len(str(x[COL_TEXT])) <= max_length
            )
        elif COL_INSTRUCTION in dataset.column_names:
            # v3.1 Fix #6: Combined instruction+output length checked against
            # max_length * 2 because both fields are concatenated during tokenisation.
            dataset = dataset.filter(
                lambda x: (
                    min_length
                    <= len(str(x.get(COL_INSTRUCTION, ""))) + len(str(x.get(COL_OUTPUT, "")))
                    <= max_length * 2
                )
            )

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
    """Handler for the Augment button in the Data tab."""
    if file is None:
        return (
            "❌ Upload a dataset first.",
            gr.update(visible=False),
            gr.update(visible=False),
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

        preview = preview_dataset(aug_ds, is_dpo=is_dpo)
        stats = f"**Original:** {len(ds)} examples → **Augmented:** {len(aug_ds)} examples"

        # v3.1 Fix #3: Wrap in gr.update(visible=True) so the hidden
        # components actually appear after the click.
        return msg, gr.update(value=preview, visible=True), gr.update(value=stats, visible=True)

    except Exception as e:
        return f"❌ {e}", gr.update(visible=False), gr.update(visible=False)


def on_quality_filter_click(file, training_mode, min_len, max_len, progress=gr.Progress()):
    """Handler for the Quality Filter button in the Data tab."""
    if file is None:
        return (
            "❌ Upload a dataset first.",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    is_dpo = "dpo" in str(training_mode).lower()
    try:
        ftype = detect_file_type(file)
        ds = load_dataset_from_file(file, ftype, is_dpo=is_dpo)
        filtered_ds, msg = quality_filter_v27(
            ds,
            min_length=int(min_len),
            max_length=int(max_len),
            is_dpo=is_dpo,
        )

        preview = preview_dataset(filtered_ds, is_dpo=is_dpo)
        stats = f"**After filter:** {len(filtered_ds)} examples"

        # v3.1 Fix #3
        return msg, gr.update(value=preview, visible=True), gr.update(value=stats, visible=True)

    except Exception as e:
        return f"❌ {e}", gr.update(visible=False), gr.update(visible=False)
