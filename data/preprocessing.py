"""
data/preprocessing.py
======================
Layer 2 — dataset validation, cleaning, preview and tokenisation.
Imports: config.constants, stdlib, pandas, datasets, transformers.

Functions
---------
validate_and_clean_dataset — filter empty/long rows; return issues list
preview_dataset            — return first N rows as a pandas DataFrame
preprocess_function        — tokenise examples; apply chat template if available

Fix log
-------
  M4 (Medium): Duplicate detection previously counted duplicates and warned
     the user but never removed them. The training loop then saw repeated
     examples, leading to overfitting and inflated epoch counts. Fixed by
     using an ordered seen-set to select unique indices via
     `Dataset.select()`, preserving original order while removing duplicates.
     The issues message now says "removed" instead of "detected".

  N-7 (Medium): `preview_dataset` called `dataset.get(col, [])` which mimics
     dict semantics and is not part of the stable HuggingFace Dataset API across
     all versions. Fixed by checking `col in dataset.column_names` before
     accessing the column, which is the documented and version-stable approach.
"""

import pandas as pd
from datasets import Dataset

from config.constants import (
    COL_INSTRUCTION,
    COL_OUTPUT,
    COL_TEXT,
    COL_PROMPT,
    COL_CHOSEN,
    COL_REJECTED,
)


def validate_and_clean_dataset(
    dataset: Dataset,
    is_dpo: bool = False,
) -> tuple:
    """Validate and clean a Dataset in-place.

    Removes empty examples, deduplicates, and reports long ones (> 2048 chars).

    Returns
    -------
    (cleaned_dataset, issues)  where issues is a list[str] of warning messages.
    """
    issues = []

    # ── Compute lengths per row (strip whitespace for accurate empty detection) ──
    if is_dpo:
        lengths = [
            len(str(p).strip()) + len(str(c).strip()) + len(str(r).strip())
            for p, c, r in zip(
                dataset[COL_PROMPT],
                dataset[COL_CHOSEN],
                dataset[COL_REJECTED],
            )
        ]
    elif COL_TEXT in dataset.column_names:
        lengths = [len(str(t).strip()) for t in dataset[COL_TEXT]]
    elif COL_INSTRUCTION in dataset.column_names and COL_OUTPUT in dataset.column_names:
        lengths = [
            len(str(i).strip()) + len(str(o).strip())
            for i, o in zip(dataset[COL_INSTRUCTION], dataset[COL_OUTPUT])
        ]
    else:
        return dataset, ["⚠️ Unknown column structure — cannot validate."]

    # ── Report and remove empty / whitespace-only examples ────────────────
    empty = sum(1 for ln in lengths if ln == 0)
    if empty:
        issues.append(f"⚠️ {empty} empty examples removed. ")

    if is_dpo:
        dataset = dataset.filter(
            lambda x: (
                len(str(x[COL_PROMPT]).strip()) > 0
                and len(str(x[COL_CHOSEN]).strip()) > 0
                and len(str(x[COL_REJECTED]).strip()) > 0
            )
        )
    elif COL_TEXT in dataset.column_names:
        # BUG 3 FIX: strip so whitespace-only strings count as empty
        dataset = dataset.filter(lambda x: len(str(x[COL_TEXT]).strip()) > 0)
    else:
        # BUG 2 FIX: check each column independently with AND, not sum with OR
        dataset = dataset.filter(
            lambda x: (
                len(str(x[COL_INSTRUCTION]).strip()) > 0
                and len(str(x[COL_OUTPUT]).strip()) > 0
            )
        )

    # ── Duplicate detection AND removal (M4 FIX) ──────────────────────────
    # Previously only warned — now actually removes duplicates using an
    # ordered seen-set so original row order is preserved.
    if COL_TEXT in dataset.column_names:
        texts = [str(t) for t in dataset[COL_TEXT]]
        seen: set[str] = set()
        unique_indices: list[int] = []
        for idx, text in enumerate(texts):
            if text not in seen:
                seen.add(text)
                unique_indices.append(idx)
        n_dups = len(texts) - len(unique_indices)
        if n_dups > 0:
            issues.append(f"⚠️ {n_dups} duplicate examples removed. ")
            dataset = dataset.select(unique_indices)

    elif COL_INSTRUCTION in dataset.column_names and COL_OUTPUT in dataset.column_names:
        pairs = list(zip(
            [str(i) for i in dataset[COL_INSTRUCTION]],
            [str(o) for o in dataset[COL_OUTPUT]],
        ))
        seen_pairs: set[tuple] = set()
        unique_pair_indices: list[int] = []
        for idx, pair in enumerate(pairs):
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_pair_indices.append(idx)
        n_dups = len(pairs) - len(unique_pair_indices)
        if n_dups > 0:
            issues.append(f"⚠️ {n_dups} duplicate examples removed. ")
            dataset = dataset.select(unique_pair_indices)

    # ── Report long examples (will be truncated by tokeniser) ─────────────
    # Recompute lengths from the current (post-filter) dataset for accuracy.
    if is_dpo:
        current_lengths = [
            len(str(p).strip()) + len(str(c).strip()) + len(str(r).strip())
            for p, c, r in zip(
                dataset[COL_PROMPT],
                dataset[COL_CHOSEN],
                dataset[COL_REJECTED],
            )
        ]
    elif COL_TEXT in dataset.column_names:
        current_lengths = [len(str(t).strip()) for t in dataset[COL_TEXT]]
    else:
        current_lengths = [
            len(str(i).strip()) + len(str(o).strip())
            for i, o in zip(dataset[COL_INSTRUCTION], dataset[COL_OUTPUT])
        ]

    long_ = sum(1 for ln in current_lengths if ln > 2048)
    if long_:
        issues.append(f"⚠️ {long_} examples exceed 2048 chars — they will be truncated. ")

    if len(dataset) == 0:
        issues.append("❌ Dataset is empty after cleaning. No valid examples remain.")

    return dataset, issues


def preview_dataset(dataset: Dataset, is_dpo: bool = False) -> pd.DataFrame:
    """Return a small preview of the dataset as a pandas DataFrame for the UI.

    N-7 FIX: The previous implementation called `dataset.get(col, [])` which
    mimics dict.get() semantics.  That method is not part of the stable public
    HuggingFace Dataset API and behaves differently across library versions.
    Replaced with explicit `col in dataset.column_names` guards, which is the
    documented, version-stable way to check column existence before access.
    """
    if len(dataset) == 0:
        return pd.DataFrame({"Status": ["⚠️ Dataset is empty after cleaning."]})

    if is_dpo:
        return pd.DataFrame({
            COL_PROMPT:   dataset[COL_PROMPT][:5],
            COL_CHOSEN:   dataset[COL_CHOSEN][:5],
            COL_REJECTED: dataset[COL_REJECTED][:5],
        })
    elif COL_TEXT in dataset.column_names:
        return pd.DataFrame({COL_TEXT: dataset[COL_TEXT][:10]})
    else:
        # N-7 FIX: use explicit column_names check instead of dataset.get()
        inst_data = dataset[COL_INSTRUCTION][:5] if COL_INSTRUCTION in dataset.column_names else []
        out_data  = dataset[COL_OUTPUT][:5]      if COL_OUTPUT      in dataset.column_names else []
        return pd.DataFrame({
            COL_INSTRUCTION: inst_data,
            COL_OUTPUT:      out_data,
        })


def preprocess_function(
    examples,
    tokenizer,
    max_length: int,
    task_type: str,
    use_chat_template: bool,
    system_prompt: str,
) -> dict:
    """Tokenise a batch of examples for causal-LM training.

    When use_chat_template is True and the tokenizer has a chat_template,
    the standard ChatML format is applied. Otherwise falls back to the
    '### Instruction / ### Response' prompt format.

    Returns a dict with input_ids, attention_mask, and labels.
    """
    if use_chat_template and tokenizer.chat_template is not None:
        texts = []
        if task_type == COL_INSTRUCTION:
            for inst, out in zip(examples[COL_INSTRUCTION], examples[COL_OUTPUT]):
                messages = [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": inst},
                    {"role": "assistant", "content": out},
                ]
                texts.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                )
        else:
            for t in examples[COL_TEXT]:
                messages = [{"role": "user", "content": t}]
                texts.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                )
    else:
        if task_type == COL_INSTRUCTION:
            texts = [
                f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                for inst, out in zip(examples[COL_INSTRUCTION], examples[COL_OUTPUT])
            ]
        else:
            texts = examples[COL_TEXT]

    # BOLT OPTIMIZATION: Use padding=False (dynamic padding) instead of
    # padding="max_length". The DataCollator will pad batches to the longest
    # sequence in that batch, significantly reducing VRAM and increasing speed.
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def tokenize_reward_function(
    examples,
    tokenizer,
    rm_max_length: int,
) -> dict:
    """Tokenise a batch of examples for Reward Model training.

    Returns a dict with input_ids and attention_mask for both chosen and
    rejected responses.
    """
    # BOLT OPTIMIZATION: Use padding=False (dynamic padding) instead of
    # padding="max_length". The DataCollator will pad batches to the longest
    # sequence in that batch, significantly reducing VRAM and increasing speed.
    chosen_tok = tokenizer(
        examples[COL_CHOSEN],
        truncation=True,
        max_length=rm_max_length,
        padding=False,
        return_attention_mask=True,
    )
    rejected_tok = tokenizer(
        examples[COL_REJECTED],
        truncation=True,
        max_length=rm_max_length,
        padding=False,
        return_attention_mask=True,
    )
    return {
        "input_ids_chosen":        chosen_tok["input_ids"],
        "attention_mask_chosen":   chosen_tok["attention_mask"],
        "input_ids_rejected":      rejected_tok["input_ids"],
        "attention_mask_rejected": rejected_tok["attention_mask"],
    }
