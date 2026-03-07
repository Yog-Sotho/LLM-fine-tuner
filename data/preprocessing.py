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

    Removes empty examples and reports long ones (> 2048 chars).

    Returns
    -------
    (cleaned_dataset, issues)  where issues is a list[str] of warning messages.
    """
    issues = []

    # ── Compute lengths per row ────────────────────────────────────────────
    if is_dpo:
        lengths = [
            len(str(p)) + len(str(c)) + len(str(r))
            for p, c, r in zip(
                dataset[COL_PROMPT],
                dataset[COL_CHOSEN],
                dataset[COL_REJECTED],
            )
        ]
    elif COL_TEXT in dataset.column_names:
        lengths = [len(str(t)) for t in dataset[COL_TEXT]]
    elif COL_INSTRUCTION in dataset.column_names and COL_OUTPUT in dataset.column_names:
        lengths = [
            len(str(i)) + len(str(o))
            for i, o in zip(dataset[COL_INSTRUCTION], dataset[COL_OUTPUT])
        ]
    else:
        return dataset, ["⚠️ Unknown column structure — cannot validate."]

    # ── Report and remove empty examples ──────────────────────────────────
    empty = sum(1 for ln in lengths if ln == 0)
    if empty:
        issues.append(f"⚠️ {empty} empty examples removed. ")

    if is_dpo:
        dataset = dataset.filter(
            lambda x: (
                len(str(x[COL_PROMPT])) > 0
                and len(str(x[COL_CHOSEN])) > 0
                and len(str(x[COL_REJECTED])) > 0
            )
        )
    elif COL_TEXT in dataset.column_names:
        dataset = dataset.filter(lambda x: len(str(x[COL_TEXT])) > 0)
    else:
        dataset = dataset.filter(
            lambda x: len(str(x[COL_INSTRUCTION])) + len(str(x[COL_OUTPUT])) > 0
        )

    # ── Report long examples (will be truncated by tokeniser) ─────────────
    long_ = sum(1 for ln in lengths if ln > 2048)
    if long_:
        issues.append(f"⚠️ {long_} examples exceed 2048 chars — they will be truncated. ")

    if len(dataset) == 0:
        issues.append("❌ Dataset is empty after cleaning. No valid examples remain.")

    return dataset, issues


def preview_dataset(dataset: Dataset, is_dpo: bool = False) -> pd.DataFrame:
    """Return a small preview of the dataset as a pandas DataFrame for the UI."""
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
        return pd.DataFrame({
            COL_INSTRUCTION: dataset.get(COL_INSTRUCTION, [])[:5],
            COL_OUTPUT:      dataset.get(COL_OUTPUT, [])[:5],
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

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
