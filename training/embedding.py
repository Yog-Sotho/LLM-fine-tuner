"""
training/embedding.py
=====================
Layer 3 — fine-tune an embedding model (sentence-transformers) for search and RAG.

Training pairs are (query, passage that answers it). Loss: MultipleNegativesRankingLoss —
every other passage in the batch is a negative, so only positive pairs are needed —
wrapped in MatryoshkaLoss, so the first 512 / 256 / … dimensions also work on their own
and embeddings can be truncated for a smaller index. Optional hard negatives are mined
with the base model (similar passages that do not answer the query). Retrieval quality
(NDCG@10, MRR@10, Recall@10) is measured on held-out pairs before and after training.

LoRA is merged into the model before saving: the result is a plain sentence-transformers
model that any tool loads (sentence-transformers, TEI, vLLM, LangChain, LlamaIndex).
Works with sentence-transformers 5.4+ (Transformers 4.56) and 6.x (Transformers 5).
"""

import gc
import math
import random
import time

import gradio as gr
import torch
from datasets import Dataset
from transformers import set_seed

from config.constants import (
    ALLOW_REMOTE_CODE,
    COL_ANCHOR,
    COL_COMPLETION,
    COL_CONTEXT,
    COL_INSTRUCTION,
    COL_MESSAGES,
    COL_NEGATIVE,
    COL_OUTPUT,
    COL_POSITIVE,
    COL_PROMPT,
    DEFAULT_EVAL_SPLIT,
    DEFAULT_REPORT_TO,
    DEFAULT_SEED,
    EMBED_LORA_ALPHA,
    EMBED_LORA_RANK,
    EMBED_MATRYOSHKA_DIMS,
    EMBED_MAX_SEQ_LENGTH,
    EMBED_METHODS,
    EMBED_MIN_EVAL_QUERIES,
    EMBED_NEGATIVE_MARGIN,
    HAS_SENTENCE_TRANSFORMERS,
)
from core.callbacks import ETAProgressCallback, LoggingCallback, StopCallback, final_train_loss
from core.hardware import (
    get_lora_targets,
    is_main_process,
    lora_dropout,
    sharding_unsupported,
    training_device_args,
)
from core.run_config import save_run_config
from core.state import app_state, redact_sensitive_info, validate_path_traversal
from data.loader import detect_file_type, load_dataset_from_file, load_table_dataset
from data.preprocessing import clean_messages

# Metrics shown before / after training (InformationRetrievalEvaluator keys, minus prefix).
EVAL_METRICS = ("cosine_ndcg@10", "cosine_mrr@10", "cosine_recall@10")


def _text(value) -> str:
    return "" if value is None else str(value).strip()


def to_embedding_pairs(ds: Dataset) -> Dataset:
    """(anchor, positive[, negative]) pairs from the formats the app reads.

    anchor/positive[/negative] are used as they are. Synthetic data from documents
    (instruction + context) pairs each question with the passage it was written from —
    the best positive for RAG. Otherwise the answer is the positive: instruction/output,
    prompt/completion, or the last user turn and assistant reply of a chat.
    """
    cols = set(ds.column_names)
    negatives: list[str] | None = None
    if {COL_ANCHOR, COL_POSITIVE} <= cols:
        anchors, positives = ds[COL_ANCHOR], ds[COL_POSITIVE]
        negatives = ds[COL_NEGATIVE] if COL_NEGATIVE in cols else None
    elif {COL_INSTRUCTION, COL_CONTEXT} <= cols:
        anchors, positives = ds[COL_INSTRUCTION], ds[COL_CONTEXT]
    elif {COL_INSTRUCTION, COL_OUTPUT} <= cols:
        anchors, positives = ds[COL_INSTRUCTION], ds[COL_OUTPUT]
    elif {COL_PROMPT, COL_COMPLETION} <= cols:
        anchors, positives = ds[COL_PROMPT], ds[COL_COMPLETION]
    elif COL_MESSAGES in cols:
        anchors, positives = [], []
        for chat in ds[COL_MESSAGES]:
            turns = [m for m in clean_messages(chat) or []  # None: unusable chat
                     if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]  # fmt: skip
            if len(turns) >= 2 and turns[-1]["role"] == "assistant" and turns[-2]["role"] == "user":
                anchors.append(turns[-2].get("content"))
                positives.append(turns[-1].get("content"))
    else:
        raise ValueError(
            "Embedding training needs anchor/positive (optional negative), instruction/context, "
            f"instruction/output, prompt/completion or chat (messages) data; found {sorted(cols)}"
        )
    rows: dict[str, list[str]] = {
        COL_ANCHOR: [],
        COL_POSITIVE: [],
        **({COL_NEGATIVE: []} if negatives else {}),
    }
    for i, (anchor, positive) in enumerate(zip(anchors, positives, strict=True)):
        anchor, positive = _text(anchor), _text(positive)
        negative = _text(negatives[i]) if negatives else None
        if not anchor or not positive or (negatives and not negative):
            continue  # a pair with an empty side teaches nothing
        rows[COL_ANCHOR].append(anchor)
        rows[COL_POSITIVE].append(positive)
        if negatives:
            rows[COL_NEGATIVE].append(negative)
    return Dataset.from_dict(rows)


def matryoshka_dims(dim: int) -> list[int]:
    """The model's own dimension plus every smaller standard one, largest first."""
    return sorted({dim, *(d for d in EMBED_MATRYOSHKA_DIMS if d < dim)}, reverse=True)


def split_pairs(pairs: Dataset, eval_split: float, seed: int) -> tuple[Dataset, Dataset, bool]:
    """(train, eval, held_out). Too few pairs to hold some out: evaluate on the train pairs."""
    n_eval = round(len(pairs) * float(eval_split))
    if n_eval < EMBED_MIN_EVAL_QUERIES or len(pairs) - n_eval < 2:
        return pairs, pairs, False
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    return pairs.select(order[n_eval:]), pairs.select(order[:n_eval]), True


def add_negatives(train_ds: Dataset, mined: Dataset, seed: int) -> tuple[Dataset, int]:
    """Every training pair with a negative: the mined one, else a random other passage.

    The miner drops pairs it finds no safe negative for; keeping them (with a random
    negative, no worse than the in-batch ones) means no training pair is lost.
    Returns (dataset, number of pairs with a mined negative).
    """
    anchor_col, positive_col, negative_col = mined.column_names[:3]
    found = dict(
        zip(
            zip(mined[anchor_col], mined[positive_col], strict=True),
            mined[negative_col],
            strict=True,
        )
    )
    passages = list(dict.fromkeys(train_ds[COL_POSITIVE]))
    rng = random.Random(seed)
    negatives, hits = [], 0
    for anchor, positive in zip(train_ds[COL_ANCHOR], train_ds[COL_POSITIVE], strict=True):
        if (anchor, positive) in found:
            negatives.append(found[(anchor, positive)])
            hits += 1
        else:
            negatives.append(rng.choice([p for p in passages if p != positive]))
    return train_ds.add_column(COL_NEGATIVE, negatives), hits


def retrieval_evaluator(eval_pairs: Dataset, corpus_texts: list[str], query_prompt: str):
    """InformationRetrievalEvaluator: find each held-out query's passage among all passages."""
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,  # lazy
    )

    corpus_ids: dict[str, str] = {}
    for text in corpus_texts:
        corpus_ids.setdefault(text, f"d{len(corpus_ids)}")
    query_ids: dict[str, str] = {}
    relevant: dict[str, set[str]] = {}
    for anchor, positive in zip(eval_pairs[COL_ANCHOR], eval_pairs[COL_POSITIVE], strict=True):
        qid = query_ids.setdefault(anchor, f"q{len(query_ids)}")
        relevant.setdefault(qid, set()).add(corpus_ids.setdefault(positive, f"d{len(corpus_ids)}"))
    return InformationRetrievalEvaluator(
        queries={qid: q for q, qid in query_ids.items()},
        corpus={cid: t for t, cid in corpus_ids.items()},
        relevant_docs=relevant,
        name="eval",
        query_prompt=query_prompt or None,
        show_progress_bar=False,
    )


def _scores(evaluator, model) -> dict[str, float]:
    results = evaluator(model)
    return {m: round(float(results.get(f"eval_{m}", 0.0)), 4) for m in EVAL_METRICS}


def format_scores(before: dict, after: dict) -> str:
    return "\n".join(
        f"   {m.split('_', 1)[1].upper():<10} {before[m]:.3f} → {after[m]:.3f}"
        f" ({after[m] - before[m]:+.3f})"
        for m in EVAL_METRICS
    )


def _load_pairs_file(file) -> Dataset:
    if detect_file_type(file) in ("json", "jsonl"):
        try:  # chats (messages) go through the main loader; other JSON keeps all columns
            return load_dataset_from_file(file, detect_file_type(file))
        except (ValueError, RuntimeError):
            pass
    return load_table_dataset(file)


def train_embedding(
    model_name: str,
    data_file,
    output_dir: str,
    method: str = EMBED_METHODS[0],
    learning_rate: float = 2e-5,
    epochs: int = 1,
    batch_size: int = 32,
    max_seq_length: int = EMBED_MAX_SEQ_LENGTH,
    matryoshka: bool = True,
    hard_negatives: bool = False,
    eval_split: float = DEFAULT_EVAL_SPLIT,
    query_prompt: str = "",
    progress=gr.Progress(),
    request: gr.Request | None = None,
) -> str:
    """Fine-tune the embedding model ``model_name`` on (query, passage) pairs."""
    model_name = (model_name or "").strip()
    output_dir = (output_dir or "").strip()
    if err := validate_path_traversal(model_name) or validate_path_traversal(output_dir):
        return err
    if err := sharding_unsupported("Embedding training"):
        return err
    if not HAS_SENTENCE_TRANSFORMERS:
        return '❌ sentence-transformers not installed. Install: pip install -e ".[embedding]"'
    if not model_name or not output_dir:
        return "❌ Give the embedding model and an output directory."
    if data_file is None:
        return "❌ Please upload training pairs (CSV / JSON / JSONL)."
    if method not in EMBED_METHODS:
        return f"❌ Unknown method '{method}'. Choose from: {', '.join(EMBED_METHODS)}"
    if int(batch_size) < 2:
        return "❌ Batch size must be at least 2: the other passages in a batch are the negatives."

    stop_event = app_state.session_for(request).stop_event
    stop_event.clear()
    set_seed(DEFAULT_SEED)  # before the model / LoRA are created, so runs are reproducible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from peft import LoraConfig, TaskType, get_peft_model  # lazy
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
        from sentence_transformers.sentence_transformer.losses import (
            MatryoshkaLoss,
            MultipleNegativesRankingLoss,
        )
        from sentence_transformers.sentence_transformer.training_args import (
            BatchSamplers,
            SentenceTransformerTrainingArguments,
        )

        if progress is not None:
            progress(0, desc="Loading training pairs…")
        pairs = to_embedding_pairs(_load_pairs_file(data_file))
        if len(pairs) < 2:
            return "❌ Need at least 2 usable (query, passage) pairs."
        train_ds, eval_ds, held_out = split_pairs(pairs, eval_split, DEFAULT_SEED)

        if progress is not None:
            progress(0.05, desc="Loading embedding model…")
        model = SentenceTransformer(model_name, device=device, trust_remote_code=ALLOW_REMOTE_CODE)
        model.max_seq_length = int(max_seq_length)
        # Kept verbatim: the trailing space of "query: " is part of the prompt.
        if not (query_prompt or "").strip():
            query_prompt = (model.prompts or {}).get("query", "")
        dim = model.get_embedding_dimension()

        mined = 0
        distinct_passages = len(set(train_ds[COL_POSITIVE]))
        if hard_negatives and COL_NEGATIVE not in train_ds.column_names and distinct_passages > 1:
            from sentence_transformers.util import mine_hard_negatives  # lazy

            if progress is not None:
                progress(0.1, desc="Mining hard negatives…")
            found = mine_hard_negatives(
                train_ds,
                model,
                num_negatives=1,
                relative_margin=EMBED_NEGATIVE_MARGIN,
                sampling_strategy="top",
                output_format="triplet",
                query_prompt=query_prompt or None,
                batch_size=int(batch_size),
                use_faiss=False,
                verbose=False,
            )
            train_ds, mined = add_negatives(train_ds, found, DEFAULT_SEED)

        corpus = list(dict.fromkeys(pairs[COL_POSITIVE]))  # every passage is a candidate
        evaluator = retrieval_evaluator(eval_ds, corpus, query_prompt)
        if progress is not None:
            progress(0.15, desc="Measuring retrieval before training…")
        before = _scores(evaluator, model)

        transformer = model[0]
        if method == "LoRA":  # wraps the inner Hugging Face model; merged again before saving
            transformer.model = get_peft_model(
                transformer.model,
                LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=EMBED_LORA_RANK,
                    lora_alpha=EMBED_LORA_ALPHA,
                    target_modules=get_lora_targets(),
                    lora_dropout=lora_dropout(transformer.model),
                ),
            )
        loss = MultipleNegativesRankingLoss(model)
        dims = matryoshka_dims(dim) if matryoshka else [dim]
        if matryoshka and len(dims) > 1:
            loss = MatryoshkaLoss(model, loss, matryoshka_dims=dims)

        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=int(epochs),
            per_device_train_batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            # 10 % warm-up, as steps: warmup_ratio is deprecated in Transformers 5.
            warmup_steps=round(0.1 * int(epochs) * math.ceil(len(train_ds) / int(batch_size))),
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # a duplicate would be a false negative
            prompts={COL_ANCHOR: query_prompt} if query_prompt else None,
            logging_steps=1,
            save_strategy="no",  # short runs; the final model is saved below
            report_to=DEFAULT_REPORT_TO,
            seed=DEFAULT_SEED,
            **training_device_args(device),
        )
        log_cb = LoggingCallback()
        callbacks = [StopCallback(stop_event), log_cb]
        if progress is not None:
            callbacks.append(
                ETAProgressCallback(gradio_progress=progress, progress_start=0.2, progress_end=0.9)
            )
        trainer = SentenceTransformerTrainer(
            model=model, args=args, train_dataset=train_ds, loss=loss, callbacks=callbacks
        )
        if progress is not None:
            progress(0.2, desc="Embedding training started… calculating ETA…")
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0
        status = "stopped by user" if stop_event.is_set() else "complete"

        if method == "LoRA":
            transformer.model = transformer.model.merge_and_unload()
        if query_prompt:  # saved with the model: encode(..., prompt_name="query")
            model.prompts = {**(model.prompts or {}), "query": query_prompt}
        if progress is not None:
            progress(0.92, desc="Measuring retrieval after training…")
        after = _scores(evaluator, model)

        if progress is not None:
            progress(0.96, desc="Saving model…")
        if is_main_process():
            model.save_pretrained(output_dir)
            save_run_config(
                output_dir,
                mode="embedding",
                model=model_name,
                dataset=pairs,
                seed=DEFAULT_SEED,
                report_to=DEFAULT_REPORT_TO,
                hyperparams={
                    "learning_rate": float(learning_rate),
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "max_seq_length": int(max_seq_length),
                    "loss": "MatryoshkaLoss(MNRL)" if len(dims) > 1 else "MNRL",
                    "matryoshka_dims": dims,
                    "hard_negatives": bool(hard_negatives),
                    "query_prompt": query_prompt,
                },
                peft={"method": method}
                | (
                    {"lora_rank": EMBED_LORA_RANK, "lora_alpha": EMBED_LORA_ALPHA}
                    if method == "LoRA"
                    else {}
                ),  # fmt: skip
                evaluation={
                    "split": "held-out pairs" if held_out else "training pairs",
                    "queries": len(set(eval_ds[COL_ANCHOR])),
                    "before": before,
                    "after": after,
                },
            )

        if progress is not None:
            progress(1.0, desc="✅ Complete!")
        negatives_line = (
            f"\n🧲 Hard negatives: mined for {mined} of {len(train_ds)} pairs"
            + (" (random passages for the rest)" if mined < len(train_ds) else "")
            if hard_negatives and COL_NEGATIVE in train_ds.column_names
            and COL_NEGATIVE not in pairs.column_names else ""
        )  # fmt: skip
        split_note = "held-out" if held_out else "training (too few pairs to hold some out)"
        return (
            f"✅ Embedding training {status}!\n"
            f"🔎 Model: {model_name} ({method}, {dim} dimensions"
            + (f", Matryoshka {dims}" if len(dims) > 1 else "")
            + f")\n📊 Pairs: {len(train_ds)} train, {len(eval_ds)} evaluation{negatives_line}\n"
            f"⏱ Elapsed: {elapsed / 60:.1f} min\n"
            f"📉 Final train loss: {final_train_loss(log_cb.records)}\n"
            f"📈 Retrieval on {split_note} queries (before → after):\n"
            f"{format_scores(before, after)}\n"
            f"📁 Model saved to: {output_dir}"
        )

    except Exception as e:
        return f"❌ Embedding training failed: {redact_sensitive_info(str(e))}"
    finally:
        try:
            del trainer
        except (NameError, UnboundLocalError):
            pass
        try:
            del model
        except (NameError, UnboundLocalError):
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
