"""
inference/synthesize.py
=======================
Layer 4 — turn document chunks into question / answer training pairs with an LLM, then
(optionally) let the LLM rate each pair and keep the good ones.

The writer is either any OpenAI-compatible server (vLLM, llama.cpp's llama-server, a
hosted API — through inference/remote.py) or a local Hugging Face model. Same recipe as
Meta's synthetic-data-kit: ask for N pairs "based ONLY on this text" as JSON, rate each
pair 1–10, drop those under the threshold, drop duplicate questions.
"""

import json
import logging
import re
from collections.abc import Callable

import torch

from config.constants import (
    COL_CONTEXT,
    COL_INSTRUCTION,
    COL_OUTPUT,
    SYNTH_CURATE_THRESHOLD,
    SYNTH_MAX_NEW_TOKENS,
    SYNTH_PAIRS_PER_CHUNK,
)
from core.state import redact_sensitive_info
from inference.evaluation import parse_judge_score
from inference.generate import _load_for_inference
from inference.remote import remote_chat

logger = logging.getLogger(__name__)

Ask = Callable[[str], str]  # prompt → model reply

QA_PROMPT = """Create {n} high-quality question-answer pairs based ONLY on the text below.

Rules:
- Each question must be answerable from the text alone, and make sense without it
  (no "according to the text", no "in this passage").
- Answers must be correct, complete and self-contained, in the language of the text.
- Cover different facts; no near-duplicate questions.

Reply with JSON only, in this format:
[{{"question": "...", "answer": "..."}}]

Text:
\"\"\"
{text}
\"\"\"
"""

RATING_PROMPT = """Rate this question-answer pair as training data for an assistant, from 1
(useless or wrong) to 10 (excellent): is the answer correct, complete and clearly written,
and is the question clear on its own?

Question: {question}
Answer: {answer}

Reply with one line: Score: N"""


def parse_qa_pairs(reply: str) -> list[dict[str, str]]:
    """Question/answer pairs from a model reply (JSON list, possibly in a code fence)."""
    text = re.sub(r"```(?:json)?", "", reply or "")
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    pairs = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        question, answer = item.get("question"), item.get("answer")
        if (
            isinstance(question, str)
            and isinstance(answer, str)
            and question.strip()
            and answer.strip()
        ):
            pairs.append({"question": question.strip(), "answer": answer.strip()})
    return pairs


def _question_key(question: str) -> str:
    return re.sub(r"\W+", " ", question.lower()).strip()


def remote_writer(url: str, model: str = "", api_key: str = "", temperature: float = 0.7,
                  max_tokens: int = SYNTH_MAX_NEW_TOKENS) -> Ask:  # fmt: skip
    """Ask function for an OpenAI-compatible server."""
    return lambda prompt: remote_chat(url, prompt, model, api_key, "", max_tokens, temperature)


def local_writer(model_name: str, temperature: float = 0.7,
                 max_new_tokens: int = SYNTH_MAX_NEW_TOKENS) -> Ask:  # fmt: skip
    """Ask function for a local model (chat template applied when it has one)."""

    def ask(prompt: str) -> str:
        model, tokenizer = _load_for_inference(model_name, None)
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=True,
                                 temperature=float(temperature), top_p=0.95,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)  # fmt: skip
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    return ask


def synthesize_pairs(
    chunks: list[tuple[str, str]],
    ask: Ask,
    pairs_per_chunk: int = SYNTH_PAIRS_PER_CHUNK,
    curate_threshold: float = SYNTH_CURATE_THRESHOLD,
    on_progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[dict], dict]:
    """Write and (if ``curate_threshold`` > 0) rate pairs for every (source, chunk).

    Returns (rows with instruction / output / context / source / score, stats); context is
    the passage the pair was written from (embedding training pairs question ↔ context).
    Failed chunks are counted and skipped, so one bad reply doesn't lose the run.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    stats = {"chunks": len(chunks), "generated": 0, "duplicates": 0, "rejected": 0,
             "failed_chunks": 0, "unrated": 0}  # fmt: skip
    for i, (source, chunk) in enumerate(chunks):
        if should_stop is not None and should_stop():
            stats["stopped"] = True
            break
        if on_progress is not None:
            on_progress(i, len(chunks))
        try:
            pairs = parse_qa_pairs(ask(QA_PROMPT.format(n=int(pairs_per_chunk), text=chunk)))
        except Exception as e:  # server down, timeout, … — keep going
            logger.warning(
                "Chunk %d of %s failed: %s", i + 1, source, redact_sensitive_info(str(e))
            )
            pairs = []
        if not pairs:
            stats["failed_chunks"] += 1
            continue
        stats["generated"] += len(pairs)
        for pair in pairs:
            key = _question_key(pair["question"])
            if key in seen:
                stats["duplicates"] += 1
                continue
            score = None
            if curate_threshold > 0:
                try:
                    score = parse_judge_score(ask(RATING_PROMPT.format(**pair)))
                except Exception as e:
                    logger.warning("Rating failed: %s", redact_sensitive_info(str(e)))
                if score is None:
                    stats["unrated"] += 1  # can't judge it: keep it out
                    continue
                if score < curate_threshold:
                    stats["rejected"] += 1
                    continue
            seen.add(key)
            rows.append({COL_INSTRUCTION: pair["question"], COL_OUTPUT: pair["answer"],
                         COL_CONTEXT: chunk, "source": source, "score": score})  # fmt: skip
    stats["kept"] = len(rows)
    return rows, stats


def format_stats(stats: dict) -> str:
    curated = stats["rejected"] or stats["unrated"]
    text = (
        f"✅ {stats['kept']} question/answer pairs from {stats['chunks']} text chunks"
        f" ({stats['generated']} written"
        + (f", {stats['rejected']} rated too low, {stats['unrated']} unrated" if curated else "")
        + (f", {stats['duplicates']} duplicates" if stats["duplicates"] else "")
        + ")"
    )
    if stats["failed_chunks"]:
        text += f"\n⚠️ {stats['failed_chunks']} chunks gave no usable pairs (no valid JSON reply)."
    if stats.get("stopped"):
        text += "\n🛑 Stopped early."
    if not stats["kept"]:
        text = "❌" + text[1:]
    return text
