"""Embedding-model fine-tuning (training/embedding.py): pairs, splits, negatives, evaluation,
and real training of a tiny sentence-transformers model through the trainer, UI and CLI."""

import json
import os

import pytest
import yaml
from datasets import Dataset
from typer.testing import CliRunner

import cli.commands as commands
import core.run_config as rc
import training.embedding as emb
import ui.handlers as handlers
from config.constants import EMBED_METHODS
from core.model_card import build_model_card

TINY_EMBEDDER = "sentence-transformers-testing/stsb-bert-tiny-safetensors"  # 128 dims


@pytest.fixture(scope="module")
def tiny_embedder() -> str:
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer

    try:
        SentenceTransformer(TINY_EMBEDDER, device="cpu")
    except OSError as exc:
        if os.environ.get("REQUIRE_SMOKE_MODELS") == "1":
            pytest.fail(f"Embedding smoke-test model unavailable: {exc}")
        pytest.skip(f"Embedding smoke-test model unavailable (offline?): {exc}")
    return TINY_EMBEDDER


class _File:
    def __init__(self, path):
        self.name = str(path)


def _pairs_file(tmp_path, n=40, name="pairs.jsonl"):
    """Synthetic-data rows (Tier 16 output): question, answer, source passage."""
    rows = [
        {"instruction": f"Which warehouse gets box {i}?", "output": f"Warehouse {i % 7}.",
         "context": f"Rule {i}: box {i} is labelled red and shipped to warehouse {i % 7}.",
         "source": "handbook.txt", "score": 9}
        for i in range(n)
    ]  # fmt: skip
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


# ── Pairs from every supported format ──────────────────────────────────────


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ({"anchor": ["q"], "positive": ["p"]}, ("q", "p")),
        ({"instruction": ["q"], "output": ["a"], "context": ["passage"]}, ("q", "passage")),
        ({"instruction": ["q"], "output": ["a"]}, ("q", "a")),
        ({"prompt": ["q"], "completion": ["a"]}, ("q", "a")),
    ],
)
def test_pairs_from_each_format(columns, expected):
    pairs = emb.to_embedding_pairs(Dataset.from_dict(columns))
    assert (pairs["anchor"][0], pairs["positive"][0]) == expected
    assert pairs.column_names == ["anchor", "positive"]


def test_pairs_from_chats_and_negatives():
    from data.preprocessing import chat_dataset

    chats = chat_dataset([{"messages": m} for m in [
        [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"},
         {"role": "assistant", "content": "a1"}],
        [{"role": "user", "content": "only a question"}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what?"}]},
         {"role": "assistant", "content": "a cat"}],
    ]])  # fmt: skip
    pairs = emb.to_embedding_pairs(chats)
    assert list(zip(pairs["anchor"], pairs["positive"], strict=True)) == [("q1", "a1")]
    triplets = emb.to_embedding_pairs(Dataset.from_dict(
        {"anchor": ["q", "q2", " "], "positive": ["p", "p2", "p3"], "negative": ["n", "", "n3"]}
    ))  # fmt: skip
    assert triplets.to_dict() == {"anchor": ["q"], "positive": ["p"], "negative": ["n"]}


def test_pairs_need_known_columns():
    with pytest.raises(ValueError, match="anchor/positive"):
        emb.to_embedding_pairs(Dataset.from_dict({"text": ["x"]}))


# ── Matryoshka dimensions, splits, negatives ──────────────────────────────


@pytest.mark.parametrize(
    ("dim", "dims"),
    [(1024, [1024, 768, 512, 256, 128, 64]), (384, [384, 256, 128, 64]), (64, [64]), (48, [48])],
)
def test_matryoshka_dims(dim, dims):
    assert emb.matryoshka_dims(dim) == dims


def test_split_holds_out_only_with_enough_pairs():
    pairs = Dataset.from_dict({"anchor": [f"q{i}" for i in range(40)],
                               "positive": [f"p{i}" for i in range(40)]})  # fmt: skip
    train, evals, held_out = emb.split_pairs(pairs, 0.25, seed=1)
    assert held_out and len(train) == 30 and len(evals) == 10
    assert not set(train["anchor"]) & set(evals["anchor"])
    assert emb.split_pairs(pairs, 0.25, seed=1)[1]["anchor"] == evals["anchor"]  # reproducible
    small_train, small_eval, small_held = emb.split_pairs(pairs.select(range(10)), 0.1, seed=1)
    assert not small_held and len(small_train) == len(small_eval) == 10
    assert not emb.split_pairs(pairs, 0.0, seed=1)[2]


def test_add_negatives_keeps_every_pair():
    train = Dataset.from_dict({"anchor": ["a", "b", "c"], "positive": ["pa", "pb", "pc"]})
    mined = Dataset.from_dict({"anchor": ["b"], "positive": ["pb"], "negative": ["pc"]})
    with_negatives, hits = emb.add_negatives(train, mined, seed=1)
    assert hits == 1 and len(with_negatives) == 3
    negatives = dict(zip(with_negatives["anchor"], with_negatives["negative"], strict=True))
    assert negatives["b"] == "pc"
    assert all(negatives[a] != f"p{a}" for a in "abc")  # never the pair's own passage


def test_evaluator_merges_duplicate_queries_and_passages():
    pytest.importorskip("sentence_transformers")
    evals = Dataset.from_dict({"anchor": ["q", "q", "r"], "positive": ["p1", "p2", "p1"]})
    evaluator = emb.retrieval_evaluator(evals, ["p1", "p2", "p3"], "query: ")
    assert len(evaluator.queries) == 2 and len(evaluator.corpus) == 3
    ids = {text: cid for cid, text in zip(evaluator.corpus_ids, evaluator.corpus, strict=True)}
    qid = {text: q for q, text in zip(evaluator.queries_ids, evaluator.queries, strict=True)}
    assert evaluator.relevant_docs[qid["q"]] == {ids["p1"], ids["p2"]}
    assert evaluator.query_prompt == "query: "


def test_score_lines():
    before = dict.fromkeys(emb.EVAL_METRICS, 0.25)
    after = dict.fromkeys(emb.EVAL_METRICS, 0.5)
    lines = emb.format_scores(before, after).splitlines()
    assert lines[0].split() == ["NDCG@10", "0.250", "→", "0.500", "(+0.250)"]
    assert len(lines) == 3


# ── Input checks ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_name": "../m"}, "Path traversal"),
        ({"data_file": None}, "Please upload training pairs"),
        ({"method": "QLoRA"}, "Unknown method"),
        ({"batch_size": 1}, "at least 2"),
        ({"model_name": " "}, "Give the embedding model"),
    ],
)
def test_input_checks(tmp_path, kwargs, message):
    args = {"model_name": "m", "data_file": _File(_pairs_file(tmp_path)),
            "output_dir": str(tmp_path / "out"), "progress": None, **kwargs}  # fmt: skip
    assert message in emb.train_embedding(**args)


def test_needs_sentence_transformers(tmp_path, monkeypatch):
    monkeypatch.setattr(emb, "HAS_SENTENCE_TRANSFORMERS", False)
    status = emb.train_embedding("m", _File(_pairs_file(tmp_path)), str(tmp_path), progress=None)
    assert "sentence-transformers not installed" in status


def test_too_few_pairs_and_bad_data(tmp_path, tiny_embedder):
    one = tmp_path / "one.csv"
    one.write_text("anchor,positive\nq,p\n")
    assert "at least 2" in emb.train_embedding(tiny_embedder, _File(one), str(tmp_path / "o"),
                                               progress=None)  # fmt: skip
    text = tmp_path / "t.csv"
    text.write_text("text\nhello\n")
    status = emb.train_embedding(tiny_embedder, _File(text), str(tmp_path / "o2"), progress=None)
    assert status.startswith("❌ Embedding training failed") and "anchor/positive" in status


# ── Real training (tiny model) ─────────────────────────────────────────────


def _train(tmp_path, model, method, **kwargs):
    out = tmp_path / f"out_{method[:4]}"
    status = emb.train_embedding(
        model, _File(_pairs_file(tmp_path)), str(out), method=method,
        learning_rate=1e-3 if method == "LoRA" else 1e-4, epochs=2, batch_size=8,
        max_seq_length=64, eval_split=0.25, progress=None, **kwargs,
    )  # fmt: skip
    return status, out


def test_full_fine_tuning_improves_retrieval(tmp_path, tiny_embedder):
    from sentence_transformers import SentenceTransformer

    status, out = _train(tmp_path, tiny_embedder, EMBED_METHODS[0], query_prompt="query: ")
    assert status.startswith("✅ Embedding training complete!"), status
    assert "Pairs: 30 train, 10 evaluation" in status and "Matryoshka [128, 64]" in status
    record = yaml.safe_load((out / "run_config.yaml").read_text())
    assert record["mode"] == "embedding" and record["evaluation"]["split"] == "held-out pairs"
    before, after = record["evaluation"]["before"], record["evaluation"]["after"]
    assert after["cosine_ndcg@10"] > before["cosine_ndcg@10"]
    assert record["hyperparams"]["query_prompt"] == "query: "
    assert "sentence-transformers" in record["libraries"]
    card = (out / "README.md").read_text()
    assert "library_name: sentence-transformers" in card and "## Retrieval evaluation" in card
    model = SentenceTransformer(str(out), device="cpu")
    assert model.prompts.get("query") == "query: "  # saved with the model
    assert model.encode(["hi"], prompt_name="query").shape == (1, 128)


def test_lora_is_merged_and_hard_negatives_keep_every_pair(tmp_path, tiny_embedder):
    from sentence_transformers import SentenceTransformer

    status, out = _train(tmp_path, tiny_embedder, "LoRA", hard_negatives=True, matryoshka=False)
    assert status.startswith("✅"), status
    assert "Pairs: 30 train" in status and "Hard negatives: mined for" in status
    assert "Matryoshka" not in status
    assert not (out / "adapter_config.json").exists()  # merged: a plain model
    model = SentenceTransformer(str(out), device="cpu")
    assert not any("lora" in name for name, _ in model.named_parameters())
    record = yaml.safe_load((out / "run_config.yaml").read_text())
    assert record["peft"] == {"method": "LoRA", "lora_rank": 16, "lora_alpha": 32}
    assert record["hyperparams"]["loss"] == "MNRL"


def test_stop_ends_the_run_early(tmp_path, tiny_embedder, monkeypatch):
    from core.callbacks import StopCallback

    monkeypatch.setattr(StopCallback, "on_step_end",
                        lambda self, args, state, control, **kw: setattr(
                            control, "should_training_stop", True) or self._stop_event.set())  # fmt: skip
    status, out = _train(tmp_path, tiny_embedder, EMBED_METHODS[0])
    assert status.startswith("✅ Embedding training stopped by user!"), status
    assert (out / "model.safetensors").exists()


# ── UI handler and CLI ─────────────────────────────────────────────────────


@pytest.fixture
def runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(rc, "RUNS_DIR", str(runs))
    return runs


def test_ui_trains_into_the_runs_folder(tmp_path, runs, tiny_embedder):
    status = handlers.on_embed_click(
        tiny_embedder, _File(_pairs_file(tmp_path)), "docs-search", "Full fine-tuning", 1e-4,
        1, 8, 64, True, False, 0.25, "", None,
    )  # fmt: skip
    assert status.startswith("✅"), status
    assert (runs / "docs-search" / "model.safetensors").exists()
    again = handlers.on_embed_click(tiny_embedder, None, "docs-search", "LoRA", 1e-4, 1, 8, 64,
                                    True, False, 0.1, "", None)  # fmt: skip
    assert "already exists" in again


def test_ui_run_names_are_checked_and_defaulted(runs, monkeypatch):
    assert handlers.on_embed_click("m", None, "../x", "LoRA", 1e-4, 1, 8, 64, True, False, 0.1,
                                   "", None).startswith("❌")  # fmt: skip
    seen = {}
    monkeypatch.setattr(handlers, "train_embedding",
                        lambda model, file, out, *a, **k: seen.update(out=out) or "✅")  # fmt: skip
    handlers.on_embed_click("m", None, "", "LoRA", 1e-4, 1, 8, 64, True, False, 0.1, "", None)
    assert seen["out"].startswith(str(runs)) and seen["out"].endswith("-embedding")


def test_cli_trains_on_synthesize_output(tmp_path, tiny_embedder):
    out = tmp_path / "cli_model"
    result = CliRunner().invoke(commands.app, [
        "embed", "--model", tiny_embedder, "--data", str(_pairs_file(tmp_path)),
        "--output", str(out), "--lora", "--lr", "1e-3", "--batch-size", "8",
        "--max-seq-length", "64", "--no-matryoshka", "--eval-split", "0.25",
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "LoRA" in result.output and "NDCG@10" in result.output
    assert (out / "model.safetensors").exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--data", "missing.jsonl"], "Dataset not found"),
        (["--data", "../x.jsonl"], "Path traversal"),
    ],
)
def test_cli_input_checks(args, message):
    result = CliRunner().invoke(commands.app, ["embed", *args])
    assert result.exit_code == 1 and message in result.stderr


def test_cli_needs_sentence_transformers(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "HAS_SENTENCE_TRANSFORMERS", False)
    result = CliRunner().invoke(commands.app, ["embed", "--data", str(_pairs_file(tmp_path))])
    assert result.exit_code == 1 and "not installed" in result.stderr


def test_cli_reports_training_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(commands, "train_embedding", lambda **kw: "❌ Embedding training failed: x")
    result = CliRunner().invoke(commands.app, ["embed", "--data", str(_pairs_file(tmp_path))])
    assert result.exit_code == 1 and "failed: x" in result.stderr


# ── Model card ─────────────────────────────────────────────────────────────


def test_model_card_for_an_embedding_model():
    record = {
        "mode": "embedding", "model": "BAAI/bge-small-en-v1.5", "seed": 42,
        "hyperparams": {"matryoshka_dims": [384, 256], "query_prompt": "query: "},
        "peft": {"method": "Full fine-tuning"},
        "evaluation": {"split": "held-out pairs", "queries": 12,
                       "before": {"cosine_ndcg@10": 0.4}, "after": {"cosine_ndcg@10": 0.6}},
    }  # fmt: skip
    card = str(build_model_card(record, is_adapter=False))
    assert "library_name: sentence-transformers" in card
    assert "pipeline_tag: sentence-similarity" in card and "- trl" not in card
    assert "truncate_dim=256" in card and 'prompt_name="query"' in card
    assert "| ndcg@10 | 0.4 | 0.6 |" in card and "12 queries (held-out pairs)" in card
    plain = str(build_model_card({**record, "hyperparams": {}, "evaluation": None}, False))
    assert "truncate_dim" not in plain and "prompt_name" not in plain
    assert "Retrieval evaluation" not in plain
