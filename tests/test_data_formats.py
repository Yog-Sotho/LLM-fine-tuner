"""Unit tests (no downloads) for chat data, near-duplicates, Hub loading and token reports."""

import pytest
from datasets import Dataset

import data.loader as loader
from data.preprocessing import (
    clean_messages,
    drop_prompts_over_limit,
    format_token_report,
    get_dataset_stats,
    preview_dataset,
    to_sft_dataset,
    token_length_report,
    validate_and_clean_dataset,
)

CONV = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": " Hi "},
    {"role": "assistant", "content": "Hello!"},
]


# ── Chat conversations ─────────────────────────────────────────────────────


def test_clean_messages_strips_and_keeps_valid_conversation():
    assert clean_messages(CONV) == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_clean_messages_cuts_after_last_answered_assistant_turn():
    conv = [*CONV, {"role": "user", "content": "Bye"}, {"role": "assistant", "content": " "}]
    assert clean_messages(conv)[-1] == {"role": "assistant", "content": "Hello!"}


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "text",
        [{"role": "user", "content": "only a question"}],
        [{"role": "assistant", "content": "answer without question"}],
        [{"role": "user", "content": "hi"}, {"role": "robot", "content": "x"}],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": 5}],
        ["not a dict"],
    ],
)
def test_clean_messages_rejects_untrainable(bad):
    assert clean_messages(bad) is None


def test_validate_messages_drops_invalid_and_duplicates_and_extra_columns():
    ds = Dataset.from_dict(
        {
            "messages": [
                CONV,
                [{"role": "system", "content": "Be brief."}, *[dict(t) for t in CONV[1:]]],
                [{"role": "user", "content": "q"}],
                [{"role": "user", "content": "HI"}, {"role": "assistant", "content": "hello!"}],
            ],
            "source": ["a", "b", "c", "d"],
        }
    )
    out, issues = validate_and_clean_dataset(ds)
    assert out.column_names == ["messages"]
    assert len(out) == 2  # exact duplicate and the unanswered conversation removed
    assert any("invalid or empty conversations" in i for i in issues)
    assert any("duplicate" in i for i in issues)


def test_messages_stats_and_preview():
    ds, _ = validate_and_clean_dataset(Dataset.from_dict({"messages": [CONV]}))
    assert get_dataset_stats(ds)["avg_length"] == len("Be brief.Hi" + "Hello!")
    assert preview_dataset(ds)["messages"][0].startswith("system: Be brief.")


def test_messages_become_prompt_and_final_answer():
    ds, _ = validate_and_clean_dataset(Dataset.from_dict({"messages": [CONV]}))
    out = to_sft_dataset(ds, use_chat_template=False, system_prompt="ignored")
    assert out[0]["prompt"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    assert out[0]["completion"] == [{"role": "assistant", "content": "Hello!"}]


# ── Near-duplicates ────────────────────────────────────────────────────────


def test_case_and_whitespace_variants_are_duplicates():
    ds = Dataset.from_dict(
        {"instruction": ["Say  hi", "say hi", "Say bye"], "output": ["Hi", "HI ", "Bye"]}
    )
    out, issues = validate_and_clean_dataset(ds)
    assert out["instruction"] == ["Say  hi", "Say bye"]  # first occurrence kept as-is
    assert "⚠️ 1 duplicate examples removed" in "".join(issues)


def test_near_duplicate_dpo_rows():
    ds = Dataset.from_dict(
        {"prompt": ["Q", "q"], "chosen": ["Good", "good"], "rejected": ["Bad", "bad "]}
    )
    out, _ = validate_and_clean_dataset(ds, is_dpo=True)
    assert len(out) == 1


# ── Token report ───────────────────────────────────────────────────────────


class _CharTokenizer:
    """One token per character; chat template joins contents."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(text)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "|".join(m["content"] for m in messages) + ("|" if add_generation_prompt else "")


def test_token_report_counts_and_flags_truncation():
    ds = Dataset.from_dict({"text": ["abcd", "ab", "abcdefghij"]})
    report = token_length_report(ds, _CharTokenizer(), max_length=5, sample=100)
    assert report["max"] == 10 and report["sampled"] == 3
    assert report["over_max_length"] == 1 and report["over_pct"] == 33.3
    assert "exceed Max Sequence Length 5" in format_token_report(report)


def test_token_report_formats():
    tok = _CharTokenizer()
    chat = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "ab"}]],
            "completion": [[{"role": "assistant", "content": "cd"}]],
        }
    )
    assert token_length_report(chat, tok, 100, 10)["max"] == len("ab|cd")
    dpo = Dataset.from_dict({"prompt": ["p"], "chosen": ["long one"], "rejected": ["x"]})
    assert token_length_report(dpo, tok, 100, 10)["max"] == len("plong one")
    assert "exceed" not in format_token_report(token_length_report(dpo, tok, 100, 10))


def test_token_report_samples():
    ds = Dataset.from_dict({"text": ["a"] * 50})
    assert token_length_report(ds, _CharTokenizer(), 10, sample=7)["sampled"] == 7


# ── Hub loading (load_dataset replaced by an in-memory stream) ──────────────


@pytest.fixture
def fake_hub(monkeypatch):
    calls = {}

    def fake_load_dataset(repo, config, split, streaming):
        calls.update(repo=repo, config=config, split=split, streaming=streaming)
        return calls["rows"].to_iterable_dataset()

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return calls


def test_hub_chat_dataset_keeps_messages_and_streams_max_rows(fake_hub):
    fake_hub["rows"] = Dataset.from_dict({"messages": [CONV] * 5, "source": ["x"] * 5})
    ds = loader.load_hub_dataset("owner/name", split="train", max_rows=3)
    assert ds.column_names == ["messages"] and len(ds) == 3
    assert (fake_hub["repo"], fake_hub["config"], fake_hub["streaming"]) == (
        "owner/name",
        None,
        True,
    )


def test_hub_dpo_requires_plain_text_columns(fake_hub):
    fake_hub["rows"] = Dataset.from_dict(
        {"prompt": ["p"], "chosen": [[{"role": "assistant", "content": "a"}]], "rejected": [[]]}
    )
    with pytest.raises(ValueError, match="stores chats"):
        loader.load_hub_dataset("owner/prefs", is_dpo=True)


def test_hub_unsupported_layout(fake_hub):
    fake_hub["rows"] = Dataset.from_dict({"question": ["q"], "answer": ["a"]})
    with pytest.raises(ValueError, match="Unsupported dataset layout"):
        loader.load_hub_dataset("owner/qa")


@pytest.mark.parametrize(
    ("repo", "kwargs", "message"),
    [
        ("no-slash", {}, "owner/name"),
        ("../etc/passwd", {}, "owner/name"),
        ("owner/..", {}, "owner/name"),
        ("owner/name", {"split": "train[:10]"}, "Split and config"),
        ("owner/name", {"config": "a b"}, "Split and config"),
        ("owner/name", {"max_rows": 0}, "Max rows"),
    ],
)
def test_hub_input_validation(repo, kwargs, message):
    with pytest.raises(ValueError, match=message):
        loader.load_hub_dataset(repo, **kwargs)


def test_hub_rejects_existing_local_directory(tmp_path, monkeypatch):
    (tmp_path / "owner" / "name").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="local path"):
        loader.load_hub_dataset("owner/name")


def test_rows_whose_prompt_fills_max_length_are_dropped():
    ds = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": "x" * 20}],
                [{"role": "user", "content": "hi"}],
            ],
            "completion": [[{"role": "assistant", "content": "a"}]] * 2,
        }
    )
    kept, dropped = drop_prompts_over_limit(ds, _CharTokenizer(), max_length=10)
    assert dropped == 1 and kept[0]["prompt"][0]["content"] == "hi"
    text = Dataset.from_dict({"text": ["x" * 50]})
    assert drop_prompts_over_limit(text, _CharTokenizer(), 10) == (text, 0)
