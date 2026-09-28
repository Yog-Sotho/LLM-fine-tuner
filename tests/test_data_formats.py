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


# ── Tool calling and reasoning ─────────────────────────────────────────────

CALL = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}
TOOL_CONV = [
    {"role": "user", "content": "Weather in Rome?"},
    {"role": "assistant", "content": None, "tool_calls": [CALL], "reasoning_content": " Think "},
    {"role": "tool", "name": "get_weather", "tool_call_id": "c1", "content": "Sunny"},
    {"role": "assistant", "content": "It is sunny."},
]


def test_clean_messages_keeps_tool_calls_tool_turns_and_reasoning():
    out = clean_messages(TOOL_CONV)
    assert out[1] == {"role": "assistant", "content": "", "tool_calls": [CALL],
                      "reasoning_content": "Think"}  # fmt: skip
    assert out[2] == {"role": "tool", "content": "Sunny", "name": "get_weather",
                      "tool_call_id": "c1"}  # fmt: skip


def test_json_string_arguments_become_objects_and_filler_nulls_go():
    conv = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}],
         "name": None},
        {"role": "assistant", "tool_calls": [{"function": {"name": "g",
         "arguments": {"a": None, "b": 2}}}]},
    ]  # fmt: skip
    out = clean_messages(conv)
    assert out[1]["tool_calls"][0] == {"type": "function",
                                       "function": {"name": "f", "arguments": {"a": 1}}}  # fmt: skip
    assert "name" not in out[1]
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"b": 2}


def test_a_final_tool_call_is_a_valid_training_target():
    assert clean_messages(TOOL_CONV[:2])[-1]["tool_calls"] == [CALL]


@pytest.mark.parametrize(
    "bad_calls",
    [[{"function": {"arguments": {}}}], [{"function": {"name": "f", "arguments": "{broken"}}],
     "not a list", [{"function": {"name": "f", "arguments": [1, 2]}}]],
)  # fmt: skip
def test_malformed_tool_calls_reject_the_conversation(bad_calls):
    conv = [{"role": "user", "content": "q"}, {"role": "assistant", "tool_calls": bad_calls}]
    assert clean_messages(conv) is None


def test_clean_tools():
    from data.preprocessing import clean_tools

    assert clean_tools(None) == "" and clean_tools(" ") == ""
    assert clean_tools([{"type": "function", "x": None}]) == '[{"type": "function"}]'
    assert clean_tools('[{"a": 1}]') == '[{"a": 1}]'
    with pytest.raises(ValueError, match="list of JSON function schemas"):
        clean_tools('{"a": 1}')


def test_chat_dataset_keeps_differently_shaped_messages_exact():
    from data.preprocessing import chat_dataset

    other = {"type": "function", "function": {"name": "lights", "arguments": {"room": "k"}}}
    ds = chat_dataset([
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "tool_calls": [CALL]}],
         "tools": [{"type": "function"}]},
        {"messages": [{"role": "user", "content": "b"},
                      {"role": "assistant", "tool_calls": [other]}], "tools": None},
    ])  # fmt: skip
    # Arrow struct inference would give each call the other's argument keys as nulls.
    assert ds[0]["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"city": "Rome"}
    assert ds[1]["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"room": "k"}
    assert ds["tools"] == ['[{"type": "function"}]', ""]
    assert len(chat_dataset([])) == 0


def test_tool_chat_file_loads_cleans_and_expands_per_assistant_turn(tmp_path):
    import json

    data = tmp_path / "tools.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in [
        {"messages": TOOL_CONV, "tools": [{"type": "function", "function": {"name": "get_weather"}}]},
        {"messages": CONV},
    ]))  # fmt: skip

    class Upload:
        name = str(data)

    ds = loader.load_dataset_from_file(Upload(), "jsonl")
    cleaned, _ = validate_and_clean_dataset(ds)
    assert cleaned.column_names == ["messages", "tools"] and len(cleaned) == 2
    assert "[calls get_weather" in preview_dataset(cleaned)["messages"][0]
    sft = to_sft_dataset(cleaned, use_chat_template=True, system_prompt="")
    # The tool chat trains its call and its answer; the plain chat only its last answer.
    assert [[m["role"] for m in r["prompt"]] for r in sft] == [
        ["user"],
        ["user", "assistant", "tool"],
        ["system", "user"],
    ]
    assert sft[0]["completion"][0]["tool_calls"] == [CALL]
    assert json.loads(sft[0]["tools"])[0]["function"]["name"] == "get_weather"
    assert sft[2]["tools"] == ""


def test_token_report_renders_tools():
    class ToolTokenizer(_CharTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                                tools=None):  # fmt: skip
            prefix = json.dumps(tools) if tools else ""
            return prefix + super().apply_chat_template(messages, tokenize, add_generation_prompt)

    import json

    from data.preprocessing import chat_dataset

    ds = chat_dataset([{"prompt": [{"role": "user", "content": "ab"}],
                        "completion": [{"role": "assistant", "content": "cd"}],
                        "tools": '[{"n": 1}]'}])  # fmt: skip
    report = token_length_report(ds, ToolTokenizer(), 1000, 10)
    assert report["max"] == len('[{"n": 1}]' + "ab|cd")
    kept, dropped = drop_prompts_over_limit(ds, ToolTokenizer(), max_length=5)
    assert dropped == 1 and len(kept) == 0  # the tool schemas count towards the prompt


# ── Vision chats (images) ──────────────────────────────────────────────────


def _png(path, colour="red"):
    from PIL import Image

    Image.new("RGB", (8, 8), colour).save(path)


def test_clean_messages_keeps_text_and_image_parts():
    from data.preprocessing import content_text, count_image_parts

    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": " Hi "}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Red."}]}]  # fmt: skip
    out = clean_messages(conv)
    assert out[0]["content"] == [{"type": "image"}, {"type": "text", "text": "Hi"}]
    assert count_image_parts(out) == 1 and content_text(out[1]["content"]) == "Red."
    bad = [{"role": "user", "content": [{"type": "video"}]}, CONV[2]]
    assert clean_messages(bad) is None


def _vision_rows(tmp_path, n_images_per_row, parts_per_row):
    import json

    rows = []
    for i, (n_images, parts) in enumerate(zip(n_images_per_row, parts_per_row, strict=True)):
        names = []
        for j in range(n_images):
            _png(tmp_path / f"{i}_{j}.png", ["red", "blue"][j % 2])
            names.append(f"{i}_{j}.png")
        user = [{"type": "image"}] * parts + [{"type": "text", "text": f"q{i}"}]
        rows.append({"messages": [{"role": "user", "content": user},
                                  {"role": "assistant", "content": f"a{i}"}], "images": names})  # fmt: skip
    data = tmp_path / "vision.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))

    class Upload:
        name = str(data)

    return loader.load_dataset_from_file(Upload(), "jsonl")


def test_image_parts_must_match_images_unless_content_is_text_only(tmp_path):
    # rows: 1 part/1 image ok; 2 parts/1 image dropped; 0 parts/1 image ok (TRL places it)
    ds = _vision_rows(tmp_path, [1, 1, 1], [1, 2, 0])
    cleaned, issues = validate_and_clean_dataset(ds)
    assert len(cleaned) == 2 and any("invalid" in i for i in issues)
    assert "🖼️ 1 image(s)" in preview_dataset(cleaned)["messages"][0]


def test_images_are_kept_byte_for_byte(tmp_path):
    from datasets import Image, List

    ds = _vision_rows(tmp_path, [1], [1])
    cleaned, _ = validate_and_clean_dataset(ds)
    sft = to_sft_dataset(cleaned, True, "")
    raw = sft.cast_column("images", List(Image(decode=False)))[0]["images"][0]
    stored = raw["bytes"] or open(raw["path"], "rb").read()
    assert stored == (tmp_path / "0_0.png").read_bytes()


def test_same_text_with_different_images_is_not_a_duplicate(tmp_path):
    import json

    for name, colour in (("a.png", "red"), ("b.png", "blue"), ("c.png", "red")):
        _png(tmp_path / name, colour)
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]},
            {"role": "assistant", "content": "a"}]  # fmt: skip
    data = tmp_path / "v.jsonl"
    data.write_text("\n".join(json.dumps({"messages": conv, "images": [n]})
                              for n in ("a.png", "b.png", "b.png")))  # fmt: skip

    class Upload:
        name = str(data)

    cleaned, issues = validate_and_clean_dataset(loader.load_dataset_from_file(Upload(), "jsonl"))
    assert len(cleaned) == 2 and any("1 duplicate" in i for i in issues)


@pytest.mark.parametrize("ref", ["../outside.png", "/etc/passwd", "missing.png", 42])
def test_local_images_must_be_files_next_to_the_data(tmp_path, ref):
    import json

    data = tmp_path / "data" / "v.jsonl"
    data.parent.mkdir()
    _png(tmp_path / "outside.png")
    data.write_text(json.dumps({"messages": CONV, "images": [ref]}))

    class Upload:
        name = str(data)

    with pytest.raises(RuntimeError, match="Image|image"):
        loader.load_dataset_from_file(Upload(), "jsonl")


def test_single_image_column_and_prompt_completion_chats_are_normalised(tmp_path):
    import json

    _png(tmp_path / "a.png")
    row = {"prompt": [{"role": "user", "content": "What is it?"}],
           "completion": [{"role": "assistant", "content": "A square."}], "image": "a.png"}  # fmt: skip
    data = tmp_path / "pc.jsonl"
    data.write_text(json.dumps(row))

    class Upload:
        name = str(data)

    ds = loader.load_dataset_from_file(Upload(), "jsonl")
    assert ds.column_names == ["messages", "images"] and len(ds[0]["messages"]) == 2


def test_hub_prompt_completion_vision_chats(fake_hub):
    import io

    from datasets import Features, Image, List, Value
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (8, 8), "red").save(buf, "PNG")
    msg = List({"role": Value("string"), "content": Value("string")})
    fake_hub["rows"] = Dataset.from_dict(
        {"prompt": [[{"role": "user", "content": "Colour?"}]],
         "completion": [[{"role": "assistant", "content": "Red."}]],
         "images": [[{"bytes": buf.getvalue(), "path": None}]]},
        features=Features({"prompt": msg, "completion": msg, "images": List(Image())}),
    )  # fmt: skip
    ds = loader.load_hub_dataset("owner/vision")
    assert ds.column_names == ["messages", "images"]
    assert [m["role"] for m in ds[0]["messages"]] == ["user", "assistant"]
