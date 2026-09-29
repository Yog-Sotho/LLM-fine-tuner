"""Deployment: serving commands, the OpenAI-compatible client and quantized export."""

import json
import sys
import types

import httpx
import pytest
from datasets import Dataset

import export.quantize as quantize
import export.serve as serve
from inference.remote import api_base, remote_chat

# ── Serving commands ───────────────────────────────────────────────────────


@pytest.fixture
def fake_bins(tmp_path, monkeypatch):
    for name in ("llama-server", "vllm"):
        path = tmp_path / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(serve, "HAS_VLLM", True)
    return tmp_path


def test_gguf_is_served_by_llama_server_with_the_key_in_the_environment(fake_bins, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    cmd, env = serve.build_serve_command(str(model), "127.0.0.1", 8081, "secret", "mine")
    assert cmd == [str(fake_bins / "llama-server"), "-m", str(model), "--host", "127.0.0.1",
                   "--port", "8081", "--alias", "mine"]  # fmt: skip
    assert env == {"LLAMA_API_KEY": "secret"} and "secret" not in cmd


def test_adapter_is_served_on_its_base_with_vllm(fake_bins, tmp_path):
    adapter = tmp_path / "run"
    adapter.mkdir()
    config = {"peft_type": "LORA", "base_model_name_or_path": "o/b"}
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    (adapter / "adapter_model.safetensors").write_bytes(b"")
    cmd, env = serve.build_serve_command(str(adapter), "0.0.0.0", 8000, "k", "bot")
    assert cmd[1:3] == ["serve", "o/b"]
    assert cmd[cmd.index("--lora-modules") + 1] == f"bot={adapter}" and "--enable-lora" in cmd
    assert env == {"VLLM_API_KEY": "k"}
    hub_cmd, env = serve.build_serve_command("owner/model", port=9000)
    assert hub_cmd[1:3] == ["serve", "owner/model"] and env == {}
    # vLLM serves LoRA adapters only; others must be merged first.
    (adapter / "adapter_config.json").write_text(json.dumps({**config, "peft_type": "IA3"}))
    with pytest.raises(ValueError, match="vLLM serves LoRA adapters only. Merge this adapter"):
        serve.build_serve_command(str(adapter))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "../x"}, "Give a model"),
        ({"model": "missing.gguf"}, "GGUF file not found"),
        ({"model": "owner/model", "port": 0}, "Port must be"),
        ({"model": "owner/model", "host": "a;b"}, "Invalid host"),
        ({"model": "owner/model", "name": "bad name"}, "Served model name"),
        ({"model": "/no/such/folder"}, "Model not found"),
    ],
)
def test_serve_validation(fake_bins, kwargs, message):
    with pytest.raises(ValueError, match=message):
        serve.build_serve_command(**kwargs)


def test_serve_needs_vllm_for_non_gguf(monkeypatch):
    monkeypatch.setattr(serve, "HAS_VLLM", False)
    with pytest.raises(ValueError, match="vLLM is not installed"):
        serve.build_serve_command("owner/model")


# ── OpenAI-compatible client (fake server) ─────────────────────────────────


@pytest.fixture
def fake_server(monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("authorization") != "Bearer good":
            return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "served-model"}]})
        body = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": f"{body['model']}: ok"}}]}
        )

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))  # fmt: skip
    return seen


def test_remote_chat_picks_the_served_model_and_sends_the_key(fake_server):
    assert remote_chat("http://host:8000", "hi", api_key="good") == "served-model: ok"
    post = json.loads(fake_server[-1].content)
    assert post["messages"] == [{"role": "user", "content": "hi"}] and post["max_tokens"] == 256
    assert remote_chat("http://host:8000/v1/", "hi", model="mine", api_key="good",
                       system_prompt="Be brief.") == "mine: ok"  # fmt: skip
    assert json.loads(fake_server[-1].content)["messages"][0]["role"] == "system"


def test_remote_chat_reports_server_errors(fake_server):
    with pytest.raises(ValueError, match="Server answered 401"):
        remote_chat("http://host:8000", "hi", api_key="bad")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://h/x", "localhost:8000", ""])
def test_only_http_endpoints(url):
    with pytest.raises(ValueError, match="http"):
        api_base(url)


def test_api_base_normalises():
    assert api_base("http://h:1") == api_base("http://h:1/v1/") == "http://h:1/v1"


# ── Quantized export ───────────────────────────────────────────────────────


class _ChatTokenizer:
    chat_template = "x"

    def apply_chat_template(self, conv, tokenize=False):
        return "|".join(m["content"] for m in conv)


def test_calibration_texts_cover_every_layout():
    from data.preprocessing import chat_dataset

    chats = chat_dataset([{"messages": [{"role": "user", "content": "q"},
                                        {"role": "assistant", "content": "a"}]}])  # fmt: skip
    assert quantize.calibration_texts(chats, _ChatTokenizer()) == ["q|a"]
    pairs = Dataset.from_dict({"instruction": ["i"], "output": ["o"]})
    assert quantize.calibration_texts(pairs, _ChatTokenizer()) == ["i\no"]
    texts = Dataset.from_dict({"text": ["t1", " ", "t2"]})
    assert quantize.calibration_texts(texts, _ChatTokenizer(), limit=2) == ["t1"]


@pytest.mark.parametrize(
    ("args", "message"),
    [(("m", "o", "int8"), "Format must be"), (("m", "o", "w4a16"), "needs calibration"),
     (("../m", "o", "fp8"), "Path traversal")],
)  # fmt: skip
def test_quantize_validation(monkeypatch, args, message):
    monkeypatch.setattr(quantize, "HAS_LLMCOMPRESSOR", True)
    assert message in quantize.quantize_model(*args)


def test_quantize_without_llm_compressor(monkeypatch):
    monkeypatch.setattr(quantize, "HAS_LLMCOMPRESSOR", False)
    assert "pip install" in quantize.quantize_model("m", "o", "fp8")


@pytest.fixture
def fake_llmcompressor(monkeypatch):
    calls: dict = {}

    class Modifier:
        def __init__(self, **kwargs):
            calls.setdefault("modifiers", []).append((type(self).__name__, kwargs))

    class QuantizationModifier(Modifier):
        pass

    class GPTQModifier(Modifier):
        pass

    def oneshot(**kwargs):
        calls["oneshot"] = kwargs

    root = types.ModuleType("llmcompressor")
    root.oneshot = oneshot
    mods = types.ModuleType("llmcompressor.modifiers")
    q = types.ModuleType("llmcompressor.modifiers.quantization")
    q.QuantizationModifier, q.GPTQModifier = QuantizationModifier, GPTQModifier
    for name, module in (("llmcompressor", root), ("llmcompressor.modifiers", mods),
                         ("llmcompressor.modifiers.quantization", q)):  # fmt: skip
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(quantize, "HAS_LLMCOMPRESSOR", True)
    return calls


@pytest.fixture
def tiny_adapter(tmp_path):
    """A LoRA adapter (safetensors) with a run record, on the cached tiny test model."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from core.run_config import save_run_config
    from tests.test_smoke_training import TINY_MODEL, _cached_model

    base = _cached_model(TINY_MODEL)
    adapter = tmp_path / "run"
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(base),
                           LoraConfig(r=4, target_modules="all-linear"))  # fmt: skip
    model.save_pretrained(adapter)
    AutoTokenizer.from_pretrained(base).save_pretrained(adapter)
    save_run_config(str(adapter), mode="sft", model=base,
                    dataset=Dataset.from_dict({"text": ["a"]}), seed=1)  # fmt: skip
    return adapter


def test_fp8_export_merges_the_adapter_and_tags_the_card(fake_llmcompressor, tiny_adapter,
                                                         tmp_path):  # fmt: skip
    from huggingface_hub import ModelCard

    out = tmp_path / "fp8"
    status = quantize.quantize_model(str(tiny_adapter), str(out), "fp8")
    assert status.startswith("✅ Exported FP8"), status
    ((name, kwargs),) = fake_llmcompressor["modifiers"]
    assert name == "QuantizationModifier" and kwargs["scheme"] == "FP8_DYNAMIC"
    assert kwargs["ignore"] == ["lm_head"] and "dataset" not in fake_llmcompressor["oneshot"]
    assert (out / "config.json").is_file()  # a full (merged) model, not an adapter
    assert not (out / "adapter_config.json").exists()
    card = ModelCard.load(str(out / "README.md")).data
    assert "fp8" in card.tags and card.library_name == "transformers"


def test_w4a16_export_calibrates_on_the_given_texts(fake_llmcompressor, tiny_adapter, tmp_path):
    status = quantize.quantize_model(str(tiny_adapter), str(tmp_path / "w4"), "w4a16",
                                     ["first text", "second text"])  # fmt: skip
    assert status.startswith("✅ Exported W4A16"), status
    ((name, kwargs),) = fake_llmcompressor["modifiers"]
    assert name == "GPTQModifier" and kwargs["scheme"] == "W4A16"
    call = fake_llmcompressor["oneshot"]
    assert len(call["dataset"]) == 2 and call["num_calibration_samples"] == 2


def test_cli_merge_writes_a_full_model(tiny_adapter, tmp_path):
    from typer.testing import CliRunner

    from cli.commands import app

    out = tmp_path / "merged"
    result = CliRunner().invoke(app, ["merge", "--adapter", str(tiny_adapter),
                                      "--output", str(out)])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (out / "config.json").is_file() and (out / "model.safetensors").is_file()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["export", "--model", "m", "--output", "o", "--format", "int3"], "--format must be"),
        (["export", "--model", "m", "--output", "o", "--quant", "q1"], "--quant must be"),
        (["export", "--model", "m", "--output", "o", "--format", "w4a16"], "calibration-data"),
        (["serve", "--model", "missing.gguf"], "GGUF file not found"),
        (["push", "--model", "tests", "--repo", "a/b", "--token", "x"], "Invalid Hugging Face"),
    ],
)
def test_cli_deploy_errors(args, message):
    from typer.testing import CliRunner

    from cli.commands import app

    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1 and message in result.output
