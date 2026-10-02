"""export/gguf.py and inference/vllm_runner.py paths that need tools we fake here."""

import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

import export.gguf as gguf
import inference.vllm_runner as vr
from core.state import app_state

# ── GGUF export (llama.cpp path) ───────────────────────────────────────────


@pytest.fixture
def model_dir(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    return str(folder)


@pytest.fixture
def llama_cpp(monkeypatch):
    """Fake convert_hf_to_gguf.py / llama-quantize; each run writes its output file."""
    state = {"convert_rc": 0, "quantize_rc": 0, "quantize": True, "timeout": False, "calls": []}
    monkeypatch.setattr(gguf, "HAS_UNSLOTH", False)

    def which(name):
        if name == "convert_hf_to_gguf.py":
            return "/tools/convert_hf_to_gguf.py"
        if name == "llama-quantize" and state["quantize"]:
            return "/tools/llama-quantize"
        return None

    def run(cmd, **kwargs):
        state["calls"].append(cmd)
        if state["timeout"]:
            raise subprocess.TimeoutExpired(cmd, 900)
        if cmd[0] == sys.executable:
            open(cmd[cmd.index("--outfile") + 1], "wb").write(b"f16")
            return SimpleNamespace(returncode=state["convert_rc"], stderr="convert error")
        open(cmd[2], "wb").write(b"q")
        return SimpleNamespace(returncode=state["quantize_rc"], stderr="quantize error")

    monkeypatch.setattr(gguf.shutil, "which", which)
    monkeypatch.setattr(gguf.subprocess, "run", run)
    return state


def test_gguf_converts_then_quantizes(llama_cpp, model_dir, tmp_path):
    status = gguf.export_to_gguf(model_dir, str(tmp_path / "out"), "q4_k_m")
    assert status.startswith("✅ GGUF exported & quantized (q4_k_m)")
    convert, quantize = llama_cpp["calls"]
    assert (
        convert[:2] == [sys.executable, "/tools/convert_hf_to_gguf.py"] and convert[2] == model_dir
    )
    assert quantize[-1] == "q4_k_m"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["model_q4_k_m.gguf"]


def test_gguf_without_quantize_tool_keeps_fp16(llama_cpp, model_dir, tmp_path):
    llama_cpp["quantize"] = False
    status = gguf.export_to_gguf(model_dir, str(tmp_path / "out"))
    assert status.startswith("✅ GGUF exported (FP16 only)") and "model_fp16.gguf" in status


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ({"convert_rc": 1}, "❌ llama.cpp conversion failed:\nconvert error"),
        ({"quantize_rc": 1}, "⚠️ Quantization failed. Using FP16 version.\nquantize error"),
        ({"timeout": True}, "❌ GGUF conversion timed out"),
    ],
)
def test_gguf_tool_failures(llama_cpp, model_dir, tmp_path, setting, message):
    llama_cpp.update(setting)
    assert gguf.export_to_gguf(model_dir, str(tmp_path / "out")).startswith(message)


def test_gguf_needs_a_converter(monkeypatch, model_dir, tmp_path):
    monkeypatch.setattr(gguf, "HAS_UNSLOTH", False)
    monkeypatch.setattr(gguf.shutil, "which", lambda name: None)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/llama.cpp either
    assert gguf.export_to_gguf(model_dir, str(tmp_path / "o")).startswith("❌ GGUF export requires")
    converter = tmp_path / "llama.cpp" / "convert_hf_to_gguf.py"
    converter.parent.mkdir()
    converter.write_text("")
    monkeypatch.setattr(
        gguf.subprocess, "run", lambda cmd, **k: SimpleNamespace(returncode=1, stderr="x")
    )
    assert gguf.export_to_gguf(model_dir, str(tmp_path / "o")).startswith("❌ llama.cpp conversion")


def test_gguf_rejects_a_bad_quantization_name(model_dir, tmp_path):
    assert gguf.export_to_gguf(model_dir, str(tmp_path), "q4/../x").startswith("❌")


def test_gguf_via_unsloth_and_fallback(monkeypatch, llama_cpp, model_dir, tmp_path):
    saved = {}

    class Model:
        def save_pretrained_gguf(self, out, tokenizer, quantization_method):
            saved["method"] = quantization_method
            if quantization_method == "broken":
                raise RuntimeError("CUDA out of memory")
            open(f"{out}/unsloth.Q8_0.gguf", "wb").write(b"g")

    fast = SimpleNamespace(from_pretrained=lambda **kwargs: (Model(), "tok"))
    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=fast))
    monkeypatch.setattr(gguf, "HAS_UNSLOTH", True)
    status = gguf.export_to_gguf(model_dir, str(tmp_path / "u"), "q8_0")
    assert status.startswith("✅ GGUF exported via Unsloth (Q8_0)") and not llama_cpp["calls"]
    status = gguf.export_to_gguf(model_dir, str(tmp_path / "f"), "broken")
    assert status.startswith("✅ GGUF exported & quantized") and llama_cpp["calls"]


def test_ui_export_returns_the_file(llama_cpp, model_dir):
    status, path = gguf.on_export_gguf(f"  {model_dir}  ", "q4_k_m")
    assert status.startswith("✅") and path.endswith("model_q4_k_m.gguf")
    app_state.session_for(None).release("gguf_dir")
    assert gguf.on_export_gguf("", "q4_k_m") == ("❌ No trained model found. Train first.", None)
    assert gguf.on_export_gguf(model_dir, "a/b")[1] is None


def test_adapter_merge_errors(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    merged, error = gguf.merge_adapter_to_temp(str(adapter))
    assert merged is None and "does not name the base model" in error
    (adapter / "adapter_config.json").write_text(
        '{"peft_type": "LORA", "base_model_name_or_path": "b"}'
    )
    monkeypatch.setattr(gguf, "merge_adapter_for_inference", lambda *a: "❌ merge failed")
    assert gguf.merge_adapter_to_temp(str(adapter)) == (None, "❌ merge failed")


# ── vLLM ───────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_vllm(monkeypatch):
    engines = []

    class LLM:
        def __init__(self, model, quantization, tensor_parallel_size, trust_remote_code):
            self.model, self.quantization = model, quantization
            engines.append(self)

        def generate(self, prompts, params):
            return [
                SimpleNamespace(outputs=[SimpleNamespace(text=f"{self.model}:{p}")])
                for p in prompts
            ]

    module = types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(vr, "HAS_VLLM", True)
    monkeypatch.setattr(app_state, "vllm_cache", {})
    monkeypatch.setattr(app_state, "max_vllm_engines", 1)
    return engines


def test_vllm_engines_are_cached_and_evicted(fake_vllm):
    assert vr.vllm_generate_v27("m1", ["a", "b"]) == ["m1:a", "m1:b"]
    vr.vllm_generate_v27("m1", ["c"])
    assert len(fake_vllm) == 1  # reused
    vr.vllm_generate_v27("m2", ["d"], vllm_quantization="awq")
    assert len(fake_vllm) == 2 and fake_vllm[1].quantization == "awq"
    assert list(app_state.vllm_cache) == [("m2", "awq", 1)]  # m1 evicted at the limit


def test_vllm_input_checks(fake_vllm, monkeypatch):
    with pytest.raises(ValueError, match="Path traversal"):
        vr.vllm_generate_v27("../m", ["a"])
    monkeypatch.setattr(vr, "HAS_VLLM", False)
    with pytest.raises(ImportError, match="vLLM not installed"):
        vr.vllm_generate_v27("m", ["a"])


def test_vllm_ui_handler(fake_vllm, tmp_path, monkeypatch):
    assert vr.on_vllm_generate(str(tmp_path), " hi ", "none", 16, 0.7, 0.9) == f"{tmp_path}:hi"
    assert (
        vr.on_vllm_generate(str(tmp_path), "  ", "none", 16, 0.7, 0.9)
        == "❌ Please enter a prompt."
    )

    def broken(**kwargs):
        raise RuntimeError("CUDA OOM")

    monkeypatch.setattr(vr, "vllm_generate_v27", broken)
    assert (
        vr.on_vllm_generate(str(tmp_path), "hi", "none", 16, 0.7, 0.9)
        == "❌ vLLM inference failed: CUDA OOM"
    )
    monkeypatch.setattr(vr, "HAS_VLLM", False)
    assert vr.on_vllm_generate(str(tmp_path), "hi", "none", 16, 0.7, 0.9).startswith(
        "❌ vLLM not installed"
    )


def test_merge_button(monkeypatch, tmp_path):
    monkeypatch.setattr(vr, "merge_adapter_for_inference", lambda base, adapter, out: "✅ merged")
    status, update = vr.on_merge_adapter_click("org/base", "", str(tmp_path))
    assert (
        status == "✅ merged"
        and update["value"].startswith("/")
        and "merged_model_" in update["value"]
    )
    app_state.session_for(None).release("merged_dir")
    monkeypatch.setattr(vr, "merge_adapter_for_inference", lambda *a: "❌ no")
    assert vr.on_merge_adapter_click("org/base", str(tmp_path), None)[0] == "❌ no"
    app_state.session_for(None).release("merged_dir")
    assert "Path traversal" in vr.on_merge_adapter_click("../b", str(tmp_path), None)[0]
    assert vr.on_merge_adapter_click("b", "", None)[0].startswith("❌ No valid adapter")


@pytest.mark.parametrize(
    ("base", "adapter", "message"),
    [
        ("", "x", "❌ Please provide the base model ID"),
        ("b", "/no/such/dir", "❌ Adapter path is invalid"),
        ("../b", "x", "Path traversal"),
    ],
)
def test_merge_input_checks(tmp_path, base, adapter, message):
    assert message in vr.merge_adapter_for_inference(base, adapter, str(tmp_path / "out"))


def test_vllm_gets_the_bitsandbytes_name(fake_vllm):
    vr.vllm_generate_v27("m", ["a"], vllm_quantization="bnb")
    assert fake_vllm[-1].quantization == "bitsandbytes"
    from config.constants import VLLM_QUANT_OPTIONS

    assert "bitsandbytes" in VLLM_QUANT_OPTIONS and "bnb" not in VLLM_QUANT_OPTIONS
