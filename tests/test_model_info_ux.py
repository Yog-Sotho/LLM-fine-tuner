from core.hardware import get_model_info

def test_get_model_info_known_models():
    # Test known models
    info = get_model_info("gpt2")
    assert "124M" in info
    assert "0.5 GiB" in info

    info_custom = get_model_info("meta-llama/Llama-2-7b-hf")
    assert "7B" in info_custom
    assert "14 GiB" in info_custom

def test_get_model_info_unknown_model():
    # Test unknown model
    info = get_model_info("some-unlisted-model")
    assert "unknown" in info

def test_get_active_model_info_logic():
    # Emulate the logic in ui/app.py
    def mock_get_active_model_info(custom, choice):
        custom_stripped = (custom or "").strip()
        active_model = custom_stripped if custom_stripped else choice
        return get_model_info(active_model)

    assert "82M" in mock_get_active_model_info("distilgpt2", "gpt2")
    assert "124M" in mock_get_active_model_info("", "gpt2")
    assert "124M" in mock_get_active_model_info("   ", "gpt2")
