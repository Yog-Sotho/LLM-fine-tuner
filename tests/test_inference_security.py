import pytest
from unittest.mock import MagicMock, patch
from inference.generate import _load_for_inference, generate_text, batch_generate

def test_load_for_inference_model_path_traversal():
    # Test with traversal sequence in model_name
    with pytest.raises(ValueError) as exc_info:
        _load_for_inference("../unsafe_model", None)
    assert "❌ Path traversal attempt detected." in str(exc_info.value)

def test_load_for_inference_lora_path_traversal():
    # Test with traversal sequence in lora_path
    with pytest.raises(ValueError) as exc_info:
        _load_for_inference("gpt2", "../unsafe_lora")
    assert "❌ Path traversal attempt detected." in str(exc_info.value)

def test_load_for_inference_null_byte_traversal():
    # Test with null byte in model_name
    with pytest.raises(ValueError) as exc_info:
        _load_for_inference("gpt2\0", None)
    assert "❌ Path traversal attempt detected." in str(exc_info.value)

def test_generate_text_traversal_fails_safely():
    # Verify that generate_text catches the ValueError and returns a secure error message
    result = generate_text("../unsafe_model", None, "Hello")
    assert "❌ Generation failed: ❌ Path traversal attempt detected." in result

def test_batch_generate_traversal_fails_safely():
    # Create a mock file object
    mock_file = MagicMock()
    mock_file.name = "prompts.csv"

    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = MagicMock()
        mock_read_csv.return_value.columns = ["prompt"]
        mock_read_csv.return_value["prompt"].tolist.return_value = ["Hello"]

        result = batch_generate("../unsafe_model", None, mock_file)
        assert "❌ Path traversal attempt detected." in result
