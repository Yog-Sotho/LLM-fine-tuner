import unittest
import os
import sys
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append(os.getcwd())

# Mock machine learning libraries that might not be installed
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['huggingface_hub'] = MagicMock()
sys.modules['gradio'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['datasets'] = MagicMock()

from core.state import validate_path_traversal, validate_identifier
from export.registry import on_registry_upload

class TestRegistrySecurityHardened(unittest.TestCase):
    def test_on_registry_upload_traversal(self):
        # Path traversal in model path
        res = on_registry_upload("../unsafe", "repo/id", "hf_1234567890123456789012345678901234", "1.0", "")
        self.assertEqual(res, "❌ Path traversal attempt detected.")

    def test_on_registry_upload_version_identifier(self):
        # Forward slash in version
        res = on_registry_upload("./ok", "repo/id", "hf_1234567890123456789012345678901234", "1.0/2.0", "")
        self.assertEqual(res, "❌ Path traversal attempt detected.")

        # Null byte in version
        res = on_registry_upload("./ok", "repo/id", "hf_1234567890123456789012345678901234", "1.0\0tag", "")
        self.assertEqual(res, "❌ Path traversal attempt detected.")

        # Double dot in version
        res = on_registry_upload("./ok", "repo/id", "hf_1234567890123456789012345678901234", "1.0..beta", "")
        self.assertEqual(res, "❌ Path traversal attempt detected.")

    def test_validate_path_traversal_null_byte(self):
        self.assertEqual(validate_path_traversal("path\0with\0null"), "❌ Path traversal attempt detected.")

if __name__ == "__main__":
    unittest.main()
