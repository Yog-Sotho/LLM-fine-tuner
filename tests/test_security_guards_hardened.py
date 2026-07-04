import unittest
from core.state import validate_path_traversal, validate_identifier

class TestSecurityGuardsHardened(unittest.TestCase):
    def test_validate_path_traversal_hardened(self):
        # Existing checks
        self.assertIsNone(validate_path_traversal("safe/path"))
        self.assertIsNotNone(validate_path_traversal("../unsafe"))
        self.assertIsNotNone(validate_path_traversal("C:\\unsafe"))

        # New null-byte check
        self.assertIsNotNone(validate_path_traversal("safe\0path"))
        self.assertEqual(validate_path_traversal("safe\0path"), "❌ Path traversal attempt detected.")

    def test_validate_identifier_hardened(self):
        # Safe identifiers
        self.assertIsNone(validate_identifier("q4_k_m"))
        self.assertIsNone(validate_identifier("v1.0.0"))
        self.assertIsNone(validate_identifier("beta-1"))

        # Unsafe identifiers (separators)
        self.assertIsNotNone(validate_identifier("q4/k_m"))
        self.assertIsNotNone(validate_identifier("../v1"))
        self.assertIsNotNone(validate_identifier("v1\\0"))

        # Unsafe identifiers (null bytes)
        self.assertIsNotNone(validate_identifier("v1\0tag"))
        self.assertEqual(validate_identifier("v1\0tag"), "❌ Path traversal attempt detected.")

if __name__ == "__main__":
    unittest.main()
