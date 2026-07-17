
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import shutil

# Create a mock for app_state
class AppStateMock:
    def __init__(self):
        self._last_batch_path = None
        self._last_merged_dir = None

    def cleanup_resource(self, attr_name, new_value=None):
        old_path = getattr(self, attr_name, None)
        print(f"DEBUG: cleanup_resource called for {attr_name}, old_path={old_path}")
        if old_path and os.path.exists(old_path):
            if os.path.isdir(old_path):
                print(f"DEBUG: Removing directory {old_path}")
                shutil.rmtree(old_path)
            else:
                print(f"DEBUG: Removing file {old_path}")
                os.unlink(old_path)
        setattr(self, attr_name, new_value)

app_state_instance = AppStateMock()

# Mock dependencies
sys.modules['gradio'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['vllm'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['unsloth'] = MagicMock()
sys.modules['PyPDF2'] = MagicMock()
sys.modules['openpyxl'] = MagicMock()

# Mock internal project modules
import core.state
core.state.app_state = app_state_instance
sys.modules['core.state'] = core.state

sys.modules['config.constants'] = MagicMock(HAS_VLLM=True)
sys.modules['data.loader'] = MagicMock()
sys.modules['data.preprocessing'] = MagicMock()
sys.modules['export.hub'] = MagicMock()
sys.modules['export.utils'] = MagicMock()
sys.modules['training.sft'] = MagicMock()

# Import the handlers
import ui.handlers
import inference.vllm_runner

class TestResourceCleanup(unittest.TestCase):
    def setUp(self):
        # Clear app_state before each test
        app_state_instance._last_batch_path = None
        app_state_instance._last_merged_dir = None
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    @patch('ui.handlers.batch_generate')
    @patch('ui.handlers.os.path.isfile')
    def test_batch_inference_cleanup(self, mock_isfile, mock_batch_gen):
        # 1. Setup a "previous" batch file
        prev_path = os.path.join(self.tmp_dir, "prev_batch.csv")
        with open(prev_path, "w") as f:
            f.write("test")
        app_state_instance._last_batch_path = prev_path

        # 2. Setup mock for current run
        curr_path = os.path.join(self.tmp_dir, "curr_batch.csv")
        mock_batch_gen.return_value = curr_path
        mock_isfile.return_value = True

        # 3. Trigger handler
        mock_file = MagicMock()
        ui.handlers.on_batch_test(mock_file, "gpt2", "", "")

        # 4. Verify previous file is deleted and new one is tracked
        self.assertFalse(os.path.exists(prev_path), "Previous batch file should be deleted")
        self.assertEqual(app_state_instance._last_batch_path, curr_path, "New batch file should be tracked")

    @patch('inference.vllm_runner.merge_adapter_for_inference')
    @patch('inference.vllm_runner.os.path.isdir')
    def test_adapter_merge_cleanup(self, mock_isdir, mock_merge):
        # 1. Setup a "previous" merged directory
        prev_dir = os.path.join(self.tmp_dir, "prev_merged")
        os.makedirs(prev_dir)
        app_state_instance._last_merged_dir = prev_dir

        # 2. Setup mock for current run
        mock_merge.return_value = "✅ Success"
        mock_isdir.return_value = True

        # 3. Trigger handler
        with patch('tempfile.mkdtemp') as mock_mkdtemp:
            curr_dir = os.path.join(self.tmp_dir, "curr_merged")
            mock_mkdtemp.return_value = curr_dir
            # We need to simulate the directory being created
            os.makedirs(curr_dir)

            inference.vllm_runner.on_merge_adapter_click("base", "adapter", "")

        # 4. Verify previous dir is deleted and new one is tracked
        self.assertFalse(os.path.exists(prev_dir), "Previous merged directory should be deleted")
        self.assertEqual(app_state_instance._last_merged_dir, curr_dir, "New merged directory should be tracked")

if __name__ == '__main__':
    unittest.main()
