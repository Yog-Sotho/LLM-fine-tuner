
import unittest
import pandas as pd
from datasets import Dataset
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from config.constants import (
    COL_INSTRUCTION, COL_OUTPUT, COL_TEXT,
    COL_PROMPT, COL_CHOSEN, COL_REJECTED
)

# Mocking external dependencies for on_train_click
class TestStatsCalculation(unittest.TestCase):
    def test_sft_stats(self):
        data = {
            COL_INSTRUCTION: ["I1", "I2"],
            COL_OUTPUT: ["O1", "O2"]
        }
        ds = Dataset.from_dict(data)

        # We want to test the logic inside on_train_click but it's hard to call it directly
        # without many mocks. Let's test the extracted logic.
        from ui.handlers import on_train_click

        # Mock dependencies
        with patch('ui.handlers.load_dataset_from_file'), \
             patch('ui.handlers.validate_and_clean_dataset', return_value=(ds, [])), \
             patch('ui.handlers.train_model', return_value=("msg", [])), \
             patch('ui.handlers.create_model_card'), \
             patch('ui.handlers.create_zip_from_folder', return_value="zip"):

            # Call on_train_click with enough arguments
            # Note: augmented_ds=ds skips the file loading
            res = on_train_click(
                file=None, model_choice="m", custom_model="", training_preset="Quick",
                peft_method="LoRA", use_lora=True, lora_rank=8, lora_alpha=16,
                prefix_tuning_num_virtual_tokens=0, prefix_tuning_token_dim=0, prefix_tuning_num_layers=0,
                prompt_tuning_num_virtual_tokens=0,
                adapter_reduction_factor=0,
                lr=1e-4, epochs=1, bs=1, grad_accum=1, max_len=512, warmup=0,
                early_stop=0, lr_sched="linear", grad_ckpt=False, resume=False,
                col_inst="", col_out="", col_text="",
                use_unsloth=False, use_chat_template=False, system_prompt="",
                training_mode="sft", dpo_beta=0.1, heretic_mode=False,
                augmented_ds=ds
            )

            # The average length should be (2+2 + 2+2)/2 = 4 (actually len("I1")+len("O1") = 4)
            # "I1" is 2 chars, "O1" is 2 chars. Total 4.
            # "I2" is 2 chars, "O2" is 2 chars. Total 4.
            # Avg is 4.0

            # We need to capture what was passed to create_model_card
            from ui.handlers import create_model_card
            args, kwargs = create_model_card.call_args
            dataset_info = args[1]
            self.assertEqual(dataset_info['num_examples'], 2)
            self.assertEqual(dataset_info['avg_length'], 4.0)

    def test_dpo_stats(self):
        data = {
            COL_PROMPT: ["P1"],
            COL_CHOSEN: ["C1"],
            COL_REJECTED: ["R1"]
        }
        ds = Dataset.from_dict(data)

        from ui.handlers import on_train_click
        with patch('ui.handlers.load_dataset_from_file'), \
             patch('ui.handlers.validate_and_clean_dataset', return_value=(ds, [])), \
             patch('ui.handlers.train_model', return_value=("msg", [])), \
             patch('ui.handlers.create_model_card'), \
             patch('ui.handlers.create_zip_from_folder', return_value="zip"):

            on_train_click(
                file=None, model_choice="m", custom_model="", training_preset="Quick",
                peft_method="LoRA", use_lora=True, lora_rank=8, lora_alpha=16,
                prefix_tuning_num_virtual_tokens=0, prefix_tuning_token_dim=0, prefix_tuning_num_layers=0,
                prompt_tuning_num_virtual_tokens=0,
                adapter_reduction_factor=0,
                lr=1e-4, epochs=1, bs=1, grad_accum=1, max_len=512, warmup=0,
                early_stop=0, lr_sched="linear", grad_ckpt=False, resume=False,
                col_inst="", col_out="", col_text="",
                use_unsloth=False, use_chat_template=False, system_prompt="",
                training_mode="dpo", dpo_beta=0.1, heretic_mode=False,
                augmented_ds=ds
            )

            from ui.handlers import create_model_card
            args, kwargs = create_model_card.call_args
            dataset_info = args[1]
            # "P1" (2) + "C1" (2) + "R1" (2) = 6
            self.assertEqual(dataset_info['avg_length'], 6.0)

if __name__ == "__main__":
    unittest.main()
