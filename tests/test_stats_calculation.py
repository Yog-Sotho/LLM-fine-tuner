import pytest
import pandas as pd
from datasets import Dataset
from ui.handlers import on_train_click
from config.constants import COL_PROMPT, COL_CHOSEN, COL_REJECTED, COL_INSTRUCTION, COL_OUTPUT, COL_TEXT

def test_stats_calculation_parity(mocker):
    # Mock dependencies of on_train_click that we don't want to run
    mocker.patch("ui.handlers.validate_path_traversal", return_value=None)
    mocker.patch("ui.handlers.train_model", return_value=("Success", []))
    mocker.patch("ui.handlers.create_model_card")
    mocker.patch("ui.handlers.create_zip_from_folder", return_value="dummy.zip")
    mocker.patch("ui.handlers.app_state")

    # Common arguments
    kwargs = {
        "file": None, "model_choice": "gpt2", "custom_model": "", "training_preset": "Quick (1 epoch)",
        "peft_method": "LoRA", "use_lora": True, "lora_rank": 8, "lora_alpha": 16,
        "prefix_tuning_num_virtual_tokens": 20, "prefix_tuning_token_dim": 768, "prefix_tuning_num_layers": 12,
        "prompt_tuning_num_virtual_tokens": 20, "adapter_reduction_factor": 16,
        "lr": 5e-4, "epochs": 1, "bs": 1, "grad_accum": 1, "max_len": 512, "warmup": 0,
        "early_stop": 0, "lr_sched": "linear", "grad_ckpt": False, "resume": False,
        "col_inst": "instruction", "col_out": "output", "col_text": "text",
        "use_unsloth": False, "use_chat_template": False, "system_prompt": "Prompt",
        "training_mode": "sft", "dpo_beta": 0.1, "heretic_mode": False,
        "augmented_ds": None
    }

    # 1. Test DPO format
    data_dpo = {
        COL_PROMPT: ["p1", "p2"],
        COL_CHOSEN: ["c1", "c22"],
        COL_REJECTED: ["r1", "r222"]
    }
    ds_dpo = Dataset.from_dict(data_dpo)
    kwargs["augmented_ds"] = ds_dpo
    kwargs["training_mode"] = "dpo"

    # We need to capture the dataset_info passed to create_model_card
    mock_card = mocker.patch("ui.handlers.create_model_card")
    on_train_click(**kwargs)

    # avg_len = ( (2+2+2) + (2+3+4) ) / 2 = (6 + 9) / 2 = 7.5
    dataset_info = mock_card.call_args[0][1]
    assert dataset_info["avg_length"] == 7.5

    # 2. Test SFT Instruction format
    data_sft = {
        COL_INSTRUCTION: ["i1", "i22"],
        COL_OUTPUT: ["o1", "o222"]
    }
    ds_sft = Dataset.from_dict(data_sft)
    kwargs["augmented_ds"] = ds_sft
    kwargs["training_mode"] = "sft"

    mock_card.reset_mock()
    on_train_click(**kwargs)

    # avg_len = ( (2+2) + (3+4) ) / 2 = (4 + 7) / 2 = 5.5
    dataset_info = mock_card.call_args[0][1]
    assert dataset_info["avg_length"] == 5.5

    # 3. Test Text-only format
    data_text = {
        COL_TEXT: ["t1", "t222"]
    }
    ds_text = Dataset.from_dict(data_text)
    kwargs["augmented_ds"] = ds_text

    mock_card.reset_mock()
    on_train_click(**kwargs)

    # avg_len = (2 + 4) / 2 = 3.0
    dataset_info = mock_card.call_args[0][1]
    assert dataset_info["avg_length"] == 3.0
