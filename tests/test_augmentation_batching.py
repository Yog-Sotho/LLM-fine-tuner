import time
import pytest
from datasets import Dataset
from data.augmentation import augment_dataset_v27
from config.constants import COL_TEXT, COL_INSTRUCTION, COL_OUTPUT

def test_augment_dataset_batching_speed():
    # Setup a larger dataset to measure speedup
    texts = ["The quick brown fox jumps over the lazy dog."] * 20
    dataset = Dataset.from_dict({COL_TEXT: texts})

    # Measure batched speed
    t0 = time.time()
    aug_ds, msg = augment_dataset_v27(dataset, augmentation_factor=3, aug_type="synonym")
    t1 = time.time()
    batched_time = t1 - t0

    assert len(aug_ds) == 60
    assert "Augmentation complete" in msg
    print(f"Batched time for 20 examples (factor 3): {batched_time:.4f}s")

def test_augment_dataset_correctness_interspersed():
    # Setup a small dataset to verify row order
    texts = ["Sentence A.", "Sentence B."]
    dataset = Dataset.from_dict({COL_TEXT: texts})

    aug_ds, _ = augment_dataset_v27(dataset, augmentation_factor=2, aug_type="synonym")

    assert len(aug_ds) == 4
    # Expected order: A, A_aug, B, B_aug
    assert aug_ds[0][COL_TEXT] == "Sentence A."
    assert aug_ds[1][COL_TEXT] != "Sentence A."
    assert aug_ds[2][COL_TEXT] == "Sentence B."
    assert aug_ds[3][COL_TEXT] != "Sentence B."

def test_augment_dataset_instruction_mapping():
    # Verify it works with COL_INSTRUCTION
    dataset = Dataset.from_dict({
        COL_INSTRUCTION: ["Tell me a joke.", "What is AI?"],
        COL_OUTPUT: ["Why did the chicken...", "Artificial Intelligence is..."]
    })

    aug_ds, _ = augment_dataset_v27(dataset, augmentation_factor=2, aug_type="synonym")

    assert len(aug_ds) == 4
    # Instruction should be augmented, output should remain same
    assert aug_ds[0][COL_INSTRUCTION] == "Tell me a joke."
    assert aug_ds[1][COL_INSTRUCTION] != "Tell me a joke."
    assert aug_ds[1][COL_OUTPUT] == "Why did the chicken..."
