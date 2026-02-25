"""
╔══════════════════════════════════════════════════════════════════════╗
║              🧠 Advanced LLM Fine-Tuner  —  v2.2                    ║
║    v2.1 bug fixes  +  batch inference optimization  +  new PEFT methods  ║
╚══════════════════════════════════════════════════════════════════════╝
CHANGES FROM v2.1:
CRITICAL BUG FIXES
──────────────────
[FIX-8] Syntax errors fixed (spaces in variable names)
[FIX-9] Tokenizer initialization handles models without EOS tokens
[FIX-10] ZIP extraction secured against path traversal attacks
[FIX-11] Dataset validation properly handles empty datasets
PERFORMANCE IMPROVEMENTS
───────────────────────
[PERF-1] Batch inference optimized with batch processing
NEW FEATURES
────────────
[NEW-6] Added support for additional parameter-efficient methods (Prefix Tuning, Prompt Tuning, Adapters)
"""
─────────────────────────────────────────────
Standard library & third-party imports
─────────────────────────────────────────────
import os
import gc
import json
import zipfile
import tempfile
import threading
import time
import shutil
import glob
import warnings
from datetime import datetime
import pandas as pd
import gradio as gr
from datasets import Dataset
from transformers import (
AutoTokenizer,
AutoModelForCausalLM,
TrainingArguments,
Trainer,
DataCollatorForLanguageModeling,
BitsAndBytesConfig,
EarlyStoppingCallback,
TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PrefixTuningConfig, PromptTuningConfig, PromptTuningInit, AdaptersConfig, AdapterConfig
import torch
import numpy as np
warnings.filterwarnings("ignore")
─────────────────────────────────────────────
Optional dependency detection
─────────────────────────────────────────────
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
try:
    from huggingface_hub import HfApi
    HAS_HUB = True
except ImportError:
    HAS_HUB = False
─────────────────────────────────────────────
Global state
─────────────────────────────────────────────
stop_event = threading.Event()          # [BF-3] thread-safe stop flag
_inference_cache: dict = {}             # [BF-9] model/tokenizer cache
─────────────────────────────────────────────
Hardware info
─────────────────────────────────────────────
def get_hardware_summary() -> str:
    lines = []
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        lines.append(f"🟢  GPU:  {name}  |  VRAM: {vram:.1f} GB ")
    else:
        lines.append("🟡  GPU:  Not available — training will use CPU (slow) ")
    if HAS_PSUTIL:
        try:
            ram = psutil.virtual_memory().total / 1e9
            lines.append(f"💾  System RAM:  {ram:.1f} GB ")
        except Exception:
            lines.append("💾  System RAM:  unavailable ")
    else:
        lines.append("💾  System RAM:  install  `psutil`  to see this ")
    lines.append(f"🐍  PyTorch:  {torch.__version__} ")
    deps = []
    if HAS_OPENPYXL: deps.append("openpyxl ✓ ")
    else:            deps.append("openpyxl ✗ (no Excel) ")
    if HAS_PDF:      deps.append("PyPDF2 ✓ ")
    else:            deps.append("PyPDF2 ✗ (no PDF) ")
    if HAS_HUB:      deps.append("huggingface_hub ✓ ")
    else:            deps.append("huggingface_hub ✗ (no Hub push) ")
    if HAS_PSUTIL:   deps.append("psutil ✓ ")
    else:            deps.append("psutil ✗ ")
    lines.append("📦  Optional deps: " + " | ".join(deps))
    return "\n".join(lines)


def auto_recommend_model() -> str:
    """[IMP-4] Suggest a sensible default based on available VRAM. """
    if not torch.cuda.is_available():
        return "gpt2 "
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram < 4:
        return "gpt2 "
    elif vram < 8:
        return "facebook/opt-350m "
    elif vram < 16:
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0 "
    else:
        return "mistralai/Mistral-7B-v0.1 "
─────────────────────────────────────────────
Model metadata helpers
─────────────────────────────────────────────
def get_model_info(model_id: str) -> str:
    m = model_id.lower()
    table = {
        "gpt2-xl ":     ("1.5B ",   "6 GB "),
        "gpt2-large ":  ("774M ",   "3 GB "),
        "gpt2-medium ": ("355M ",   "1.5 GB "),
        "gpt2 ":        ("124M ",   "0.5 GB "),
        "distilgpt2 ":  ("82M ",    "0.3 GB "),
        "opt-125m ":    ("125M ",   "0.5 GB "),
        "opt-350m ":    ("350M ",   "1.4 GB "),
        "opt-1.3b ":    ("1.3B ",   "2.7 GB "),
        "pythia-70m ":  ("70M ",    "0.3 GB "),
        "pythia-160m ": ("160M ",   "0.6 GB "),
        "tinyllama ":   ("1.1B ",   "2.2 GB "),
        "llama-2-7b ":  ("7B ",     "14 GB "),
        "mistral-7b ":  ("7B ",     "14 GB "),
        "llama-2-13b ": ("13B ",    "26 GB "),
    }
    for key, (params, mem) in table.items():
        if key in m:
            return f" Parameters:  {params}  |   Estimated RAM/VRAM:  {mem} "
    return " Parameters:  unknown  |   Estimated RAM/VRAM:  unknown "


[BF-4] Per-architecture LoRA target modules
LORA_TARGET_MAP = {
    "gpt2 ":      ["c_attn "],
    "gpt_neo ":   ["q_proj ", "v_proj "],
    "opt ":       ["q_proj ", "v_proj "],
    "llama ":     ["q_proj ", "v_proj "],
    "mistral ":   ["q_proj ", "v_proj "],
    "pythia ":    ["query_key_value "],
    "falcon ":    ["query_key_value "],
    "tinyllama ": ["q_proj ", "v_proj "],
    "default ":   ["q_proj ", "v_proj "],
}


def get_lora_targets(model_name: str) -> list:
    m = model_name.lower()
    for key, targets in LORA_TARGET_MAP.items():
        if key in m:
            return targets
    return LORA_TARGET_MAP["default"]
─────────────────────────────────────────────
Dataset helpers
─────────────────────────────────────────────
def detect_file_type(file) -> str | None:
    name = file.name.lower()
    if name.endswith(".csv "):   return "csv "
    if name.endswith(".jsonl "): return "jsonl "
    if name.endswith(".json "):  return "json "   # [NEW] plain JSON array support
    if name.endswith(".txt "):   return "txt "
    if name.endswith(".xlsx ") and HAS_OPENPYXL: return "excel "
    if name.endswith(".pdf ")  and HAS_PDF:      return "pdf "
    return None


def extract_text_from_pdf(pdf_path: str) -> str:
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)


def load_dataset_from_file(file, file_type: str, column_mapping: dict | None = None) -> Dataset:
    """[BF-1] Fixed: removed dead-code branch with undefined  `dataset` . """
    try:
        if file_type == "jsonl ":
            data = []
            with open(file.name, "r ", encoding="utf-8 ") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return Dataset.from_list(data)
        
        if file_type == "json ":  # [NEW] plain JSON array of objects
            with open(file.name, "r ", encoding="utf-8 ") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a top-level array of objects. ")
            return Dataset.from_list(data)


        if file_type == "txt ":
            with open(file.name, "r ", encoding="utf-8 ") as f:
                lines = [l.strip() for l in f if l.strip()]
            return Dataset.from_dict({"text ": lines})


        if file_type == "pdf ":
            text = extract_text_from_pdf(file.name)
            paragraphs = [p.strip() for p in text.split("\n\n ") if p.strip()]
            return Dataset.from_dict({"text ": paragraphs})


        # CSV / Excel — need column mapping logic
        if file_type == "csv ":
            df = pd.read_csv(file.name)
        elif file_type == "excel ":
            df = pd.read_excel(file.name, engine="openpyxl ")
        else:
            raise ValueError(f"Unsupported file type: {file_type} ")


        if column_mapping:
            df = df.rename(columns=column_mapping)


        if "instruction " in df.columns and "output " in df.columns:
            return Dataset.from_pandas(df[["instruction ", "output "]].astype(str))
        elif "text " in df.columns:
            return Dataset.from_pandas(df[["text "]].astype(str))
        else:
            raise ValueError(
                f"Cannot determine columns automatically. Available: {list(df.columns)}.  "
                "Please use the column mapping dropdowns above. "
            )


    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e} ")


def safe_extract_zip(zip_path: str, extract_dir: str) -> str:
    """Safely extract ZIP file with path traversal protection."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for file_info in zf.infolist():
            # Normalize the file path and prevent traversal
            file_path = os.path.normpath(file_info.filename)
            if file_path.startswith(("../", "..\\")):
                raise ValueError("Invalid file path in ZIP (potential path traversal)")
            # Extract to the target directory
            zf.extract(file_info, extract_dir)
    return extract_dir


def validate_and_clean_dataset(dataset: Dataset):
    issues = []
    if "text " in dataset.column_names:
        lengths = [len(str(t)) for t in dataset["text "]]
    elif "instruction " in dataset.column_names and "output " in dataset.column_names:
        lengths = [len(str(i)) + len(str(o))
                  for i, o in zip(dataset["instruction "], dataset["output "])]
    else:
        return dataset, ["⚠️ Unknown column structure — cannot validate."]
    
    empty = sum(1 for l in lengths if l == 0)
    if empty:
        issues.append(f"⚠️ {empty} empty examples removed. ")
        if "text " in dataset.column_names:
            dataset = dataset.filter(lambda x: len(str(x["text "])) > 0)
        else:
            dataset = dataset.filter(
                lambda x: len(str(x["instruction "])) + len(str(x["output "])) > 0
            )
    
    # Check if dataset is empty after cleaning
    if len(dataset) == 0:
        issues.append("❌ Dataset is empty after cleaning. No valid examples remain.")
        return dataset, issues
    
    long_ = sum(1 for l in lengths if l > 2048)
    if long_:
        issues.append(f"⚠️ {long_} examples exceed 2048 chars — they will be truncated. ")


    return dataset, issues


def preview_dataset(dataset: Dataset) -> tuple[str, str]:
    # [FIX-7] Guard against empty dataset after cleaning
    if len(dataset) == 0:
        return "⚠️ Dataset is empty after cleaning. Please check your file.", "No examples remaining."
    
    if "text " in dataset.column_names:
        examples = [str(t) for t in dataset.select(range(min(5, len(dataset))))["text "]]
        lengths = [len(str(t)) for t in dataset["text "]]
        if lengths:
            stats = (f"**Total examples:** {len(dataset)}\n "
                     f"**Avg length:** {np.mean(lengths):.0f} chars  |   "
                     f"**Min:** {np.min(lengths)}  |  **Max:** {np.max(lengths)} ")
        else:
            stats = f"**Total examples:** {len(dataset)} "
    elif "instruction " in dataset.column_names and "output " in dataset.column_names:
        examples = [
            f"**Instruction:** {i}\n**Output:** {o} "
            for i, o in zip(dataset["instruction "][:5], dataset["output "][:5])
        ]
        lengths = [len(str(i)) + len(str(o))
                   for i, o in zip(dataset["instruction "], dataset["output "])]
        stats = (f"**Total examples:** {len(dataset)}\n "
                 f"**Avg inst+output length:** {np.mean(lengths):.0f} chars " if lengths
                 else f"**Total examples:** {len(dataset)} ")
    else:
        examples = ["Unknown format "]
        stats = "Cannot preview. "
    return "\n\n---\n\n".join(examples), stats
─────────────────────────────────────────────
Training
─────────────────────────────────────────────
def preprocess_function(examples, tokenizer, max_length: int, task_type: str):
    """[BF-7] pad_token set before calling map(), not mutated here. """
    if task_type == "instruction ":
        texts = [
            f"### Instruction:\n{inst}\n\n### Response:\n{out} "
            for inst, out in zip(examples["instruction "], examples["output "])
        ]
    else:
        texts = examples["text "]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


class StopCallback(TrainerCallback):
    """[BF-3] Uses threading.Event instead of bare global bool."""
    def on_step_end(self, args, state, control, **kwargs):
        if stop_event.is_set():
            control.should_training_stop = True
        return control


class LoggingCallback(TrainerCallback):
    """Accumulates (step, loss) pairs for the live loss chart."""
    def __init__(self):
        self.records: list[dict] = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.records.append({
                "step": state.global_step,
                "train_loss": round(logs["loss"], 4),
                "eval_loss": round(logs.get("eval_loss", float("nan")), 4),
            })


def train_model(
    model_name, dataset, output_dir, hyperparams,
    device, peft_method, use_lora, lora_rank, lora_alpha,
    prefix_tuning_num_virtual_tokens, prefix_tuning_token_dim, prefix_tuning_num_layers,
    prompt_tuning_num_virtual_tokens, prompt_tuning_num_layers,
    adapter_reduction_factor,
    resume_from_checkpoint, early_stop,
    lr_scheduler_type, gradient_checkpointing,
    progress=gr.Progress(),
):
    stop_event.clear()
    log_callback = LoggingCallback()
    try:
        progress(0, desc="Loading tokenizer… ")
        
        # Handle models without EOS tokens
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Add fallbacks for missing special tokens
        if tokenizer.eos_token is None:
            if hasattr(tokenizer, "bos_token") and tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
            elif hasattr(tokenizer, "unk_token") and tokenizer.unk_token:
                tokenizer.eos_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({'eos_token': '</s>'})
                tokenizer.eos_token = '</s>'
        tokenizer.pad_token = tokenizer.eos_token
        
        task_type = (
            "instruction "
            if "instruction " in dataset.column_names and "output " in dataset.column_names
            else "lm "
        )


        progress(0.05, desc="Tokenising dataset… ")
        tokenized = dataset.map(
            lambda x: preprocess_function(x, tokenizer, hyperparams["max_length "], task_type),
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenising ",
        )


        split = tokenized.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = split["train "], split["test "]


        progress(0.1, desc="Loading model… ")
        if device == "cuda ":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4 ",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb,
                device_map="auto ",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )


        # Apply parameter-efficient fine-tuning method
        if peft_method != "Full Fine-tuning":
            progress(0.15, desc=f"Applying {peft_method}… ")
            
            if peft_method == "LoRA" or (peft_method == "Auto" and use_lora):
                targets = get_lora_targets(model_name)  # [BF-4]
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=targets,
                    lora_dropout=0.05,
                    bias="none ",
                )
                model = get_peft_model(model, lora_cfg)
                
            elif peft_method == "Prefix Tuning":
                prefix_cfg = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=prefix_tuning_num_virtual_tokens,
                    token_dim=prefix_tuning_token_dim,
                    num_transformer_layers=prefix_tuning_num_layers,
                )
                model = get_peft_model(model, prefix_cfg)
                
            elif peft_method == "Prompt Tuning":
                prompt_cfg = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=prompt_tuning_num_virtual_tokens,
                    num_transformer_layers=prompt_tuning_num_layers,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    prompt_tuning_init_text="Classify the sentiment of this review:",
                    tokenizer_name_or_path=model_name,
                )
                model = get_peft_model(model, prompt_cfg)
                
            elif peft_method == "Adapters":
                adapter_cfg = AdapterConfig(
                    non_linearity="relu",
                    reduction_factor=adapter_reduction_factor,
                    leave_out=[],
                )
                # Adapters use a different API
                model.add_adapter("default", config=adapter_cfg)
                model.train_adapter(["default"])


        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=hyperparams["epochs "],
            per_device_train_batch_size=hyperparams["batch_size "],
            per_device_eval_batch_size=hyperparams["batch_size "],
            gradient_accumulation_steps=hyperparams["grad_accum "],
            learning_rate=hyperparams["learning_rate "],
            warmup_steps=hyperparams["warmup_steps "],
            logging_steps=10,
            evaluation_strategy="steps ",   # [BF-10] plain string
            eval_steps=50,
            save_strategy="steps ",          # [BF-10]
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss ",
            greater_is_better=False,
            fp16=(device == "cuda "),
            report_to="none ",
            disable_tqdm=False,
            lr_scheduler_type=lr_scheduler_type,
            gradient_checkpointing=gradient_checkpointing,
        )


        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


        callbacks = [StopCallback(), log_callback]
        if early_stop > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stop)
            )


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )


        # [BF-5] Proper checkpoint detection via glob
        resume_path = None
        if resume_from_checkpoint:
            ckpts = sorted(
                glob.glob(os.path.join(output_dir, "checkpoint-* ")),
                key=lambda p: int(p.rsplit("-", 1)[-1]),
            )
            if ckpts:
                resume_path = ckpts[-1]


        progress(0.3, desc="Training started… ")
        t0 = time.time()
        trainer.train(resume_from_checkpoint=resume_path)


        elapsed = time.time() - t0
        status = "stopped by user " if stop_event.is_set() else "complete "


        progress(0.9, desc="Saving model… ")
        if peft_method != "Full Fine-tuning":
            model.save_pretrained(output_dir)
        else:
            model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


        # [IMP-5] Free VRAM 
        del model
        if device == "cuda ":
            torch.cuda.empty_cache()
        gc.collect()


        progress(1.0, desc="Done! ")
        summary = (
            f"✅ Training {status}!\n "
            f"⏱ Elapsed: {elapsed/60:.1f} min\n "
            f"📁 Model saved to: {output_dir}\n "
        )
        if log_callback.records:
            final = log_callback.records[-1]
            summary += f"📉 Final train loss: {final['train_loss']} "
        return summary, log_callback.records


    except Exception as e:
        raise RuntimeError(f"Training failed: {e} ")


def create_zip_from_folder(folder_path: str) -> str:
    """[BF-6] tempfile.mktemp replaced with NamedTemporaryFile."""
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(folder_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    zf.write(fpath, os.path.relpath(fpath, start=os.path.dirname(folder_path)))
        return zip_path


def create_model_card(model_name, dataset_info, hyperparams, output_dir, peft_method):
    """[IMP-11] Mentions whether LoRA or full fine-tune was used."""
    mode = peft_method if peft_method != "Full Fine-tuning" else "full fine-tune"
    card = f"""---
language: en
tags:
fine-tuned
{"lora" if peft_method == "LoRA" else "peft" if peft_method != "Full Fine-tuning" else "full-finetune"}
causal-lm
datasets:
custom


# Fine-tuned Model Card


This model is a {mode} of `{model_name}`.


## Training Data
- Examples: {dataset_info.get('num_examples', 'N/A')}
- Average length: {dataset_info.get('avg_length', 'N/A'):.0f} chars


## Hyperparameters
| Param | Value |
| --- | --- |
| Learning rate | {hyperparams.get('learning_rate')} |
| Epochs | {hyperparams.get('epochs')} |
| Batch size | {hyperparams.get('batch_size')} |
| Max length | {hyperparams.get('max_length')} |
| PEFT Method | {peft_method} |
"""
    # Add method-specific parameters
    if peft_method == "LoRA":
        card += f"| LoRA rank | {hyperparams.get('lora_rank', 'N/A')} |\n"
        card += f"| LoRA alpha | {hyperparams.get('lora_alpha', 'N/A')} |\n"
    elif peft_method == "Prefix Tuning":
        card += f"| Prefix tokens | {hyperparams.get('prefix_tuning_num_virtual_tokens', 'N/A')} |\n"
        card += f"| Token dimension | {hyperparams.get('prefix_tuning_token_dim', 'N/A')} |\n"
        card += f"| Layers | {hyperparams.get('prefix_tuning_num_layers', 'N/A')} |\n"
    elif peft_method == "Prompt Tuning":
        card += f"| Prompt tokens | {hyperparams.get('prompt_tuning_num_virtual_tokens', 'N/A')} |\n"
        card += f"| Layers | {hyperparams.get('prompt_tuning_num_layers', 'N/A')} |\n"
    elif peft_method == "Adapters":
        card += f"| Reduction factor | {hyperparams.get('adapter_reduction_factor', 'N/A')} |\n"
        
    card += f"""| LR scheduler | {hyperparams.get('lr_scheduler', 'linear')} |


Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card)
─────────────────────────────────────────────
Inference
─────────────────────────────────────────────
def _load_for_inference(model_name: str, lora_path: str | None):
    """[BF-9] Caches model+tokenizer between calls."""
    key = (model_name, lora_path)
    if key not in _inference_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Handle models without EOS tokens for inference too
        if tokenizer.eos_token is None:
            if hasattr(tokenizer, "bos_token") and tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
            elif hasattr(tokenizer, "unk_token") and tokenizer.unk_token:
                tokenizer.eos_token = tokenizer.unk_token
            else:
                tokenizer.add_special_tokens({'eos_token': '</s>'})
                tokenizer.eos_token = '</s>'
        tokenizer.pad_token = tokenizer.eos_token
        
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if lora_path and os.path.isdir(lora_path):
            model = PeftModel.from_pretrained(base, lora_path)
        else:
            model = base
        model.eval()
        _inference_cache.clear()          # keep only one model in memory
        _inference_cache[key] = (model, tokenizer)
    return _inference_cache[key]


def generate_text(
    model_name: str,
    lora_path: str | None,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    try:
        model, tokenizer = _load_for_inference(model_name, lora_path)
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        return f"❌ Generation failed: {e}"


def batch_generate(model_name: str, lora_path: str | None, prompts_file, max_new_tokens=150) -> str:
    """[PERF-1] Batch inference optimized with batch processing"""
    try:
        # Read prompts
        if prompts_file.name.endswith(".csv "):
            df = pd.read_csv(prompts_file.name)
            if "prompt " not in df.columns:
                return "CSV must have a 'prompt' column. "
            prompts = df["prompt "].tolist()
        else:
            with open(prompts_file.name, "r ", encoding="utf-8 ") as f:
                prompts = [l.strip() for l in f if l.strip()]
        
        # Process in batches for efficiency
        batch_size = min(8, len(prompts))  # Adjust based on available memory
        all_responses = []
        
        model, tokenizer = _load_for_inference(model_name, lora_path)
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_responses.extend(responses)
        
        result_df = pd.DataFrame({"prompt": prompts, "response": all_responses})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            result_df.to_csv(tmp.name, index=False)
            return tmp.name
    except Exception as e:
        return str(e)


def push_to_hub(model_path: str, repo_id: str, token: str) -> str:
    # [FIX-6] Input validation before calling the API
    if not model_path or not os.path.isdir(model_path):
        return "❌ No model found. Please train a model first. "
    if not repo_id or "/ " not in repo_id:
        return "❌ Invalid Repo ID. Format:  `username/model-name` "
    if not token or len(token) < 8:
        return "❌ Please provide a valid Hugging Face write token. "
    if not HAS_HUB:
        return "❌ huggingface_hub not installed. Run: pip install huggingface_hub "
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model ",
            token=token,
        )
        return f"✅ Pushed to https://huggingface.co/{repo_id} "
    except Exception as e:
        return f"❌ Push failed: {e} "
─────────────────────────────────────────────
Gradio event handlers
─────────────────────────────────────────────
def on_file_upload(file):
    """[BF-2] Now returns 6 outputs (added text_col). """
    if file is None:
        return (
            "No file uploaded. ",
            gr.update(visible=False, choices=[]),
            gr.update(visible=False, choices=[]),
            gr.update(visible=False, choices=[]),
            " ",  " ",
        )
    ftype = detect_file_type(file)
    if ftype is None:
        return (
            "⚠️ Unsupported file type. ",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            " ",  " ",
        )
    try:
        ds = load_dataset_from_file(file, ftype)
        ds, issues = validate_and_clean_dataset(ds)
        preview, stats = preview_dataset(ds)
        issues_txt = "\n".join(issues) if issues else "✅ No issues. "
        if ftype in ("csv ", "excel "):
            df = pd.read_csv(file.name) if ftype == "csv " else pd.read_excel(file.name)
            cols = list(df.columns)
            need_map = not (
                ("instruction " in cols and "output " in cols) or "text " in cols
            )
            if need_map:
                return (
                    f"⚠️ Ambiguous columns ({cols}). Please map them below. ",
                    gr.update(visible=True, choices=cols),
                    gr.update(visible=True, choices=cols),
                    gr.update(visible=True, choices=cols),
                    preview, stats + "\n\n" + issues_txt,
                )


        return (
            f"✅ Loaded {len(ds)} examples. ",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            preview, stats + "\n\n" + issues_txt,
        )
    except Exception as e:
        return (
            f"❌ Error: {e} ",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            " ",  " ",
        )


def on_train_click(
    file, model_choice, custom_model, training_preset, peft_method,
    use_lora, lora_rank, lora_alpha,
    prefix_tuning_num_virtual_tokens, prefix_tuning_token_dim, prefix_tuning_num_layers,
    prompt_tuning_num_virtual_tokens, prompt_tuning_num_layers,
    adapter_reduction_factor,
    lr, epochs, bs, grad_accum, max_len, warmup,
    early_stop, lr_sched, grad_ckpt, resume,
    col_inst, col_out, col_text,
    progress=gr.Progress(),
):
    stop_event.clear()
    if file is None:
        return "❌ Please upload a data file first. ", None, None, []


    model_name = custom_model.strip() if custom_model.strip() else model_choice
    device = "cuda " if torch.cuda.is_available() else "cpu "


    if device == "cuda ":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if "7b " in model_name.lower() and vram < 14:
            return (
                "❌ Model too large for your GPU VRAM.  "
                "Try a smaller model or reduce batch size. ",
                None, None, [],
            )
    elif "7b " in model_name.lower():
        return "❌ 7B models are too large for CPU. Use gpt2/opt-125m/TinyLlama. ", None, None, []


    ftype = detect_file_type(file)
    col_map = {}
    if col_inst and col_out:
        col_map[col_inst] = "instruction "
        col_map[col_out] = "output "
    elif col_text:
        col_map[col_text] = "text "


    try:
        ds = load_dataset_from_file(file, ftype, col_map)
    except Exception as e:
        return str(e), None, None, []


    ds, issues = validate_and_clean_dataset(ds)
    # Check if dataset is empty after validation
    if len(ds) == 0:
        return "❌ Dataset is empty after cleaning. Please check your data.", None, None, []
    
    issues_str = "\n".join(issues) if issues else "✅ No data issues. "


    # Apply preset  [FIX-1] strings now match the Radio widget labels exactly
    if training_preset == "Quick (1 epoch) ":
        epochs, lr = 1, 5e-4
    elif training_preset == "Balanced (3 epochs) ":
        epochs, lr = 3, 2e-4
    elif training_preset == "Accurate (5 epochs) ":
        epochs, lr = 5, 1e-4
    # "Advanced " → keep slider values as-is


    hyperparams = dict(
        learning_rate=lr, epochs=epochs, batch_size=bs,
        grad_accum=grad_accum, max_length=max_len,
        warmup_steps=warmup, lora_rank=lora_rank,
        lora_alpha=lora_alpha, lr_scheduler=lr_sched,
        prefix_tuning_num_virtual_tokens=prefix_tuning_num_virtual_tokens,
        prefix_tuning_token_dim=prefix_tuning_token_dim,
        prefix_tuning_num_layers=prefix_tuning_num_layers,
        prompt_tuning_num_virtual_tokens=prompt_tuning_num_virtual_tokens,
        prompt_tuning_num_layers=prompt_tuning_num_layers,
        adapter_reduction_factor=adapter_reduction_factor,
    )


    output_dir = tempfile.mkdtemp()
    dataset_info = {
        "num_examples ": len(ds),
        "avg_length ": (
            np.mean([len(str(t)) for t in ds["text "]])
            if "text " in ds.column_names
            else np.mean([len(str(i)) + len(str(o))
                          for i, o in zip(ds["instruction "], ds["output "])])
        ),
    }


    try:
        msg, log_records = train_model(
            model_name, ds, output_dir, hyperparams,
            device, peft_method, use_lora, lora_rank, lora_alpha,
            prefix_tuning_num_virtual_tokens, prefix_tuning_token_dim, prefix_tuning_num_layers,
            prompt_tuning_num_virtual_tokens, prompt_tuning_num_layers,
            adapter_reduction_factor,
            resume, early_stop, lr_sched, grad_ckpt,
            progress=progress,
        )
        create_model_card(model_name, dataset_info, hyperparams, output_dir, peft_method)
        zip_path = create_zip_from_folder(output_dir)
        full_msg = msg + "\n\n" + issues_str
        return full_msg, zip_path, output_dir, log_records
    except Exception as e:
        return f"❌ Training failed: {e}\n\n{issues_str} ", None, None, []


def on_stop():
    stop_event.set()
    return "🛑 Stop signal sent — will halt after the current step."


def on_generate(prompt, model_choice, custom_model, lora_path, max_tok, temp, top_p):
    model_name = custom_model.strip() if custom_model.strip() else model_choice
    return generate_text(model_name, lora_path, prompt, int(max_tok), temp, top_p)


def on_batch_test(f, model_choice, custom_model, lora_path):
    model_name = custom_model.strip() if custom_model.strip() else model_choice
    return batch_generate(model_name, lora_path, f)


def on_push(model_path, repo_id, token):
    return push_to_hub(model_path, repo_id, token)


def build_loss_chart(log_records: list):
    """[IMP-2] Build Gradio dataframe for the loss plot. """
    if not log_records:
        return pd.DataFrame(columns=["Step ", "Train Loss ", "Eval Loss "])
    return pd.DataFrame({
        "Step ":       [r["step "]       for r in log_records],
        "Train Loss ": [r["train_loss "] for r in log_records],
        "Eval Loss ":  [r["eval_loss "]  for r in log_records],
    })
─────────────────────────────────────────────
Custom CSS for the dark/purple theme
─────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Root variables ───────────────────────── */
:root {
--bg-main:    #0f0f18;
--bg-card:    #1a1a2e;
--bg-input:   #16213e;
--accent:     #7c3aed;
--accent-lt:  #a78bfa;
--accent-glow:rgba(124, 58, 237, 0.35);
--success:    #10b981;
--warn:       #f59e0b;
--danger:     #ef4444;
--text-main:  #e2e8f0;
--text-muted: #94a3b8;
--border:     #334155;
--radius:     12px;
}
/* ── Global body ───────────────────────────── */
body, .gradio-container {
background: var(--bg-main) !important;
color: var(--text-main) !important;
font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
/* ── Header banner ─────────────────────────── */
#header-banner {
background: linear-gradient(135deg, #1e0a3c 0%, #0f1e4c 50%, #0a2b4c 100%);
border: 1px solid var(--accent);
border-radius: var(--radius);
padding: 24px 32px;
margin-bottom: 20px;
box-shadow: 0 0 40px var(--accent-glow);
}
#header-banner h1 {
font-size: 2rem;
font-weight: 800;
background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
margin: 0 0 6px 0;
}
#header-banner p {
color: var(--text-muted);
margin: 0;
font-size: 0.95rem;
}
/* ── Hardware info box ─────────────────────── */
#hw-info {
background: var(--bg-card);
border: 1px solid var(--border);
border-left: 4px solid var(--accent);
border-radius: var(--radius);
padding: 14px 18px;
font-size: 0.88rem;
color: var(--text-muted);
}
/* ── Tab bar ───────────────────────────────── */
.tab-nav button {
background: var(--bg-card) !important;
color: var(--text-muted) !important;
border: 1px solid var(--border) !important;
border-radius: 8px 8px 0 0 !important;
padding: 10px 20px !important;
font-weight: 600 !important;
transition: all 0.2s;
}
.tab-nav button.selected {
background: var(--accent) !important;
color: white !important;
border-color: var(--accent) !important;
box-shadow: 0 0 12px var(--accent-glow);
}
/* ── Cards / panels ────────────────────────── */
.gr-box, .gr-form, .gr-panel,
.gradio-box, .block {
background: var(--bg-card) !important;
border: 1px solid var(--border) !important;
border-radius: var(--radius) !important;
}
/* ── Inputs ────────────────────────────────── */
input, textarea, select,
.gr-input, .gr-textarea {
background: var(--bg-input) !important;
color: var(--text-main) !important;
border: 1px solid var(--border) !important;
border-radius: 8px !important;
}
input:focus, textarea:focus {
border-color: var(--accent) !important;
box-shadow: 0 0 8px var(--accent-glow) !important;
outline: none !important;
}
/* ── Primary button ────────────────────────── */
.gr-button-primary, button[data-testid="primary"] {
background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
color: white !important;
border: none !important;
border-radius: 8px !important;
font-weight: 700 !important;
padding: 10px 24px !important;
transition: all 0.2s !important;
box-shadow: 0 4px 15px var(--accent-glow);
}
.gr-button-primary:hover {
transform: translateY(-2px);
box-shadow: 0 6px 20px var(--accent-glow) !important;
}
/* ── Stop button ───────────────────────────── */
button[data-testid="stop"], .gr-button-stop {
background: linear-gradient(135deg, #b91c1c, #7f1d1d) !important;
color: white !important;
border: none !important;
border-radius: 8px !important;
font-weight: 700 !important;
}
/* ── Secondary buttons ─────────────────────── */
button[data-testid="secondary"] {
background: var(--bg-input) !important;
color: var(--text-main) !important;
border: 1px solid var(--border) !important;
border-radius: 8px !important;
}
/* ── Accordion ─────────────────────────────── */
.gr-accordion {
border: 1px solid var(--accent) !important;
border-radius: var(--radius) !important;
}
.gr-accordion > .label-wrap {
background: rgba(124,58,237,0.15) !important;
color: var(--accent-lt) !important;
font-weight: 600 !important;
}
/* ── Labels ────────────────────────────────── */
label, .gr-label {
color: var(--text-muted) !important;
font-size: 0.85rem !important;
font-weight: 500 !important;
text-transform: uppercase;
letter-spacing: 0.05em;
}
/* ── Sliders ───────────────────────────────── */
input[type="range"] {
accent-color: var(--accent) !important;
}
/* ── Loss chart section ────────────────────── */
#loss-chart-wrap {
border: 1px solid var(--accent) !important;
border-radius: var(--radius) !important;
background: var(--bg-card) !important;
padding: 12px;
margin-top: 12px;
}
/* ── Status pill ───────────────────────────── */
.status-ok  { color: var(--success); font-weight: 700; }
.status-warn{ color: var(--warn);    font-weight: 700; }
.status-err { color: var(--danger);  font-weight: 700; }
/* ── Scrollbar ─────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 3px; }
"""
─────────────────────────────────────────────
Build the Gradio app
─────────────────────────────────────────────
recommended_model = auto_recommend_model()
with gr.Blocks(
    title="🧠 LLM Fine-Tuner v2",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:
    # ── Header ──────────────────────────────────
    gr.HTML("""
    <div id="header-banner">
      <h1>🧠 LLM Fine-Tuner v2.2</h1>
      <p>Fine-tune language models on your own data — no coding required.
        Upload data → pick a model → hit <strong>Start Training</strong>.</p>
    </div>
    """)


    # ── Hardware info bar ────────────────────────
    hw_md = gr.Markdown(get_hardware_summary(), elem_id="hw-info ")


    # ── Main tabs ────────────────────────────────
    with gr.Tabs():


        # ════════════════════════════════════════
        #  TAB 1 — DATA
        # ════════════════════════════════════════
        with gr.Tab("📂 Data "):
            gr.Markdown("### Upload your training data ")
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload File ",
                        file_types=[".csv ", ".jsonl ", ".json ", ".txt ", ".xlsx ", ".pdf "],
                    )
                    file_status = gr.Markdown("_No file loaded yet._ ")
                with gr.Column(scale=3):
                    with gr.Row():
                        col_inst = gr.Dropdown(label="→ Instruction column ", visible=False, interactive=True)
                        col_out  = gr.Dropdown(label="→ Output column ",      visible=False, interactive=True)
                        col_text = gr.Dropdown(label="→ Text column ",         visible=False, interactive=True)
                    preview_box = gr.Textbox(
                        label="Data Preview (first 5 examples) ",
                        lines=10, interactive=False,
                    )
                    stats_box = gr.Markdown("_Statistics will appear here._ ")


        # ════════════════════════════════════════
        #  TAB 2 — TRAINING
        # ════════════════════════════════════════
        with gr.Tab("🚀 Training "):
            with gr.Row():
                # ── Left column: config ──────────
                with gr.Column(scale=2):
                    gr.Markdown("### Model selection ")
                    model_choice = gr.Dropdown(
                        choices=[
                            "gpt2 ", "distilgpt2 ",
                            "facebook/opt-125m ", "facebook/opt-350m ",
                            "EleutherAI/pythia-70m ", "EleutherAI/pythia-160m ",
                            "TinyLlama/TinyLlama-1.1B-Chat-v1.0 ",
                            "mistralai/Mistral-7B-v0.1 ",
                        ],
                        value=recommended_model,
                        label="Base Model ",
                    )
                    custom_model = gr.Textbox(
                        label="Or enter any HuggingFace model ID ",
                        placeholder="e.g., meta-llama/Llama-2-7b-hf ",
                    )
                    model_info_md = gr.Markdown(get_model_info(recommended_model))


                    gr.Markdown("### Parameter-efficient method ")
                    peft_method = gr.Radio(
                        choices=["Full Fine-tuning", "Auto", "LoRA", "Prefix Tuning", "Prompt Tuning", "Adapters"],
                        value="Auto",
                        label="",
                        info="Auto selects LoRA for most models; Full Fine-tuning uses more VRAM"
                    )


                    gr.Markdown("### Training preset ")
                    training_preset = gr.Radio(
                        choices=["Quick (1 epoch) ", "Balanced (3 epochs) ", "Accurate (5 epochs) ", "Advanced "],
                        value="Balanced (3 epochs) ",
                        label=" ",
                    )


                    with gr.Accordion("⚙️ Advanced hyperparameters ", open=False):
                        # Common PEFT settings
                        with gr.Group():
                            gr.Markdown("### PEFT Method Settings")
                            with gr.Tab("LoRA"):
                                use_lora = gr.Checkbox(label="Enable LoRA", value=True)
                                lora_rank = gr.Slider(1, 64,   value=8,   step=1,  label="LoRA Rank", 
                                                    info="Higher values capture more model aspects but use more memory. Default 8 is usually sufficient.")
                                lora_alpha = gr.Slider(1, 128,  value=16,  step=1,  label="LoRA Alpha", 
                                                     info="Scaling factor for LoRA updates. Typically set to 2x the rank (e.g., 16 for rank 8).")
                            
                            with gr.Tab("Prefix Tuning"):
                                prefix_tuning_num_virtual_tokens = gr.Slider(10, 100, value=30, step=5, label="Virtual Tokens", 
                                                                          info="Number of prefix tokens to insert")
                                prefix_tuning_token_dim = gr.Slider(100, 1024, value=512, step=64, label="Token Dimension", 
                                                                 info="Dimension of the prefix embeddings")
                                prefix_tuning_num_layers = gr.Slider(1, 32, value=2, step=1, label="Layers", 
                                                               info="Number of transformer layers to apply prefix to")
                            
                            with gr.Tab("Prompt Tuning"):
                                prompt_tuning_num_virtual_tokens = gr.Slider(10, 100, value=20, step=5, label="Virtual Tokens", 
                                                                          info="Number of prompt tokens to insert")
                                prompt_tuning_num_layers = gr.Slider(1, 32, value=1, step=1, label="Layers", 
                                                               info="Number of transformer layers to apply prompt to")
                            
                            with gr.Tab("Adapters"):
                                adapter_reduction_factor = gr.Slider(2, 64, value=16, step=2, label="Reduction Factor", 
                                                                  info="How much to reduce the hidden dimension (e.g., 16 means 1/16 of original)")


                        # Common training settings
                        lr = gr.Number(value=2e-4, label="Learning Rate ",   precision=6)
                        epochs = gr.Slider(1, 20,   value=3,   step=1,  label="Epochs ")
                        bs = gr.Slider(1, 16,   value=2,   step=1,  label="Batch Size ")
                        grad_accum = gr.Slider(1, 16,   value=4,   step=1,  label="Gradient Accumulation Steps ")
                        max_len = gr.Slider(64, 2048, value=256, step=64, label="Max Sequence Length ")
                        warmup = gr.Slider(0, 500,  value=100, step=10, label="Warmup Steps ")
                        early_stop = gr.Slider(0, 10,   value=3,   step=1,  label="Early Stopping Patience (0 = off) ")
                        lr_sched = gr.Dropdown(
                            choices=["linear ", "cosine ", "cosine_with_restarts ", "constant "],
                            value="cosine ", label="LR Scheduler ",
                        )
                        grad_ckpt = gr.Checkbox(label="Gradient Checkpointing (saves VRAM, ~20% slower) ", value=False)
                        resume_ckpt = gr.Checkbox(label="Resume from last checkpoint ", value=False)


                    with gr.Row():
                        train_btn = gr.Button("▶  Start Training ", variant="primary ", scale=3)
                        stop_btn  = gr.Button("⏹  Stop ",           variant="stop ",    scale=1)


                # ── Right column: logs + chart ───
                with gr.Column(scale=3):
                    gr.Markdown("### Training log ")
                    log_output = gr.Textbox(
                        label=" ", lines=14, interactive=False,
                        placeholder="Training output will appear here… ",
                    )
                    with gr.Column(elem_id="loss-chart-wrap "):
                        gr.Markdown("### 📉 Loss Curve ")
                        loss_df    = gr.Dataframe(
                            headers=["Step ", "Train Loss ", "Eval Loss "],
                            datatype=["number ", "number ", "number "],
                            label=" ", interactive=False,
                        )
                    clear_gpu_btn = gr.Button("🧹 Clear GPU Cache ", variant="secondary ")


            model_path_state = gr.State()
            log_records_state = gr.State([])


        # ════════════════════════════════════════
        #  TAB 3 — INFERENCE
        # ════════════════════════════════════════
        with gr.Tab("💬 Inference "):
            gr.Markdown("### Test your fine-tuned model ")
            with gr.Row():
                with gr.Column():
                    infer_model = gr.Dropdown(
                        choices=["gpt2 ", "distilgpt2 ", "facebook/opt-125m ",
                                 "TinyLlama/TinyLlama-1.1B-Chat-v1.0 ",
                                 "mistralai/Mistral-7B-v0.1 "],
                        value="gpt2 ", label="Model ",
                    )
                    infer_custom = gr.Textbox(label="Or custom model ID ", placeholder="username/my-model ")
                    lora_path    = gr.Textbox(label="PEFT adapter path (auto-filled after training) ", interactive=False)
                    prompt_in    = gr.Textbox(label="Prompt ", lines=4, placeholder="Enter your prompt here… ")
                    with gr.Row():
                        max_tok  = gr.Slider(10, 500,  value=200,  step=10,  label="Max new tokens ")
                        temp     = gr.Slider(0.1, 2.0, value=0.7,  step=0.1, label="Temperature ")
                        top_p    = gr.Slider(0.1, 1.0, value=0.9,  step=0.05, label="Top-p ")
                    gen_btn  = gr.Button("Generate ✨ ", variant="primary ")
                with gr.Column():
                    gen_out  = gr.Textbox(label="Model response ", lines=12, interactive=False)


            gr.Markdown("### Batch inference ")
            with gr.Row():
                batch_file = gr.File(label="Upload prompts (CSV with 'prompt' col, or .txt one per line) ")
                batch_btn  = gr.Button("Run Batch ", variant="secondary ")
            batch_out = gr.File(label="Download responses CSV ")


            gr.Markdown("### Load a saved PEFT adapter ")
            with gr.Row():
                lora_zip_upload = gr.File(label="Upload PEFT ZIP (previously downloaded model) ", file_types=[".zip "])
                lora_zip_status = gr.Markdown("_Upload a ZIP to restore a fine-tuned adapter._ ")
            lora_zip_dir_state = gr.State(" ")


        # ════════════════════════════════════════
        #  TAB 4 — EXPORT
        # ════════════════════════════════════════
        with gr.Tab("📦 Export "):
            gr.Markdown("### Download your model ")
            download_btn = gr.File(label="Model ZIP (available after training) ", visible=True)


            gr.Markdown("### Push to Hugging Face Hub ")
            with gr.Row():
                repo_id   = gr.Textbox(label="Repo ID ", placeholder="username/my-finetuned-model ")
                hf_token  = gr.Textbox(label="HF Token (write access) ", type="password ")
            push_btn    = gr.Button("🚀 Push to Hub ", variant="primary ")
            push_status = gr.Markdown(" ")


# ─────────────────────────────────────────
#  Wire up events
# ─────────────────────────────────────────


# File upload
file_input.change(
    fn=on_file_upload,
    inputs=file_input,
    outputs=[file_status, col_inst, col_out, col_text, preview_box, stats_box],
)


# Model info refresh
def refresh_model_info(mid, custom):
    name = custom.strip() if custom.strip() else mid
    return get_model_info(name)


model_choice.change(refresh_model_info, [model_choice, custom_model], model_info_md)
custom_model.change(refresh_model_info, [model_choice, custom_model], model_info_md)


# Training — disable Start button while running to prevent concurrent sessions [FIX-4]
train_btn.click(
    fn=lambda: gr.update(interactive=False, value="⏳ Training… "),
    outputs=train_btn,
).then(
    fn=on_train_click,
    inputs=[
        file_input, model_choice, custom_model, training_preset, peft_method,
        use_lora, lora_rank, lora_alpha,
        prefix_tuning_num_virtual_tokens, prefix_tuning_token_dim, prefix_tuning_num_layers,
        prompt_tuning_num_virtual_tokens, prompt_tuning_num_layers,
        adapter_reduction_factor,
        lr, epochs, bs, grad_accum, max_len, warmup,
        early_stop, lr_sched, grad_ckpt, resume_ckpt,
        col_inst, col_out, col_text,
    ],
    outputs=[log_output, download_btn, model_path_state, log_records_state],
).then(
    fn=lambda: gr.update(interactive=True, value="▶  Start Training "),
    outputs=train_btn,
)


# Update loss chart when records change
log_records_state.change(
    fn=build_loss_chart,
    inputs=log_records_state,
    outputs=loss_df,
)


# [FIX-5] Auto-fill lora_path AND inference model name after training
def _sync_inference(out_dir, model_dropdown, custom_id):
    path = out_dir or " "
    name = custom_id.strip() if custom_id.strip() else model_dropdown
    return path, name


model_path_state.change(
    fn=_sync_inference,
    inputs=[model_path_state, model_choice, custom_model],
    outputs=[lora_path, infer_custom],
)


stop_btn.click(fn=on_stop, outputs=log_output)


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        free = torch.cuda.memory_reserved(0) / 1e9
        return f"🧹 GPU cache cleared. Reserved: {free:.2f} GB "
    return "ℹ️ No GPU detected. "


clear_gpu_btn.click(fn=clear_gpu_cache, outputs=log_output)


gen_btn.click(
    fn=on_generate,
    inputs=[prompt_in, infer_model, infer_custom, lora_path, max_tok, temp, top_p],
    outputs=gen_out,
)


batch_btn.click(
    fn=on_batch_test,
    inputs=[batch_file, infer_model, infer_custom, lora_path],
    outputs=batch_out,
)


push_btn.click(
    fn=on_push,
    inputs=[model_path_state, repo_id, hf_token],
    outputs=push_status,
)


# [NEW-4] PEFT zip upload: extract and set as lora_path for inference
def on_peft_zip_upload(zip_file):
    if zip_file is None:
        return " ", "No file uploaded. ", " "
    try:
        extract_dir = tempfile.mkdtemp(prefix="peft_zip_ ")
        safe_extract_zip(zip_file.name, extract_dir)  # Use secure extraction
        
        # Find the directory containing adapter_config.json or equivalent
        adapter_dir = extract_dir
        for root, dirs, files in os.walk(extract_dir):
            if "adapter_config.json" in files or "adapter_model.bin" in files or "pytorch_model.bin" in files:
                adapter_dir = root
                break
        return adapter_dir, f"✅ PEFT adapter extracted to: `{adapter_dir}` ", adapter_dir
    except Exception as e:
        return " ", f"❌ Failed to extract ZIP: {e} ", " "


lora_zip_upload.change(
    fn=on_peft_zip_upload,
    inputs=lora_zip_upload,
    outputs=[lora_path, lora_zip_status, lora_zip_dir_state],
)
─────────────────────────────────────────────
Launch
─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
