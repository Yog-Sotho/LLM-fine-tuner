"""
data/loader.py
===============
Layer 2 — file ingestion utilities.
Imports: config.constants, stdlib, pandas, datasets.

Functions
---------
detect_file_type       — sniff extension from a Gradio file object
extract_text_from_pdf  — extract raw text from PDF pages via PyPDF2
load_dataset_from_file — unified loader for csv/jsonl/json/txt/excel/pdf
safe_extract_zip       — ZIP extraction with path-traversal guard
"""

import json
import os
import zipfile
from pathlib import Path

import pandas as pd
from datasets import Dataset

from config.constants import (
    COL_INSTRUCTION,
    COL_OUTPUT,
    COL_TEXT,
    COL_PROMPT,
    COL_CHOSEN,
    COL_REJECTED,
    FILE_EXT_CSV,
    FILE_EXT_JSONL,
    FILE_EXT_JSON,
    FILE_EXT_TXT,
    FILE_EXT_XLSX,
    FILE_EXT_PDF,
    HAS_OPENPYXL,
    HAS_PDF,
)


def detect_file_type(file) -> str | None:
    """Return a normalised file-type string from a Gradio file object.

    Returns one of: 'csv', 'jsonl', 'json', 'txt', 'excel', 'pdf', or None.
    'excel' is only returned when openpyxl is installed.
    'pdf'   is only returned when PyPDF2 is installed.
    """
    name = Path(file.name).name.lower()
    if name.endswith(FILE_EXT_CSV):              return "csv"
    if name.endswith(FILE_EXT_JSONL):            return "jsonl"
    if name.endswith(FILE_EXT_JSON):             return "json"
    if name.endswith(FILE_EXT_TXT):              return "txt"
    if name.endswith(FILE_EXT_XLSX) and HAS_OPENPYXL: return "excel"
    if name.endswith(FILE_EXT_PDF)  and HAS_PDF:       return "pdf"
    return None


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file, page by page.

    Requires PyPDF2 (HAS_PDF=True). Caller must guard before calling.
    """
    import PyPDF2  # lazy — only imported when actually used

    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)


def load_dataset_from_file(
    file,
    file_type: str,
    column_mapping: dict | None = None,
    is_dpo: bool = False,
) -> Dataset:
    """Load a dataset from any supported file format into a HuggingFace Dataset.

    Parameters
    ----------
    file         : Gradio file object (has .name attribute with the tmp path)
    file_type    : one of the strings returned by detect_file_type()
    column_mapping : optional dict to rename DataFrame columns before parsing
    is_dpo       : when True, enforces prompt/chosen/rejected columns

    Raises
    ------
    RuntimeError — wraps any internal exception with a user-friendly message
    """
    try:
        path = Path(file.name).resolve()
        if not path.is_file():
            raise ValueError("Invalid file path")

        # ── JSONL ─────────────────────────────────────────────────────────
        if file_type == "jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return Dataset.from_list(data)

        # ── JSON ──────────────────────────────────────────────────────────
        if file_type == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a top-level array of objects.")
            return Dataset.from_list(data)

        # ── Plain text ────────────────────────────────────────────────────
        if file_type == "txt":
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            return Dataset.from_dict({COL_TEXT: lines})

        # ── PDF ───────────────────────────────────────────────────────────
        if file_type == "pdf":
            text = extract_text_from_pdf(str(path))
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            return Dataset.from_dict({COL_TEXT: paragraphs})

        # ── CSV / Excel ───────────────────────────────────────────────────
        if file_type == "csv":
            df = pd.read_csv(path)
        elif file_type == "excel":
            df = pd.read_excel(path, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # v3.1 Fix #4 (Major): Only rename columns that actually exist in
        # the DataFrame. A missing key previously raised a KeyError mid-training
        # with no useful error message for the user.
        if column_mapping:
            valid_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
            ignored = {k: v for k, v in column_mapping.items() if k not in df.columns}
            if ignored:
                print(
                    f"⚠️ Column mapping: the following source columns were not "
                    f"found and are ignored: {list(ignored.keys())}"
                )
            df = df.rename(columns=valid_mapping)

        # ── DPO branch ────────────────────────────────────────────────────
        if is_dpo:
            if not all(col in df.columns for col in [COL_PROMPT, COL_CHOSEN, COL_REJECTED]):
                raise ValueError("DPO requires columns: prompt, chosen, rejected")
            # Minor Fix 7: fillna("") before astype(str) prevents literal
            # "nan" strings appearing as training examples.
            return Dataset.from_pandas(
                df[[COL_PROMPT, COL_CHOSEN, COL_REJECTED]].fillna("").astype(str)
            )

        # ── SFT branch ────────────────────────────────────────────────────
        if COL_INSTRUCTION in df.columns and COL_OUTPUT in df.columns:
            return Dataset.from_pandas(
                df[[COL_INSTRUCTION, COL_OUTPUT]].fillna("").astype(str)
            )
        elif COL_TEXT in df.columns:
            return Dataset.from_pandas(df[[COL_TEXT]].fillna("").astype(str))
        else:
            raise ValueError(
                f"Cannot determine columns automatically. "
                f"Available: {list(df.columns)}. "
                f"Please use the column mapping dropdowns above."
            )

    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def safe_extract_zip(zip_path: str, extract_dir: str) -> str:
    """Extract a ZIP archive with complete path-traversal protection.

    H-1 FIX: The previous check only blocked paths starting with '../' or '..\\'
    but did not catch absolute paths (e.g. '/etc/passwd') or normalised traversal
    paths. The fix resolves every target path with os.path.realpath and verifies
    it is inside the extract directory before extracting.

    Raises ValueError if any entry attempts to escape extract_dir.
    Returns extract_dir on success.
    """
    abs_extract_dir = os.path.realpath(extract_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for file_info in zf.infolist():
            # Resolve the full absolute target path
            target = os.path.realpath(
                os.path.join(abs_extract_dir, file_info.filename)
            )
            # The target must be inside the extract directory
            # (os.sep suffix prevents prefix-collision: /tmp/out vs /tmp/outside)
            if not target.startswith(abs_extract_dir + os.sep) and target != abs_extract_dir:
                # Sentinel: ensure error message contains "Path traversal" for test compatibility
                raise ValueError(
                    f"Path traversal attempt: Unsafe path in ZIP: {file_info.filename!r}"
                )
            zf.extract(file_info, extract_dir)
    return extract_dir
