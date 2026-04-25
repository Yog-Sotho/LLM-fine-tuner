"""
setup.py — LLM Fine-Tuner v3.2
================================
Packaging shim that works alongside pyproject.toml (PEP 517/518).

pyproject.toml is the canonical source of truth for project metadata.
This file exists as a compatibility layer so that:

  • ``pip install .`` works even on pip < 21.3 (pre-PEP-660)
  • ``python setup.py develop`` works in environments that need the legacy
    editable install path (some Docker layers, older tools)
  • ``python -m build`` produces an identical wheel regardless of whether
    it reads from pyproject.toml or setup.py (both are present, setuptools
    prefers pyproject.toml when build-backend is specified)

All metadata here is kept byte-for-byte consistent with pyproject.toml.
If you change one, change the other.

Usage
-----
Standard install (end users):
    pip install .

Editable install (developers):
    pip install -e ".[dev]"

With all optional extras:
    pip install -e ".[eval,quant,vllm,heretic,dev]"

Build a wheel + sdist:
    python -m build

Verify the built artefacts:
    twine check dist/*
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from setuptools import find_packages, setup

# ── Minimum Python version guard ──────────────────────────────────────────────
# Checked here (at setup time) so the error is readable even when pip's own
# python_requires check doesn't surface a helpful message.
_REQUIRED = (3, 10)
if sys.version_info < _REQUIRED:
    raise SystemExit(
        f"LLM Fine-Tuner requires Python {_REQUIRED[0]}.{_REQUIRED[1]} or later.\n"
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        "Please upgrade: https://www.python.org/downloads/"
    )

HERE = Path(__file__).parent.resolve()


# ── Version ────────────────────────────────────────────────────────────────────
# The authoritative version lives in pyproject.toml [project] version.
# config/constants.py does NOT define __version__, so we read from pyproject.toml.
def _get_version() -> str:
    toml = HERE / "pyproject.toml"
    if toml.is_file():
        m = re.search(r'^version\s*=\s*"([^"]+)"', toml.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            return m.group(1)
    return "3.2.0.dev0"


# ── Long description ───────────────────────────────────────────────────────────
def _read_readme() -> tuple[str, str]:
    readme = HERE / "README.md"
    if readme.is_file():
        return readme.read_text(encoding="utf-8"), "text/markdown"
    return "", "text/plain"


_version = _get_version()
_long_description, _long_desc_content_type = _read_readme()

# ── Core runtime dependencies ──────────────────────────────────────────────────
# Mirrors [project.dependencies] in pyproject.toml exactly.
# torch is listed without a CUDA variant — the smart installer (install.sh) and
# the Dockerfiles choose the correct whl index per-machine.
INSTALL_REQUIRES: list[str] = [
    "gradio>=5.0.0",
    "transformers>=4.48.0",
    "datasets>=3.0.0",
    "peft>=0.14.0",
    "accelerate>=1.3.0",
    "bitsandbytes>=0.45.0",
    "trl>=0.8.0",
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "torchaudio>=2.5.0",
    "numpy>=2.0.0",
    "pandas>=2.2.0",
    "safetensors>=0.4.3",
    "tqdm>=4.66.0",
    "einops>=0.8.0",
    "matplotlib>=3.9.0",
    "huggingface_hub>=0.25.0",
    "typer>=0.15.0",
    "PyPDF2>=3.0.0",
    "openpyxl>=3.1.0",
    "psutil>=6.0.0",
    # heretic-llm intentionally NOT listed here — see [heretic] extra.
    # H-11 FIX: moving it to optional prevents install failures when the
    # package is unavailable on a given PyPI mirror.
]

# ── Optional dependency groups ─────────────────────────────────────────────────
# Mirrors [project.optional-dependencies] in pyproject.toml exactly.
EXTRAS_REQUIRE: dict[str, list[str]] = {
    # Evaluation metrics (BLEU, ROUGE, BERTScore, HF evaluate hub)
    "eval": [
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.13",
        "nltk>=3.8.0",
        "nlpaug>=1.1.10",
    ],
    # Quantised inference backends
    "quant": [
        "auto-gptq>=0.7.1",
        "exllamav2",
    ],
    # High-throughput inference (CUDA required at runtime)
    "vllm": [
        "vllm>=0.2.0",
    ],
    # H-11 FIX: heretic-llm isolated so a missing package never breaks a
    # standard install. Install with: pip install "llm-fine-tuner[heretic]"
    "heretic": [
        "heretic-llm>=1.2.0",
    ],
    # Developer / CI toolchain (matches pyproject.toml dev group)
    "dev": [
        "black>=24.0.0",
        "pytest>=8.0.0",
        "pytest-cov>=5.0.0",
        "pytest-timeout>=2.3.0",
        "ruff>=0.4.0",
        "build>=1.2.0",
        "twine>=5.0.0",
        "pip-audit>=2.7.0",
    ],
    # Documentation (MkDocs + Material theme)
    "docs": [
        "mkdocs>=1.6.0",
        "mkdocs-material>=9.5.0",
        "mkdocstrings[python]>=0.25.0",
    ],
}

# "all" convenience group — includes every extra except build-heavy ones
# (flash-attn, unsloth, vllm, auto-gptq, exllamav2) that require special
# CUDA compilation steps documented in install.sh and the Dockerfiles.
EXTRAS_REQUIRE["all"] = sorted(
    set(
        pkg
        for key, pkgs in EXTRAS_REQUIRE.items()
        if key not in {"quant", "vllm", "dev", "docs"}
        for pkg in pkgs
    )
)

# ── Package discovery ──────────────────────────────────────────────────────────
# All packages live flat at the repo root (no src/ layout) matching the
# [tool.setuptools.packages.find] where = ["."] in pyproject.toml.
# Excluded:
#   archive/ — original monolith, not importable
#   tests/   — test suite, not distributed
#   docs/    — documentation sources, not distributed
PACKAGES = find_packages(
    where=".",
    exclude=["archive", "archive.*", "tests", "tests.*", "docs", "docs.*"],
)

# ── Entry points ───────────────────────────────────────────────────────────────
# Mirrors [project.scripts] in pyproject.toml exactly.
# After `pip install .`:
#   $ llm-finetune          → launches Gradio UI (no args) or CLI (any arg)
#   $ app                   → same binary (legacy Vercel / platform entry point)
ENTRY_POINTS: dict[str, list[str]] = {
    "console_scripts": [
        "llm-finetune = main:main",
        "app = main:main",
    ],
}

# ── Classifiers ───────────────────────────────────────────────────────────────
# Mirrors [project.classifiers] in pyproject.toml + additional detail.
CLASSIFIERS: list[str] = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Environment :: GPU :: NVIDIA CUDA",
    "Environment :: Web Environment",
    "Typing :: Typed",
]

# ── Project URLs ───────────────────────────────────────────────────────────────
# Mirrors [project.urls] in pyproject.toml.
PROJECT_URLS: dict[str, str] = {
    "Homepage":    "https://github.com/Yog-Sotho/LLM-fine-tuner",
    "Repository":  "https://github.com/Yog-Sotho/LLM-fine-tuner",
    "Issues":      "https://github.com/Yog-Sotho/LLM-fine-tuner/issues",
    "Changelog":   "https://github.com/Yog-Sotho/LLM-fine-tuner/releases",
}

# ── setup() ────────────────────────────────────────────────────────────────────
setup(
    # ── Identity ──────────────────────────────────────────────────────────
    name="llm-fine-tuner",                          # matches pyproject.toml name
    version=_version,                               # read from pyproject.toml
    description=(
        "Advanced LLM Fine-Tuner — SFT, DPO, ORPO, PPO, reward modelling and evaluation. "
        "No-code Gradio UI + full CLI. QLoRA, Unsloth, GGUF, vLLM, Heretic Mode."
    ),
    long_description=_long_description,
    long_description_content_type=_long_desc_content_type,

    # ── Author ────────────────────────────────────────────────────────────
    author="Yog-Sotho",
    author_email="",
    maintainer="Yog-Sotho",
    maintainer_email="",

    # ── URLs ──────────────────────────────────────────────────────────────
    url="https://github.com/Yog-Sotho/LLM-fine-tuner",
    project_urls=PROJECT_URLS,

    # ── License ───────────────────────────────────────────────────────────
    license="MIT",                                  # matches pyproject.toml

    # ── Python version ────────────────────────────────────────────────────
    python_requires=">=3.10",                       # matches pyproject.toml

    # ── Packages ──────────────────────────────────────────────────────────
    packages=PACKAGES,
    package_dir={"": "."},                          # flat layout, no src/

    # ── Static assets bundled into the wheel ──────────────────────────────
    package_data={
        # Include CSS so ui.css is accessible at runtime via importlib.resources
        "ui":     ["*.py", "css.py"],
        # Include docs index so help text can be read from the package
        "docs":   ["index.md"],
        # Include all Python sources for the config layer (needed by some
        # tools that inspect the installed package for constants)
        "config": ["*.py"],
    },
    include_package_data=True,

    # ── Dependencies ──────────────────────────────────────────────────────
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    # ── Entry points ──────────────────────────────────────────────────────
    entry_points=ENTRY_POINTS,

    # ── Metadata ──────────────────────────────────────────────────────────
    classifiers=CLASSIFIERS,
    keywords=[
        "llm", "fine-tuning", "machine-learning", "deep-learning",
        "gradio", "peft", "lora", "qlora", "dpo", "rlhf", "ppo", "orpo",
        "transformers", "huggingface", "unsloth", "gguf", "vllm",
        "nlp", "pytorch", "ai",
    ],
    platforms=["any"],

    # Must be False: gradio, transformers, and huggingface_hub all read their
    # own package resources at runtime (templates, tokeniser files, etc.).
    zip_safe=False,
)
