"""
tests/conftest.py
==================
pytest session-scoped configuration.

M-8 FIX: All test modules use bare absolute imports (``from cli.commands import
app``, ``from data.loader import ...``, etc.) that resolve correctly only when
the repository root is on ``sys.path``.  Running ``pytest`` from a non-root
working directory, inside a virtual environment, or via a CI runner that sets
a different root dir caused ``ModuleNotFoundError`` on every test file.

This conftest.py inserts the repository root at the front of ``sys.path``
unconditionally, mirroring what ``python -m pytest`` does but making it
explicit so it works regardless of how pytest is invoked.

No test logic lives here — this file is intentionally minimal.
"""

import sys
from pathlib import Path

# Insert the repository root (the directory that contains both the source
# packages and this tests/ directory) at position 0 so it takes priority
# over any installed copies of the same package names.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
