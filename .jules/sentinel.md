## 2025-05-15 - [Credential Validation & XSS Prevention]
**Vulnerability:** Weak HuggingFace token validation in the Model Registry and incomplete HTML escaping in the evaluation preview.
**Learning:** Legacy components (`export/registry.py`) often lag behind more recently hardened ones (`export/hub.py`) in credential validation logic. Additionally, manual HTML escaping functions frequently miss the single quote (`'`), which is essential for preventing XSS in attribute-based injection scenarios.
**Prevention:** Centralize sensitive credential format constants (like `HF_TOKEN_PREFIX`) to ensure uniform validation across all entry points. Always include `&#x27;` in custom HTML escaping helpers, following OWASP standards.

## 2026-05-17 - [Input Length Validation for DoS Mitigation]
**Vulnerability:** Denial of Service (DoS) risk due to unsanitized input lengths in UI Textboxes and missing tokenizer-level truncation.
**Learning:** Even with model-level truncation, extremely large raw string inputs can consume excessive CPU/Memory during initial processing or UI rendering. Defensive programming should enforce limits at both the UI entry point (`max_length`) and the first point of backend processing (`truncation=True`).
**Prevention:** Always enforce `max_length` on Gradio Textbox components and use explicit `truncation=True` in all tokenizer calls that handle potentially unbounded user input.

## 2025-05-18 - [Path Traversal & Identifier Hardening]
**Vulnerability:** Missing whitespace stripping and path traversal validation in Hub push, Registry, and Adapter Merge handlers.
**Learning:** High-level UI handlers (Layer 4/5) often assume sanitized input from the frontend, but must independently enforce security boundaries to prevent path traversal via '..' or '\' and logic errors caused by leading/trailing whitespace in identifiers.
**Prevention:** Always strip whitespace from Repo IDs, tokens, and version tags at the handler entry point. Enforce validation against '..' and '\' for all user-provided strings that are used to construct local paths or remote repository names.

## 2025-05-20 - [Handler-Level Path Traversal Hardening]
**Vulnerability:** Missing path traversal validation in GGUF Export and vLLM generation handlers.
**Learning:** Even when UI components are marked as non-interactive (e.g., auto-filled paths), handlers must independently validate inputs to prevent exploitation via direct backend calls or future UI changes. Standardizing on `validate_path_traversal` and `.strip()` across all entry points ensures defense-in-depth.
**Prevention:** Explicitly validate all user-provided or state-derived paths in Gradio handlers before passing them to low-level filesystem or subprocess operations.

## 2025-05-22 - [Quantization Parameter Hardening]
**Vulnerability:** User-controlled quantization strings used in GGUF/vLLM path construction without validation.
**Learning:** Security guards often focus on "path" arguments but miss secondary string parameters (like quantization types) that are interpolated into filenames. These can be exploited for path traversal or arbitrary directory creation if not sanitized.
**Prevention:** Always strip whitespace and validate ALL user-provided strings that contribute to local file paths, even if they aren't "paths" themselves. Specifically, block forward slashes in such parameters to prevent sub-directory injection.
