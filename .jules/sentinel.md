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

## 2025-05-22 - [Insufficient Path Traversal Validation for Joins]
**Vulnerability:** `validate_path_traversal` only blocks `..` and `\`, allowing forward slashes `/` to bypass security when used in `os.path.join`.
**Learning:** In identifiers that are used to construct filenames (like quantization types), a leading or embedded forward slash can be used to write files to arbitrary locations or create unintended subdirectories, even if `..` is blocked.
**Prevention:** For parameters that should be simple identifiers and not paths, explicitly block forward slashes `/` in addition to calling `validate_path_traversal`.

## 2025-05-24 - [Null Byte Injection & Centralized Identifier Validation]
**Vulnerability:** `validate_path_traversal` lacked null byte (`\0`) protection, and identifier-based parameters (like version tags or quantization types) were inconsistently validated.
**Learning:** Null byte injection can bypass some string-based path checks depending on the underlying OS/filesystem API. Furthermore, parameters that are used to construct filenames but are not meant to be paths should be strictly validated as identifiers to prevent any directory separator injection.
**Prevention:** Centralize identifier validation in `validate_identifier` to block `/`, `\\`, `..`, and `\0`. Always include `\0` in path traversal guards to ensure defense-in-depth against legacy injection techniques.

## 2025-05-26 - [SFT Hardening & Batch Inference DoS Protection]
**Vulnerability:** Missing path traversal validation in the core SFT training pipeline and untracked temporary files in batch inference.
**Learning:** Security guards at the UI layer (Layer 5) are insufficient if the underlying logic (Layer 3) is exposed via other interfaces like CLI or used as a library. Furthermore, temporary artifacts like batch generation CSVs can lead to disk exhaustion DoS if not explicitly tracked and cleaned up via the application's resource manager.
**Prevention:** Always enforce path traversal validation at the earliest possible entry point in core logic modules. Register all temporary file creation with the global `AppState` resource tracker to ensure deterministic cleanup in long-running sessions.
