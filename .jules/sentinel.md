## 2025-05-15 - [Credential Validation & XSS Prevention]
**Vulnerability:** Weak HuggingFace token validation in the Model Registry and incomplete HTML escaping in the evaluation preview.
**Learning:** Legacy components (`export/registry.py`) often lag behind more recently hardened ones (`export/hub.py`) in credential validation logic. Additionally, manual HTML escaping functions frequently miss the single quote (`'`), which is essential for preventing XSS in attribute-based injection scenarios.
**Prevention:** Centralize sensitive credential format constants (like `HF_TOKEN_PREFIX`) to ensure uniform validation across all entry points. Always include `&#x27;` in custom HTML escaping helpers, following OWASP standards.

## 2026-05-17 - [Input Length Validation for DoS Mitigation]
**Vulnerability:** Denial of Service (DoS) risk due to unsanitized input lengths in UI Textboxes and missing truncation in tokenizer calls.
**Learning:** Even with model-level truncation, extremely large raw string inputs can consume excessive CPU/Memory during initial processing or UI rendering. Defensive programming should enforce limits at both the UI entry point (`max_length`) and the first point of backend processing (`truncation=True`).
**Prevention:** Always enforce `max_length` on Gradio Textbox components and use explicit `truncation=True` in all tokenizer calls that handle potentially unbounded user input.

## 2025-05-18 - [Path Traversal & Identifier Hardening]
**Vulnerability:** Missing whitespace stripping and path traversal validation in Hub push, Registry, and Adapter Merge handlers.
**Learning:** High-level UI handlers (Layer 4/5) often assume sanitized input from the frontend, but must independently enforce security boundaries to prevent path traversal via '..' or '\' and logic errors caused by leading/trailing whitespace in identifiers.
**Prevention:** Always strip whitespace from Repo IDs, tokens, and version tags at the handler entry point. Enforce validation against '..' and '\' for all user-provided strings that are used to construct local paths or remote repository names.

## 2025-05-20 - [Immediate Resource Tracking for DoS Prevention]
**Vulnerability:** Potential disk exhaustion (DoS) due to untracked temporary training directories when training fails or is stopped early.
**Learning:** High-level handlers (`ui/handlers.py`) that create temporary filesystem resources must register them with the global state *immediately* after creation. Deferring registration until the end of a complex function creates a leak window where an exception or early return prevents the resource from ever being cleaned up in subsequent runs.
**Prevention:** Always assign temporary paths to state-tracked attributes (like `app_state._last_model_dir`) immediately after `tempfile.mkdtemp()` or similar calls, ensuring they are eligible for the next run's cleanup routine regardless of the current run's outcome.
