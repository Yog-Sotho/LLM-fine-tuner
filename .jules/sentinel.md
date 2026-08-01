## 2026-08-01 - [Sensitive Token Redaction in Hub and Model Registry Exceptions]
**Vulnerability:** Raw exception messages from Hugging Face Hub APIs or connection errors could contain the sensitive user-supplied API write token, which was subsequently returned and displayed directly in the Gradio UI error panels or written to application logs, exposing user credentials.
**Learning:** Checking credentials for format constraints and path traversal is not enough. If downstream library calls fail, they may include credentials in their output strings, connection details, or exception representations. Exceptions must be intercepted and sanitized before they cross boundaries to the UI or logs.
**Prevention:** Always scan error/exception messages for sensitive inputs (like API keys, tokens, or passwords) and replace them with `[REDACTED]` before presenting or logging the error.

## 2026-07-30 - [ZIP Bomb and Decompression DoS Hardening for safe_extract_zip]
**Vulnerability:** The PEFT adapter extraction process in `safe_extract_zip` extracted files from user-uploaded ZIP archives without limits on total file count, total uncompressed size, or individual file decompression ratio. This exposed the server to Denial of Service (DoS) attacks via disk space exhaustion from decompression bombs (Zip Bombs).
**Learning:** Validating paths for traversal is not enough when handling archive files; the decompression payload size and complexity must also be bounded at the parsing layer to prevent resource exhaustion attacks.
**Prevention:** Always enforce strict safety thresholds on file count, total uncompressed size, and decompression ratio at the entry level of archive extraction functions.

## 2026-07-28 - [Defense-in-Depth Hardening for ModelRegistry and vLLM Runner]
**Vulnerability:** Core model registration and adapter-merging functions (`ModelRegistry` in `export/registry.py` and vLLM runners in `inference/vllm_runner.py`) relied entirely on handler-level or UI-level validation. If accessed programmatically as a library or via direct API calls, these layers had no self-contained path traversal or null-byte protection.
**Learning:** Security controls must be redundant and encapsulated inside class initializers and core functional interfaces (Layer 4/5) to ensure defense-in-depth, protecting core system APIs from execution context bypass.
**Prevention:** Always bundle path traversal and identifier guards (`validate_path_traversal` and `validate_identifier`) inside core API constructor hooks and low-level processing/merging entry points.

## 2026-07-24 - [Internal Inference Batch Processing Path Traversal Hardening]
**Vulnerability:** The internal core batched inference function (`batch_generate` in `inference/generate.py`) loaded and read files directly from the `prompts_file.name` argument without invoking path traversal or null-byte guards.
**Learning:** Even if the UI handler wraps the file upload parameter in a security check, the underlying shared core inference service must independently enforce security validation bounds to protect against direct library integration exploitation or command injection.
**Prevention:** Always apply the centralized `validate_path_traversal` guard directly at the entry points of internal processing or inference functions that read or write file paths.

## 2026-07-18 - [Centralized Ingestion Path Traversal Hardening]
**Vulnerability:** The centralized ingestion loader (`load_dataset_from_file` in `data/loader.py`) accepted any file object and resolved its `name` attribute directly without path traversal or null byte validation.
**Learning:** While CLI and UI layers had validation guards for path traversal parameters, secondary input fields or API routes (like library calls or direct handler invocation) could bypass those boundaries and read arbitrary system files during parsing if the shared ingestion layer itself was not validated.
**Prevention:** Always apply centralized traversal guards (`validate_path_traversal`) directly at the entry point of the shared ingestion/loading logic (`load_dataset_from_file`) for complete defense-in-depth coverage.

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
## 2025-05-25 - [Resource Exhaustion DoS Mitigation for Batch & Merge]
**Vulnerability:** Denial of Service (DoS) risk via disk exhaustion due to untracked temporary files (batch inference CSVs) and merged model directories.
**Learning:** Transient artifacts generated during inference or model manipulation (like merging) often escape the cleanup logic applied to primary training outputs. Standardizing on `tempfile.mkdtemp` and tracking "last seen" paths in a centralized `AppState` ensures these large resources are reclaimed on subsequent operations.
**Prevention:** Always track and clean up temporary paths for batch results and merged models at the handler entry point. Prefer `tempfile.mkdtemp` for unique, permission-restricted directory creation over predictable path suffixes.

## 2026-07-13 - [Defense-in-Depth Path Traversal Hardening]
**Vulnerability:** Core training modules (`training/sft.py`) relied on UI-level sanitization for model paths and output directories, creating a security gap if called directly via the CLI or external scripts.
**Learning:** Security boundaries must be enforced at the lowest possible entry point in the core logic, not just in the UI handlers. Hardening the `train_model` and `load_qlora_model_v27` functions ensures that path traversal attempts are blocked regardless of how the training pipeline is invoked.
**Prevention:** Always re-validate and strip whitespace from user-provided paths (like `model_name` and `output_dir`) inside core logic functions, even if they are already validated by UI handlers.

## 2026-07-16 - [Credential Input Hardening]
**Vulnerability:** HuggingFace tokens in `push_to_hub` and `ModelRegistry` were validated for format (prefix/length) but not for dangerous characters (null bytes, path traversal), potentially leading to injection in downstream logs or library calls.
**Learning:** Security validation must be exhaustive even for "opaque" strings like API tokens. If a string is passed from user input to any internal API, it should be checked against common injection patterns (`\0`, `..`, `\`) regardless of its intended use.
**Prevention:** Apply the centralized `validate_path_traversal` guard to all user-provided strings, including credentials and identifiers, before processing them.
## 2026-07-14 - [Ubiquitous Input Hardening across CLI & Credentials]
**Vulnerability:** CLI arguments (`--data`) and sensitive credentials (HF tokens) escaped traversal and null-byte validation, despite these checks being present for other path parameters.
**Learning:** Security debt often persists in "secondary" entry points like CLI commands or non-path parameters that can still carry injection payloads (e.g., null bytes in tokens). Standardized guards should be applied ubiquitously to all user-controlled strings that interact with the filesystem or external APIs.
**Prevention:** Audit all command-line options and credential fields to ensure they utilize the centralized `validate_path_traversal` guard, ensuring defense-in-depth even for alphanumeric fields to prevent legacy injection techniques.

## 2026-07-20 - [Validation Order and Filesystem Interception]
**Vulnerability:** Security checks via `validate_path_traversal` in `push_to_hub` were executed after a standard directory check (`os.path.isdir`), exposing the application to potentially unsafe filesystem checks on raw/unsanitized user input.
**Learning:** Performing filesystem operations (like existence or directory checks) before input validation violates the security-by-design principle of defense in depth, allowing potentially malicious paths to be queried against the system's files before validation.
**Prevention:** Always place validation guards (e.g., `validate_path_traversal`) at the absolute top of the function body, preceding any filesystem checks, directory lookups, or API calls on the untrusted path inputs.

## 2026-07-22 - [Secondary File Upload Path Traversal Gaps]
**Vulnerability:** Gradio handlers accepting file uploads directly (PEFT zip, batch test file, evaluation file) processed filename attributes (`file.name`) without invoking path traversal or null-byte guards, bypassing central ingestion validation.
**Learning:** Hardening shared ingestion layers (`load_dataset_from_file`) is insufficient if secondary entry points directly read from or construct temporary paths using raw, unvalidated upload filenames.
**Prevention:** Always apply the centralized `validate_path_traversal` guard to `.name` attributes of all uploaded file objects at the absolute handler-level entry points.

## 2026-07-23 - [Internal Inference Load Path Traversal Hardening]
**Vulnerability:** The internal core model/tokenizer loading function (`_load_for_inference` in `inference/generate.py`) loaded models directly from `model_name` and `lora_path` arguments without whitespace stripping, null-byte validation, or path traversal checks.
**Learning:** Hardening UI handlers and CLI commands is insufficient for complete security, as internal functions (e.g. called via APIs, custom scripts, or secondary entrypoints) can still receive raw user-controlled paths. Applying validation at the shared boundary layer ensures uniform defense-in-depth.
**Prevention:** Always strip whitespace and apply centralized path traversal/null-byte validation guards directly at the entry points of internal resource loading helpers.
