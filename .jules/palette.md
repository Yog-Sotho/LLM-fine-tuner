# Palette's UX and Accessibility Journal

This journal documents critical UX and accessibility (a11y) learnings discovered while working on this repository.

## 2025-07-28 - [Accessible Tab Labels in Gradio]
**Learning:** Empty or whitespace labels in Gradio inputs/widgets impair screen readers and lead to inconsistent visual alignment. Applying proper labels and descriptions matches existing CSS design tokens (such as uppercase letter-spaced headings) and unifies the theme.
**Action:** Always verify that interactive Gradio components have meaningful semantic labels rather than blank or spacer values.

## 2025-07-29 - [Avoid Non-Interactive Input Locked Fields for Shared States]
**Learning:** Automatically-populated text boxes (such as PEFT adapters or exported GGUF model paths) in multi-stage workflows should not be set to `interactive=False` when users may want to bypass earlier stages in future sessions. Locking inputs entirely prevents keyboard focus, manual pasting, and keyboard navigation, breaking basic accessibility and workflow resumption.
**Action:** Keep output path fields interactive but use placeholder text to clarify that they are both auto-filled and editable.

## 2025-08-08 - [Interactive Component Info for Forms and Critical Credentials]
**Learning:** Interactive Gradio widgets like `gr.Dropdown`, `gr.Textbox`, and `gr.Slider` support descriptive, high-context sub-labels or descriptions using the `info` parameter to dramatically improve keyboard/screen reader accessibility (a11y) and user onboarding, while layout-only or media-only widgets like `gr.File` do not natively support this parameter. Incorporating explicit required indicators (`*` in labels) alongside helpful `info` text reduces validation confusion and increases completion rates for credential/repository-related fields.
**Action:** Always include explicit required indicators (`*`) in labels and detailed descriptions in the `info` parameter of critical input fields like repository IDs, access tokens, and tag versionings.
