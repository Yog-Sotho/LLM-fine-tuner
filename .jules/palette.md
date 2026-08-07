# Palette's UX and Accessibility Journal

This journal documents critical UX and accessibility (a11y) learnings discovered while working on this repository.

## 2025-07-28 - [Accessible Tab Labels in Gradio]
**Learning:** Empty or whitespace labels in Gradio inputs/widgets impair screen readers and lead to inconsistent visual alignment. Applying proper labels and descriptions matches existing CSS design tokens (such as uppercase letter-spaced headings) and unifies the theme.
**Action:** Always verify that interactive Gradio components have meaningful semantic labels rather than blank or spacer values.

## 2025-07-29 - [Avoid Non-Interactive Input Locked Fields for Shared States]
**Learning:** Automatically-populated text boxes (such as PEFT adapters or exported GGUF model paths) in multi-stage workflows should not be set to `interactive=False` when users may want to bypass earlier stages in future sessions. Locking inputs entirely prevents keyboard focus, manual pasting, and keyboard navigation, breaking basic accessibility and workflow resumption.
**Action:** Keep output path fields interactive but use placeholder text to clarify that they are both auto-filled and editable.

## 2025-07-30 - [Required Asterisk Indicators and API Token Contextual Onboarding]
**Learning:** Textboxes with highly specific security scopes (such as Hugging Face Hub write API tokens or repository IDs) require both visual indicators (`*`) to denote mandatory status and contextual help (`info` attribute) to provide immediate, accessible guidelines for screen readers and novice users alike. Without this, users suffer from process failures due to incorrect token privileges or bad repository naming schemes.
**Action:** Always append an asterisk `*` to labels of required fields, and supply descriptive, concise `info` tooltips/help text to text inputs involving specific formats, write credentials, or third-party scopes.
