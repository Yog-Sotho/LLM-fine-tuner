# Palette's UX and Accessibility Journal

This journal documents critical UX and accessibility (a11y) learnings discovered while working on this repository.

## 2025-07-28 - [Accessible Tab Labels in Gradio]
**Learning:** Empty or whitespace labels in Gradio inputs/widgets impair screen readers and lead to inconsistent visual alignment. Applying proper labels and descriptions matches existing CSS design tokens (such as uppercase letter-spaced headings) and unifies the theme.
**Action:** Always verify that interactive Gradio components have meaningful semantic labels rather than blank or spacer values.

## 2025-07-29 - [Avoid Non-Interactive Input Locked Fields for Shared States]
**Learning:** Automatically-populated text boxes (such as PEFT adapters or exported GGUF model paths) in multi-stage workflows should not be set to `interactive=False` when users may want to bypass earlier stages in future sessions. Locking inputs entirely prevents keyboard focus, manual pasting, and keyboard navigation, breaking basic accessibility and workflow resumption.
**Action:** Keep output path fields interactive but use placeholder text to clarify that they are both auto-filled and editable.

## 2025-07-30 - [Keep Model Resource Estimations Synced Dynamically]
**Learning:** Displaying static resource estimations based solely on dropdown selections when custom text override fields exist causes a disconnect between user input and interface feedback. Automatically syncing both events to update parameter/VRAM estimates ensures consistent user expectations and prevents confusion.
**Action:** Always bind the change events of both override inputs and baseline choices to update shared status blocks, with proper fallback logic for empty or whitespace-only strings.
