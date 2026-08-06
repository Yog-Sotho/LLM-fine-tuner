# Palette's UX and Accessibility Journal

This journal documents critical UX and accessibility (a11y) learnings discovered while working on this repository.

## 2025-07-28 - [Accessible Tab Labels in Gradio]
**Learning:** Empty or whitespace labels in Gradio inputs/widgets impair screen readers and lead to inconsistent visual alignment. Applying proper labels and descriptions matches existing CSS design tokens (such as uppercase letter-spaced headings) and unifies the theme.
**Action:** Always verify that interactive Gradio components have meaningful semantic labels rather than blank or spacer values.

## 2025-07-29 - [Avoid Non-Interactive Input Locked Fields for Shared States]
**Learning:** Automatically-populated text boxes (such as PEFT adapters or exported GGUF model paths) in multi-stage workflows should not be set to `interactive=False` when users may want to bypass earlier stages in future sessions. Locking inputs entirely prevents keyboard focus, manual pasting, and keyboard navigation, breaking basic accessibility and workflow resumption.
**Action:** Keep output path fields interactive but use placeholder text to clarify that they are both auto-filled and editable.

## 2025-07-30 - [Explicit Override Helper Info in Complex Forms]
**Learning:** Overriding inputs (e.g. custom model text overrides for adjacent dropdown selections) can be highly confusing if the precedence order is not immediately apparent. Leveraging `info` sub-labels on overriding text inputs explicitly clarifies behavior, guiding both standard users and screen readers on input hierarchy.
**Action:** For fields that dynamically override other selections, always use the `info` parameter to specify precedence rules explicitly.
