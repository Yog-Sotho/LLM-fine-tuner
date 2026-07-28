# Palette's UX and Accessibility Journal

This journal documents critical UX and accessibility (a11y) learnings discovered while working on this repository.

## 2025-07-28 - [Accessible Tab Labels in Gradio]
**Learning:** Empty or whitespace labels in Gradio inputs/widgets impair screen readers and lead to inconsistent visual alignment. Applying proper labels and descriptions matches existing CSS design tokens (such as uppercase letter-spaced headings) and unifies the theme.
**Action:** Always verify that interactive Gradio components have meaningful semantic labels rather than blank or spacer values.
