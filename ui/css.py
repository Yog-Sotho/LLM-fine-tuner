"""
ui/css.py
==========
Full custom CSS for the LLM Fine-Tuner v3.2 dark-purple UI theme.
Import CUSTOM_CSS and pass it directly to gr.Blocks(css=CUSTOM_CSS).
"""

CUSTOM_CSS = """
/* ── Root variables ───────────────────────── */
:root {
    --bg-main:    #0f0f18;
    --bg-card:    #1a1a2e;
    --bg-input:   #16213e;
    --accent:     #7c3aed;
    --accent-lt:  #a78bfa;
    --accent-glow:rgba(124, 58, 237, 0.35);
    --success:    #10b981;
    --warn:       #f59e0b;
    --danger:     #ef4444;
    --text-main:  #e2e8f0;
    --text-muted: #94a3b8;
    --border:     #334155;
    --radius:     12px;
}
/* ── Global body ───────────────────────────── */
body, .gradio-container {
    background: var(--bg-main) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
/* ── Header banner ─────────────────────────── */
#header-banner {
    background: linear-gradient(135deg, #1e0a3c 0%, #0f1e4c 50%, #0a2b4c 100%);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    padding: 24px 32px;
    margin-bottom: 20px;
    box-shadow: 0 0 40px var(--accent-glow);
}
#header-banner h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
#header-banner p {
    color: var(--text-muted);
    margin: 0;
    font-size: 0.95rem;
}
/* ── Hardware info box ─────────────────────── */
#hw-info {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 14px 18px;
    font-size: 0.88rem;
    color: var(--text-main) !important;
}
#hw-info ul, #hw-info li, #hw-info p, #hw-info span {
    color: var(--text-main) !important;
}
#hw-info ul {
    margin: 0 !important;
    padding-left: 20px !important;
}
#hw-info li {
    margin-bottom: 4px !important;
    line-height: 1.4;
}
#hw-info li:last-child {
    margin-bottom: 0 !important;
}
/* ── Tab bar ───────────────────────────────── */
.tab-nav button {
    background: var(--bg-card) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s;
}
.tab-nav button.selected {
    background: var(--accent) !important;
    color: white !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 12px var(--accent-glow);
}
/* ── Cards / panels ────────────────────────── */
.gr-box, .gr-form, .gr-panel,
.gradio-box, .block {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
/* ── Inputs ────────────────────────────────── */
input, textarea, select,
.gr-input, .gr-textarea {
    background: var(--bg-input) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 8px var(--accent-glow) !important;
    outline: none !important;
}
/* ── Primary button ────────────────────────── */
.gr-button-primary, button[data-testid="primary"] {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px var(--accent-glow);
}
.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px var(--accent-glow) !important;
}
/* ── Stop button ───────────────────────────── */
button[data-testid="stop"], .gr-button-stop {
    background: linear-gradient(135deg, #b91c1c, #7f1d1d) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
/* ── Secondary buttons ─────────────────────── */
button[data-testid="secondary"] {
    background: var(--bg-input) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
/* ── Accordion ─────────────────────────────── */
.gr-accordion {
    border: 1px solid var(--accent) !important;
    border-radius: var(--radius) !important;
}
.gr-accordion > .label-wrap {
    background: rgba(124,58,237,0.15) !important;
    color: var(--accent-lt) !important;
    font-weight: 600 !important;
}
/* ── Labels ────────────────────────────────── */
label, .gr-label {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
/* ── Sliders ───────────────────────────────── */
input[type="range"] {
    accent-color: var(--accent) !important;
}
/* ── Loss chart section ────────────────────── */
#loss-chart-wrap {
    border: 1px solid var(--accent) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-card) !important;
    padding: 12px;
    margin-top: 12px;
}
/* ── Status pill ───────────────────────────── */
.status-ok  { color: var(--success); font-weight: 700; }
.status-warn{ color: var(--warn);    font-weight: 700; }
.status-err { color: var(--danger);  font-weight: 700; }
/* ── Scrollbar ─────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 3px; }
/* ── v2.7 RLHF section highlight ───────────── */
#rlhf-banner {
    background: linear-gradient(135deg, #0a2b2b 0%, #0a1e3c 100%);
    border: 1px solid var(--success);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin-bottom: 12px;
}
/* ── v2.7 evaluation section ───────────────── */
#eval-banner {
    background: linear-gradient(135deg, #1a0a3c 0%, #0f2a1e 100%);
    border: 1px solid var(--warn);
    border-radius: var(--radius);
    padding: 14px 18px;
    margin-bottom: 12px;
}
/* ── FIX 2e: Preview refresh button styling ── */
#refresh-preview-btn {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.35) !important;
}
/* ── Keyboard Focus Indicators (Accessibility) ── */
button:focus-visible,
.tab-nav button:focus-visible,
input:focus-visible,
select:focus-visible,
textarea:focus-visible {
    outline: 2px solid var(--accent-lt) !important;
    outline-offset: 2px !important;
}
"""
