"""ui/tabs/evaluation_tab.py — BLEU / ROUGE / BERTScore / LLM-as-Judge evaluation."""
import gradio as gr

from config.constants import HAS_NLTK, HAS_ROUGE, HAS_BERTSCORE, LLM_JUDGE_CRITERIA


def build_evaluation_tab() -> dict:
    with gr.Tab("🧪 Evaluation"):
        gr.HTML(
            '<div id="eval-banner">'
            '<h3 style="color:#f59e0b;margin:0">'
            "🧪 v2.7 Advanced Evaluation Suite (Batched)"
            "</h3></div>"
        )
        gr.Markdown(
            f"BLEU: {'✅ (nltk)' if HAS_NLTK else '❌ pip install nltk'} | "
            f"ROUGE: {'✅ (rouge_score)' if HAS_ROUGE else '❌ pip install rouge-score'} | "
            f"BERTScore: {'✅ (bert_score)' if HAS_BERTSCORE else '❌ pip install bert-score'}"
        )
        with gr.Row():
            with gr.Column():
                eval_model_choice = gr.Dropdown(
                    choices=[
                        "gpt2", "distilgpt2", "facebook/opt-125m",
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    ],
                    value="gpt2", label="Model to Evaluate",
                )
                eval_custom_model = gr.Textbox(
                    label="Or custom model / local path",
                    placeholder="./output/model or username/model",
                )
                eval_lora_path_in = gr.Textbox(
                    label="PEFT Adapter Path (optional)",
                    placeholder="./output or leave empty",
                )
                eval_file = gr.File(
                    label="Test Dataset (CSV/JSONL with 'prompt' and optionally 'reference' columns)",
                    file_types=[".csv", ".jsonl"],
                )
                eval_max_new_tokens_slider = gr.Slider(
                    minimum=64, maximum=2048, value=150, step=64,
                    label="Max New Tokens (generation)",
                    info="Maximum tokens generated per prompt during evaluation.",
                )  # Minor Fix 6: was previously hardcoded at 150
                eval_run_bertscore = gr.Checkbox(
                    label="Compute BERTScore (slow, requires GPU for speed)", value=False
                )
                eval_use_judge  = gr.Checkbox(label="Run LLM-as-Judge", value=False)
                judge_model_name = gr.Textbox(
                    label="Judge Model ID (used when LLM-as-Judge enabled)",
                    placeholder="gpt2 or any local model",
                )
                judge_criteria = gr.Dropdown(
                    choices=LLM_JUDGE_CRITERIA,
                    value="helpfulness",
                    label="Judge Criterion",
                )
                eval_btn = gr.Button("🧪 Run Evaluation", variant="primary")
            with gr.Column():
                eval_metrics_out = gr.Markdown("_Metrics will appear here after evaluation._")
                eval_results_df  = gr.DataFrame(
                    label="Predictions vs References", interactive=False
                )

    return dict(
        eval_model_choice=eval_model_choice, eval_custom_model=eval_custom_model,
        eval_lora_path_in=eval_lora_path_in, eval_file=eval_file,
        eval_max_new_tokens_slider=eval_max_new_tokens_slider,
        eval_run_bertscore=eval_run_bertscore, eval_use_judge=eval_use_judge,
        judge_model_name=judge_model_name, judge_criteria=judge_criteria,
        eval_btn=eval_btn, eval_metrics_out=eval_metrics_out, eval_results_df=eval_results_df,
    )
