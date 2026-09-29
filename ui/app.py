"""
ui/app.py
==========
Top-level Gradio application builder.
Assembles all tabs, wires every event, and returns the gr.Blocks demo object.

Rule: ALL .click() / .change() event wiring lives HERE and ONLY here.
Tab files define layout; handlers.py defines logic; this file connects them.

Patch log
---------
  F-6  : ``eval_btn.click`` outputs now include ``et["eval_preview_html"]``
         as a third output.  ``on_evaluate_click()`` returns 3 values:
         (metrics_str, result_dataframe, preview_html).  The HTML preview
         is rendered by the ``eval_preview_html`` gr.HTML component added
         to evaluation_tab.py.
  F-2  : ``build_loss_chart`` now produces a DataFrame with an ETA column
         (when timing data is present in log_records).  The loss_df
         Dataframe headers in train_tab.py are unchanged — the ETA column
         appears as an extra column which Gradio renders automatically.
"""

import gradio as gr

from config.constants import APP_NAME, GPU_JOB_CONCURRENCY, GPU_QUEUE_ID
from core.hardware import get_hardware_summary, get_model_info
from core.state import app_state
from data.augmentation import on_augment_click, on_quality_filter_click
from export.gguf import on_export_gguf
from export.merge import on_merge_adapters_click
from export.registry import on_registry_list, on_registry_upload
from export.utils import clear_gpu_cache, on_peft_zip_upload
from inference.benchmarks import on_benchmark_click
from inference.evaluation import on_evaluate_click
from inference.vllm_runner import on_merge_adapter_click, on_vllm_generate
from training.distill import train_distill
from training.grpo import train_grpo
from training.kto import train_kto
from training.orpo import train_orpo_v27
from training.reward import train_reward_model_v27
from ui.handlers import (
    build_loss_chart,
    on_batch_test,
    on_file_upload,
    on_generate,
    on_hub_load,
    on_push,
    on_quantize_export,
    on_refresh_preview,
    on_remote_chat,
    on_stop,
    on_train_click,
)
from ui.tabs.data_tab import build_data_tab
from ui.tabs.evaluation_tab import build_evaluation_tab
from ui.tabs.gguf_tab import build_gguf_tab
from ui.tabs.inference_tab import build_inference_tab
from ui.tabs.rlhf_tab import build_rlhf_tab
from ui.tabs.share_tab import build_share_tab
from ui.tabs.train_tab import build_train_tab


def build_theme() -> gr.themes.Base:
    """Return the app theme (Gradio 6 takes theme and css in launch(), not Blocks)."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    )


# Heavy GPU jobs share one queue: extra requests wait (Gradio shows their place in line)
# instead of loading a second model onto the same GPU.
GPU_JOB = {"concurrency_id": GPU_QUEUE_ID, "concurrency_limit": GPU_JOB_CONCURRENCY}


def build_demo() -> gr.Blocks:
    """Build and return the fully-wired Gradio Blocks demo."""

    with gr.Blocks(title=f"🧠 {APP_NAME}") as demo:
        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div id="header-banner">
            <h1>🧠 {APP_NAME}</h1>
            <p>SFT · DPO · ORPO · KTO · Reward model · GRPO — LoRA / QLoRA · GGUF export · vLLM inference</p>
        </div>
        """)
        hw_md = gr.Markdown(get_hardware_summary(), elem_id="hw-info")  # noqa: F841

        # ── Build all tabs, collect component dicts ─────────────────────────
        with gr.Tabs():
            dt = build_data_tab()
            tt = build_train_tab()
            gt = build_gguf_tab()
            it = build_inference_tab()
            rlt = build_rlhf_tab()
            et = build_evaluation_tab()
            st = build_share_tab()

        # ══════════════════════════════════════════════════════════════════
        # EVENT WIRING  (v2.9 Fix H — was entirely missing in the original)
        # ══════════════════════════════════════════════════════════════════

        # ── Data Tab ───────────────────────────────────────────────────────
        dt["file_input"].change(
            fn=on_file_upload,
            inputs=[dt["file_input"], tt["training_mode"]],
            outputs=[
                dt["file_status"],
                dt["col_inst"],
                dt["col_out"],
                dt["col_text"],
                dt["preview_box"],
                dt["stats_box"],
                dt["raw_df_state"],
                dt["file_type_state"],
            ],
        )
        # C-5 FIX: Reset augmented_ds_state to None whenever a new file is uploaded
        # so old augmented data doesn't accidentally persist into a new training job.
        dt["file_input"].change(
            fn=lambda _: None,
            inputs=[dt["file_input"]],
            outputs=[dt["augmented_ds_state"]],
        )

        dt["hub_load_btn"].click(
            fn=on_hub_load,
            inputs=[
                dt["hub_dataset_id"],
                dt["hub_config"],
                dt["hub_split"],
                dt["hub_max_rows"],
                tt["training_mode"],
            ],
            outputs=[
                dt["file_status"],
                dt["preview_box"],
                dt["stats_box"],
                dt["augmented_ds_state"],
            ],
        )

        dt["refresh_preview_btn"].click(
            fn=on_refresh_preview,
            inputs=[
                dt["file_input"],
                tt["training_mode"],
                dt["col_inst"],
                dt["col_out"],
                dt["col_text"],
                dt["raw_df_state"],
                dt["file_type_state"],
            ],
            outputs=[dt["preview_box"], dt["stats_box"]],
        )

        # C-5 FIX: Augment button now has 4 outputs — the fourth is augmented_ds_state.
        # Previously the augmented Dataset was returned to the UI but never stored in
        # state, so training always used the original file.
        dt["aug_btn"].click(
            fn=on_augment_click,
            inputs=[dt["file_input"], tt["training_mode"], dt["aug_factor"], dt["aug_type"]],
            outputs=[
                dt["aug_status"],
                dt["aug_preview"],
                dt["aug_stats"],
                dt["augmented_ds_state"],
            ],
        )

        # C-5 FIX: Quality filter button also stores its result in augmented_ds_state.
        dt["qf_btn"].click(
            fn=on_quality_filter_click,
            inputs=[dt["file_input"], tt["training_mode"], dt["qf_min_len"], dt["qf_max_len"]],
            outputs=[dt["qf_status"], dt["aug_preview"], dt["aug_stats"], dt["augmented_ds_state"]],
        )

        # ── Training Tab ───────────────────────────────────────────────────
        tt["model_choice"].change(
            fn=get_model_info,
            inputs=[tt["model_choice"]],
            outputs=[tt["model_info_md"]],
        )

        # v2.9 Major Fix #3: single combined handler avoids race condition between
        # on_train_click and build_loss_chart firing separately.
        def _on_train_and_chart(
            file_input,
            model_choice,
            custom_model,
            training_preset,
            peft_method,
            use_lora,
            lora_rank,
            lora_alpha,
            prefix_tuning_num_virtual_tokens,
            prefix_tuning_token_dim,
            prefix_tuning_num_layers,
            prompt_tuning_num_virtual_tokens,
            lr,
            epochs,
            bs,
            grad_accum,
            max_len,
            warmup,
            early_stop,
            lr_sched,
            grad_ckpt,
            resume_ckpt,
            col_inst,
            col_out,
            col_text,
            use_unsloth,
            use_chat_template,
            system_prompt,
            training_mode,
            dpo_beta,
            heretic_mode,
            use_flash_attn,
            use_qlora_enhanced,
            augmented_ds,  # C-5 FIX: augmented dataset state injected here
            packing,
            run_name,
            seed,
            report_to,
            eval_split,
            lora_variant,
            progress=gr.Progress(),
            request: gr.Request | None = None,
        ):
            msg, zip_path, model_path, log_records = on_train_click(
                file_input,
                model_choice,
                custom_model,
                training_preset,
                peft_method,
                use_lora,
                lora_rank,
                lora_alpha,
                prefix_tuning_num_virtual_tokens,
                prefix_tuning_token_dim,
                prefix_tuning_num_layers,
                prompt_tuning_num_virtual_tokens,
                lr,
                epochs,
                bs,
                grad_accum,
                max_len,
                warmup,
                early_stop,
                lr_sched,
                grad_ckpt,
                resume_ckpt,
                col_inst,
                col_out,
                col_text,
                use_unsloth,
                use_chat_template,
                system_prompt,
                training_mode,
                dpo_beta,
                heretic_mode,
                use_flash_attn,
                use_qlora_enhanced,
                augmented_ds=augmented_ds,  # C-5 FIX
                packing=packing,
                run_name=run_name,
                seed=seed,
                report_to=report_to,
                eval_split=eval_split,
                lora_variant=lora_variant,
                progress=progress,
                request=request,
            )
            return msg, zip_path, model_path, log_records, build_loss_chart(log_records)

        tt["train_btn"].click(
            fn=_on_train_and_chart,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[
                dt["file_input"],
                tt["model_choice"],
                tt["custom_model"],
                tt["training_preset"],
                tt["peft_method"],
                tt["use_lora"],
                tt["lora_rank"],
                tt["lora_alpha"],
                tt["prefix_tuning_num_virtual_tokens"],
                tt["prefix_tuning_token_dim"],
                tt["prefix_tuning_num_layers"],
                tt["prompt_tuning_num_virtual_tokens"],
                tt["lr"],
                tt["epochs"],
                tt["bs"],
                tt["grad_accum"],
                tt["max_len"],
                tt["warmup"],
                tt["early_stop"],
                tt["lr_sched"],
                tt["grad_ckpt"],
                tt["resume_ckpt"],
                dt["col_inst"],
                dt["col_out"],
                dt["col_text"],
                tt["use_unsloth"],
                tt["use_chat_template"],
                tt["system_prompt"],
                tt["training_mode"],
                tt["dpo_beta"],
                tt["heretic_mode"],
                tt["use_flash_attn"],
                tt["use_qlora_enhanced"],
                dt["augmented_ds_state"],  # C-5 FIX: new input
                tt["packing"],
                tt["run_name"],
                tt["seed"],
                tt["report_to"],
                tt["eval_split"],
                tt["lora_variant"],
            ],
            outputs=[
                tt["log_output"],
                st["download_btn"],
                tt["model_path_state"],
                tt["log_records_state"],
                tt["loss_df"],
            ],
        )
        tt["stop_btn"].click(fn=on_stop, inputs=[], outputs=[tt["log_output"]])
        tt["clear_gpu_btn"].click(fn=clear_gpu_cache, inputs=[], outputs=[tt["log_output"]])

        # Auto-fill export / lora / merge adapter paths from model_path_state
        tt["model_path_state"].change(
            fn=lambda p: (p or "", p or "", p or ""),
            inputs=[tt["model_path_state"]],
            outputs=[gt["export_model_path"], it["lora_path"], it["merge_adapter_path_in"]],
        )

        # ── GGUF Export Tab ────────────────────────────────────────────────
        gt["export_btn"].click(
            fn=on_export_gguf,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[gt["export_model_path"], gt["quantization"]],
            outputs=[gt["export_status"], gt["gguf_file"]],
        )

        # ── Inference Tab ──────────────────────────────────────────────────
        gt["quant_btn"].click(
            fn=on_quantize_export,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[gt["export_model_path"], gt["quant_format"], dt["file_input"],
                    dt["augmented_ds_state"]],
            outputs=[gt["quant_status"]],
        )  # fmt: skip
        gt["merge_btn"].click(
            fn=on_merge_adapters_click,
            **GPU_JOB,  # loads the base model: one heavy job at a time
            inputs=[gt["merge_adapters"], gt["merge_weights"], gt["merge_method"],
                    gt["merge_density"], gt["merge_output"]],
            outputs=[gt["merge_status"]],
        )  # fmt: skip
        it["remote_btn"].click(
            fn=on_remote_chat,
            inputs=[it["remote_url"], it["remote_model"], it["remote_key"], it["remote_system"],
                    it["remote_prompt"], it["remote_max_tokens"], it["remote_temp"]],
            outputs=[it["remote_out"]],
        )  # fmt: skip
        it["gen_btn"].click(
            fn=on_generate,
            inputs=[
                it["prompt_in"],
                it["infer_model"],
                it["infer_custom"],
                it["lora_path"],
                it["max_tok"],
                it["temp"],
                it["top_p"],
            ],
            outputs=[it["gen_out"]],
        )
        it["batch_btn"].click(
            fn=on_batch_test,
            inputs=[it["batch_file"], it["infer_model"], it["infer_custom"], it["lora_path"]],
            outputs=[it["batch_out"]],
        )
        it["lora_zip_upload"].change(
            fn=on_peft_zip_upload,
            inputs=[it["lora_zip_upload"]],
            outputs=[it["lora_path"], it["lora_zip_status"], it["lora_zip_dir_state"]],
        )
        it["merge_btn"].click(
            fn=on_merge_adapter_click,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[
                it["merge_base_model_in"],
                it["merge_adapter_path_in"],
                tt["model_path_state"],
            ],
            outputs=[it["merge_status_out"], it["merged_model_path_state"]],
        )
        it["vllm_gen_btn"].click(
            fn=on_vllm_generate,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[
                it["merged_model_path_state"],
                it["vllm_prompt_in"],
                it["vllm_quant_select"],
                it["vllm_max_tokens"],
                it["vllm_temp_sl"],
                it["vllm_top_p_sl"],
            ],
            outputs=[it["vllm_gen_out"]],
        )

        # ── RLHF Pipeline Tab ──────────────────────────────────────────────
        rlt["rm_train_btn"].click(
            fn=train_reward_model_v27,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[
                rlt["rm_model_choice"],
                rlt["rm_file"],
                rlt["rm_output_dir"],
                rlt["rm_epochs"],
                rlt["rm_lr"],
                rlt["rm_batch"],
                rlt["rm_eval_steps"],
                rlt["rm_max_length"],
            ],
            outputs=[rlt["rm_status"]],
        )
        # Input order matches train_grpo's positional parameters.
        rlt["grpo_train_btn"].click(
            fn=train_grpo,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[
                rlt["grpo_policy_model"],
                rlt["grpo_reward_path"],
                rlt["grpo_file"],
                rlt["grpo_output_dir"],
                rlt["grpo_lr"],
                rlt["grpo_epochs"],
                rlt["grpo_num_generations"],
                rlt["grpo_prompts_per_step"],
                rlt["grpo_max_completion"],
                rlt["grpo_beta"],
                rlt["grpo_resume"],
                rlt["grpo_loss_type"],
                rlt["grpo_rewards"],
                rlt["grpo_regex"],
                rlt["grpo_lora_rank"],
                rlt["grpo_lora_alpha"],
                rlt["grpo_lora_variant"],
                rlt["grpo_use_vllm"],
            ],
            outputs=[rlt["grpo_status"]],
        )
        # Input order matches train_kto's positional parameters.
        rlt["kto_train_btn"].click(
            fn=train_kto,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[
                rlt["kto_model_choice"],
                rlt["kto_file"],
                rlt["kto_output_dir"],
                rlt["kto_lr"],
                rlt["kto_beta"],
                rlt["kto_epochs"],
                rlt["kto_batch"],
                rlt["kto_max_length"],
                rlt["kto_resume"],
            ],
            outputs=[rlt["kto_status"]],
        )
        # Input order matches train_distill's positional parameters.
        rlt["distill_train_btn"].click(
            fn=train_distill,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[
                rlt["distill_student"],
                rlt["distill_teacher"],
                rlt["distill_file"],
                rlt["distill_output_dir"],
                rlt["distill_lr"],
                rlt["distill_epochs"],
                rlt["distill_batch"],
                rlt["distill_max_length"],
                rlt["distill_lmbda"],
                rlt["distill_beta"],
                rlt["distill_temperature"],
                rlt["distill_max_new_tokens"],
                rlt["distill_resume"],
            ],
            outputs=[rlt["distill_status"]],
        )
        rlt["orpo_train_btn"].click(
            fn=train_orpo_v27,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[
                rlt["orpo_model_choice"],
                rlt["orpo_file"],
                rlt["orpo_output_dir"],
                rlt["orpo_lr"],
                rlt["orpo_beta"],
                rlt["orpo_alpha"],
                rlt["orpo_epochs"],
                rlt["orpo_batch"],
            ],
            outputs=[rlt["orpo_status"]],
        )

        # ── Evaluation Tab ─────────────────────────────────────────────────
        # F-6: on_evaluate_click now returns 3 values.  The third value is the
        # HTML prediction preview which populates eval_preview_html.
        et["eval_btn"].click(
            fn=on_evaluate_click,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[
                et["eval_model_choice"],
                et["eval_custom_model"],
                et["eval_lora_path_in"],
                et["eval_file"],
                et["eval_run_bertscore"],
                et["eval_use_judge"],
                et["judge_model_name"],
                et["judge_criteria"],
                et["eval_max_new_tokens_slider"],
                et["eval_compare_base"],
            ],
            outputs=[
                et["eval_metrics_out"],
                et["eval_results_df"],
                et["eval_preview_html"],  # F-6: new third output
            ],
        )
        et["bench_btn"].click(
            fn=on_benchmark_click,
            **GPU_JOB,  # one shared GPU queue (see config.constants)
            inputs=[
                et["eval_model_choice"],
                et["eval_custom_model"],
                et["eval_lora_path_in"],
                et["bench_tasks"],
                et["bench_limit"],
                et["bench_compare_base"],
            ],
            outputs=[et["bench_status"], et["bench_results_df"]],
        )

        # ── Share Tab ──────────────────────────────────────────────────────
        st["push_btn"].click(
            fn=on_push,
            inputs=[tt["model_path_state"], st["repo_id"], st["hf_token"]],
            outputs=[st["push_status"]],
        )
        st["registry_upload_btn"].click(
            fn=on_registry_upload,
            inputs=[
                tt["model_path_state"],
                st["registry_repo_id"],
                st["registry_token"],
                st["registry_version"],
                st["registry_notes"],
            ],
            outputs=[st["registry_status"]],
        )
        st["registry_list_btn"].click(
            fn=on_registry_list,
            inputs=[st["registry_repo_id"], st["registry_token"]],
            outputs=[st["registry_status"]],
        )

        # Closing or refreshing the tab stops that session's job and frees its temp files.
        def _close_session(request: gr.Request) -> None:
            app_state.close_session(request.session_hash)

        demo.unload(_close_session)

    return demo
