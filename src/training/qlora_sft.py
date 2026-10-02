"""QLoRA SFT entry point with a fast dependency/configuration failure mode."""
import argparse, json, time
from pathlib import Path
from ..aml.registry import get_model

def _format_messages(example):
    """Serialize chat records to plain text for MistralCommonBackend."""
    messages = example.get("messages", [])
    return "\n".join(f"[{message.get('role', 'user').upper()}]\n{message.get('content', '')}" for message in messages)


def _render_mistral_training_text(example, tokenizer):
    """Render a completed conversation as ordinary training text.

    MistralCommonBackend rejects a completed assistant message when TRL sees
    the raw ``messages`` column as a serving conversation.  Continuing the
    final assistant message is the training form; storing the rendered result
    under ``text`` prevents TRL from re-applying the serving formatter.
    """
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            continue_final_message=True,
        )
    }

def train(model_name: str, dataset: Path, output_dir: Path, *, max_steps=100, seed=42, max_samples=None, **kwargs):
    try:
        from datasets import load_dataset
        from transformers import BitsAndBytesConfig, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
        import torch
        from peft import LoraConfig
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError("QLoRA training requires datasets, transformers, peft, bitsandbytes, and trl") from exc
    spec = get_model(model_name); started = time.time(); output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset("json", data_files=str(dataset), split="train")
    if max_samples is not None:
        data = data.select(range(min(int(max_samples), len(data))))
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    if model_name == "ministral":
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
        try:
            tokenizer = MistralCommonBackend.from_pretrained(spec["model_id"], revision=spec["revision"], fix_mistral_regex=True)
        except (TypeError, ValueError):
            tokenizer = MistralCommonBackend.from_pretrained(spec["model_id"], revision=spec["revision"])
        model = Mistral3ForConditionalGeneration.from_pretrained(spec["model_id"], revision=spec["revision"], quantization_config=quantization, device_map="auto")
        # Pre-render completed conversations as text. Otherwise TRL treats the
        # final assistant answer as a serving prompt and MistralCommonBackend
        # raises InvalidMessageStructureException during tokenization.
        data = data.map(
            lambda example: _render_mistral_training_text(example, tokenizer),
            remove_columns=data.column_names,
        )
        formatting_func = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"])
        model = AutoModelForCausalLM.from_pretrained(spec["model_id"], revision=spec["revision"], quantization_config=quantization, device_map="auto")
        if not getattr(tokenizer, "chat_template", None):
            # Base checkpoints such as the configured Gemma/Qwen variants may
            # not ship a chat template. Convert conversations to plain text so
            # TRL does not call apply_chat_template during preprocessing.
            data = data.map(
                lambda example: {"text": _format_messages(example)},
                remove_columns=data.column_names,
            )
        formatting_func = None
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    args = TrainingArguments(output_dir=str(output_dir), max_steps=max_steps, per_device_train_batch_size=1, gradient_accumulation_steps=8, logging_steps=1, save_strategy="steps", save_steps=max(1, max_steps // 2), seed=seed, report_to=[])
    try:
        trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=data, peft_config=lora, formatting_func=formatting_func, args=args)
    except TypeError:
        trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=data, peft_config=lora, formatting_func=formatting_func, args=args)
    result = trainer.train(); trainer.save_model(str(output_dir)); trainer.save_state(); tokenizer.save_pretrained(str(output_dir))
    (output_dir / "loss_history.json").write_text(json.dumps(trainer.state.log_history, default=str, indent=2))
    (output_dir / "training_run.json").write_text(json.dumps({"model": spec, "dataset": str(dataset), "dataset_rows": len(data), "seed": seed, "max_steps": max_steps, "wall_clock_seconds": time.time() - started, "train_metrics": result.metrics, "output_dir": str(output_dir)}, default=str, indent=2))
    return result.metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True, choices=["gemma", "granite", "ministral", "qwen"]); p.add_argument("--dataset", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--max-steps", type=int, default=100); a = p.parse_args(); print(train(a.model, a.dataset, a.output_dir, max_steps=a.max_steps))
