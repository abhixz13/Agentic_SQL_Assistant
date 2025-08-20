from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TrainingExample:
    """Represents a single training example consisting of a prompt/completion pair."""

    prompt: str
    completion: str

    @classmethod
    def from_jsonl(cls, path: str | Path) -> List["TrainingExample"]:
        """Load a list of :class:`TrainingExample` from a JSONL file.

        Each line in ``path`` must be a JSON object with ``prompt`` and ``completion``
        fields.  Empty lines are ignored.
        """

        path = Path(path)
        examples: List[TrainingExample] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                examples.append(cls(prompt=data["prompt"], completion=data["completion"]))
        return examples


def _build_dataset(examples: List[TrainingExample]):
    """Convert examples into a HuggingFace ``Dataset`` object.

    The dataset contains a single ``text`` column with the prompt and completion
    concatenated.  The exact formatting can be adjusted depending on the target
    model.  Here we separate prompt and completion with a newline.
    """

    from datasets import Dataset  # Lazy import to avoid heavy dependency at import time

    records = [{"text": f"{ex.prompt}\n{ex.completion}"} for ex in examples]
    return Dataset.from_list(records)


def fine_tune(
    base_model: str,
    train_path: str | Path,
    output_dir: str | Path,
    validation_path: str | Path | None = None,
) -> str:
    """Fine‑tune ``base_model`` using Unsloth/HuggingFace tooling.

    Parameters
    ----------
    base_model:
        Name of the base model to fine‑tune (e.g., ``"google/gemma-2b"``).
    train_path:
        Path to a JSONL file containing training examples.
    output_dir:
        Directory where the fine‑tuned model will be saved.
    validation_path:
        Optional path to validation examples; if provided the trainer will
        evaluate during training.

    Returns
    -------
    str
        The path to the directory containing the fine‑tuned model.
    """

    examples = TrainingExample.from_jsonl(train_path)
    train_dataset = _build_dataset(examples)

    eval_dataset = None
    if validation_path:
        eval_dataset = _build_dataset(TrainingExample.from_jsonl(validation_path))

    # Import Unsloth and HuggingFace components lazily to avoid imposing a
    # heavy dependency when this module is imported but training is not run.
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(model)

    trainer = model.get_trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
    )
    trainer.train()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return str(output_dir)


def evaluate_model(model_path: str | Path, validation_path: str | Path) -> Dict[str, float]:
    """Evaluate a fine‑tuned model on a validation set.

    The evaluation performed here is intentionally lightweight and is meant to
    provide a quick sanity check rather than a rigorous benchmark.  It computes
    the fraction of validation examples for which the model's output contains
    the expected completion string.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    examples = TrainingExample.from_jsonl(validation_path)
    if not examples:
        return {"accuracy": 0.0}

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    for ex in examples:
        inputs = tokenizer(ex.prompt, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=64)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if ex.completion.strip() in text:
            correct += 1

    accuracy = correct / len(examples)
    return {"accuracy": accuracy}
