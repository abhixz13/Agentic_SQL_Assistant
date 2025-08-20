"""Command line interface for fine‑tuning models.

This script orchestrates loading training data, running the fine‑tuning
process, evaluating the resulting model, and registering it so that the rest
of the application can discover and use it.
"""

from __future__ import annotations

import argparse

from learning.fine_tune import evaluate_model, fine_tune
from services import model_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine‑tune a language model")
    parser.add_argument(
        "--train",
        default="training_examples.jsonl",
        help="Path to training data in JSONL format",
    )
    parser.add_argument(
        "--validation",
        default=None,
        help="Optional path to validation data in JSONL format",
    )
    parser.add_argument(
        "--base-model",
        default="google/gemma-2b",
        help="Name of the base model to fine‑tune",
    )
    parser.add_argument(
        "--output-dir",
        default="fine_tuned_models",
        help="Directory where the fine‑tuned model will be saved",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = fine_tune(
        base_model=args.base_model,
        train_path=args.train,
        output_dir=args.output_dir,
        validation_path=args.validation,
    )

    metrics = None
    if args.validation:
        metrics = evaluate_model(output_dir, args.validation)

    model_registry.register_model(
        base_model=args.base_model,
        model_path=output_dir,
        metrics=metrics,
    )

    print(f"Model saved to {output_dir} and registered.")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
