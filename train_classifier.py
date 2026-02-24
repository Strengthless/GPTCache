"""
BERT-based Binary Classification Training Pipeline
Fine-tunes encoder models (BERT, DistilBERT, ModernBERT, DeBERTa) for cache classification.

Note: This uses HuggingFace Trainer. Unsloth does NOT support BERT/encoder models.
If you want to use unsloth, you'll need a decoder LLM approach (Phi-3-mini, Qwen2, etc.)
which is less efficient for pure classification tasks.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


@dataclass
class ModelConfig:
    """Model configuration."""
    # Recommended models (choose based on your needs)
    # 1. answerdotai/ModernBERT-base - Best modern choice (2025+), 8192 context, fast
    # 2. distilbert/distilbert-base-uncased - Classic lightweight, 66M params
    # 3. microsoft/deberta-v3-small or base - High accuracy but slower
    # 4. sentence-transformers/all-MiniLM-L6-v2 - Ultra-light, 22M params

    model_name: str = "distilbert/distilbert-base-uncased"  # Change to ModernBERT if available
    max_length: int = 512  # 8192 for ModernBERT
    num_labels: int = 2


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "models/cache_classifier"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    seed: int = 42


class CacheClassifierTrainer:
    """Train BERT-based binary classifier for cache decisions."""

    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize tokenizer
        print(f"Loading tokenizer: {model_config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

        # Model will be loaded later
        self.model = None

    def load_and_prepare_data(self,
                             data_file: str,
                             test_size: float = 0.15,
                             val_size: float = 0.15) -> DatasetDict:
        """
        Load labeled dataset and prepare train/val/test splits.

        Args:
            data_file: Path to labeled JSONL file
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining after test split)

        Returns:
            DatasetDict with train/val/test splits
        """
        print("="*60)
        print("Loading and preparing dataset")
        print("="*60)

        # Load data
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Only include labeled items
                if item.get("label") is not None and item["label"] in [0, 1]:
                    data.append({
                        "text": item["text"],
                        "label": item["label"],
                    })

        print(f"Loaded {len(data)} labeled examples")

        # Check class balance
        labels = [d["label"] for d in data]
        n_cacheable = sum(labels)
        n_not_cacheable = len(labels) - n_cacheable

        print(f"\nClass distribution:")
        print(f"  Cacheable (1): {n_cacheable} ({n_cacheable/len(labels)*100:.1f}%)")
        print(f"  Not cacheable (0): {n_not_cacheable} ({n_not_cacheable/len(labels)*100:.1f}%)")

        # Warn if imbalanced
        if max(n_cacheable, n_not_cacheable) / min(n_cacheable, n_not_cacheable) > 2:
            print("\n⚠️  WARNING: Classes are imbalanced!")
            print("   Consider using class weights or resampling.")

        # Split into train/val/test with stratification
        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            stratify=labels,
            random_state=self.training_config.seed
        )

        train_labels = [d["label"] for d in train_data]
        train_data, val_data = train_test_split(
            train_data,
            test_size=val_size,
            stratify=train_labels,
            random_state=self.training_config.seed
        )

        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_data)}")
        print(f"  Validation: {len(val_data)}")
        print(f"  Test: {len(test_data)}")

        # Convert to HuggingFace Dataset
        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data),
        })

        # Tokenize
        print("\nTokenizing dataset...")
        tokenized_dataset = dataset_dict.map(
            self._tokenize_function,
            batched=True,
            desc="Tokenizing"
        )

        return tokenized_dataset

    def _tokenize_function(self, examples):
        """Tokenize text examples."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.model_config.max_length,
        )

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Accuracy
        accuracy = accuracy_score(labels, predictions)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_no_cache": precision_per_class[0],
            "recall_no_cache": recall_per_class[0],
            "f1_no_cache": f1_per_class[0],
            "precision_cache": precision_per_class[1],
            "recall_cache": recall_per_class[1],
            "f1_cache": f1_per_class[1],
        }

    def train(self, tokenized_dataset: DatasetDict) -> Trainer:
        """
        Train the model.

        Args:
            tokenized_dataset: Tokenized dataset with train/val/test splits

        Returns:
            Trained Trainer object
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        # Load model
        print(f"Loading model: {self.model_config.model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name,
            num_labels=self.model_config.num_labels,
            id2label={0: "no_cache", 1: "cache"},
            label2id={"no_cache": 0, "cache": 1}
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_dir=f"{self.training_config.output_dir}/logs",
            logging_steps=self.training_config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.training_config.eval_steps,
            save_strategy="steps",
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            seed=self.training_config.seed,
            fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
            report_to="none",  # Disable wandb/tensorboard for now
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train
        print("\nStarting training...")
        trainer.train()

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)

        return trainer

    def evaluate(self, trainer: Trainer, tokenized_dataset: DatasetDict):
        """
        Evaluate on test set and print detailed metrics.

        Args:
            trainer: Trained Trainer object
            tokenized_dataset: Dataset with test split
        """
        print("\n" + "="*60)
        print("Evaluating on Test Set")
        print("="*60)

        # Evaluate
        eval_results = trainer.evaluate(tokenized_dataset["test"])

        print("\nTest Set Metrics:")
        for key, value in eval_results.items():
            if not key.startswith("eval_"):
                continue
            metric_name = key.replace("eval_", "")
            print(f"  {metric_name}: {value:.4f}")

        # Get predictions for confusion matrix
        predictions = trainer.predict(tokenized_dataset["test"])
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = predictions.label_ids

        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 No Cache  |  Cache")
        print(f"Actual No Cache:  {cm[0][0]:6d}    |  {cm[0][1]:6d}")
        print(f"Actual Cache:     {cm[1][0]:6d}    |  {cm[1][1]:6d}")

        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            true_labels,
            pred_labels,
            target_names=["No Cache", "Cache"],
            digits=4
        ))

        # Save results
        results_file = Path(self.training_config.output_dir) / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)

        print(f"\nResults saved to {results_file}")

    def save_model(self, output_dir: Optional[str] = None):
        """Save model and tokenizer."""
        if output_dir is None:
            output_dir = self.training_config.output_dir

        output_path = Path(output_dir) / "final_model"
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving model to {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        print("Model saved successfully!")
        print(f"\nTo use the model:")
        print(f"  from transformers import AutoModelForSequenceClassification, AutoTokenizer")
        print(f"  model = AutoModelForSequenceClassification.from_pretrained('{output_path}')")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_path}')")


def main():
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Train BERT-based cache classifier")
    parser.add_argument("--data", type=str,
                       default="cache_classifier_data/labeled_dataset.jsonl",
                       help="Path to labeled dataset JSONL file")
    parser.add_argument("--model", type=str,
                       default="distilbert/distilbert-base-uncased",
                       help="Pretrained model name (distilbert, ModernBERT, deberta, etc.)")
    parser.add_argument("--output", type=str,
                       default="models/cache_classifier",
                       help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")

    args = parser.parse_args()

    # Configuration
    model_config = ModelConfig(
        model_name=args.model,
        max_length=args.max_length,
    )

    training_config = TrainingConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Initialize trainer
    cache_trainer = CacheClassifierTrainer(model_config, training_config)

    # Load and prepare data
    tokenized_dataset = cache_trainer.load_and_prepare_data(args.data)

    # Train
    trainer = cache_trainer.train(tokenized_dataset)

    # Evaluate
    cache_trainer.evaluate(trainer, tokenized_dataset)

    # Save final model
    cache_trainer.save_model()

    print("\n" + "="*60)
    print("Training Pipeline Complete!")
    print("="*60)
    print("\nNext step: Use inference.py to test your model")


if __name__ == "__main__":
    main()
