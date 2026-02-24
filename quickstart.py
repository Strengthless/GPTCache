#!/usr/bin/env python3
"""
Quick Start Script for Cache Classifier Pipeline
Runs the entire pipeline or individual steps.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        print(f"Command failed with exit code: {result.returncode}")
        sys.exit(1)

    print(f"\n✅ Completed: {description}")


def check_requirements():
    """Check if requirements are installed."""
    print("Checking requirements...")

    try:
        import torch
        import transformers
        import datasets
        import sklearn
        print("✅ All requirements installed")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements_classifier.txt")
        return False


def check_ollama():
    """Check if Ollama is running."""
    print("Checking Ollama...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except Exception:
        pass

    print("⚠️  Ollama might not be running")
    print("   Make sure to start Ollama with:")
    print("   ollama run hf.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Quick start script for cache classifier training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything (full pipeline)
  python quickstart.py --all

  # Run specific steps
  python quickstart.py --generate-dataset
  python quickstart.py --label --max-queries 1000  # Test with 1000 queries
  python quickstart.py --train --epochs 2
  python quickstart.py --demo

  # Run from dataset generation to training
  python quickstart.py --generate-dataset --label --train
        """
    )

    # Pipeline steps
    parser.add_argument("--all", action="store_true",
                       help="Run entire pipeline (generate, label, train)")
    parser.add_argument("--generate-dataset", action="store_true",
                       help="Step 1: Generate dataset")
    parser.add_argument("--label", action="store_true",
                       help="Step 2: Label with LLM")
    parser.add_argument("--train", action="store_true",
                       help="Step 3: Train classifier")
    parser.add_argument("--demo", action="store_true",
                       help="Step 4: Run interactive demo")
    parser.add_argument("--benchmark", action="store_true",
                       help="Step 5: Benchmark on test set")

    # Options
    parser.add_argument("--max-queries", type=int,
                       help="Max queries to label (for testing)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Training epochs (default: 3)")
    parser.add_argument("--model", type=str, default="distilbert/distilbert-base-uncased",
                       help="Model to train (default: distilbert-base-uncased)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size (default: 16)")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip requirement and Ollama checks")

    args = parser.parse_args()

    # If no steps specified, show help
    if not any([args.all, args.generate_dataset, args.label, args.train, args.demo, args.benchmark]):
        parser.print_help()
        sys.exit(0)

    print("="*60)
    print("Cache Classifier Training Pipeline")
    print("="*60)

    # Check requirements
    if not args.skip_checks:
        if not check_requirements():
            sys.exit(1)

    # Determine which steps to run
    run_generate = args.all or args.generate_dataset
    run_label = args.all or args.label
    run_train = args.all or args.train
    run_demo = args.demo
    run_benchmark = args.benchmark

    # Step 1: Generate dataset
    if run_generate:
        run_command(
            [sys.executable, "dataset_generation.py"],
            "Generating dataset from public sources"
        )

    # Step 2: Label with LLM
    if run_label:
        if not args.skip_checks:
            check_ollama()

        cmd = [sys.executable, "llm_labeling.py"]
        if args.max_queries:
            cmd.extend(["--max-queries", str(args.max_queries)])

        run_command(cmd, "Labeling queries with local LLM")

    # Step 3: Train classifier
    if run_train:
        # Check if dataset exists
        dataset_path = Path("cache_classifier_data/labeled_dataset.jsonl")
        if not dataset_path.exists():
            print("\n❌ Error: Labeled dataset not found!")
            print("   Run --generate-dataset and --label first, or check the path:")
            print(f"   {dataset_path}")
            sys.exit(1)

        cmd = [
            sys.executable, "train_classifier.py",
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
        ]

        run_command(cmd, f"Training {args.model} classifier")

    # Step 4: Demo
    if run_demo:
        model_path = Path("models/cache_classifier/final_model")
        if not model_path.exists():
            print("\n❌ Error: Trained model not found!")
            print("   Run --train first to create the model")
            sys.exit(1)

        run_command(
            [sys.executable, "inference.py", "--demo"],
            "Running interactive demo"
        )

    # Step 5: Benchmark
    if run_benchmark:
        model_path = Path("models/cache_classifier/final_model")
        dataset_path = Path("cache_classifier_data/labeled_dataset.jsonl")

        if not model_path.exists():
            print("\n❌ Error: Trained model not found!")
            print("   Run --train first")
            sys.exit(1)

        if not dataset_path.exists():
            print("\n❌ Error: Labeled dataset not found!")
            sys.exit(1)

        run_command(
            [sys.executable, "inference.py", "--benchmark", str(dataset_path)],
            "Benchmarking on test set"
        )

    print("\n" + "="*60)
    print("✅ Pipeline completed successfully!")
    print("="*60)

    # Next steps
    if run_train:
        print("\nYour trained model is ready at:")
        print("  models/cache_classifier/final_model/")
        print("\nNext steps:")
        print("  1. Run demo: python quickstart.py --demo")
        print("  2. Benchmark: python quickstart.py --benchmark")
        print("  3. Integrate with GPTCache (see README_CLASSIFIER.md)")


if __name__ == "__main__":
    main()
