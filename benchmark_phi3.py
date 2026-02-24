#!/usr/bin/env python3
"""
Quick benchmark: Compare Phi-3-mini vs Llama 3 on cache classification
Shows quality and speed improvements
"""

import time
import json
from pathlib import Path
from langchain_ollama import OllamaLLM
import yaml


class ModelBenchmark:
    """Benchmark different models for cache classification."""

    def __init__(self, prompt_file: str = "prompts.yaml"):
        """Initialize benchmark."""
        with open(prompt_file, 'r') as f:
            prompts = yaml.safe_load(f)
            self.prompt_template = prompts.get('classifier_prompt_1', '')

        self.test_queries = [
            # Cacheable (should return "yes")
            ("What is the capital of France?", True),
            ("Who wrote Pride and Prejudice?", True),
            ("What does HTTP stand for?", True),
            ("What is the chemical symbol for gold?", True),
            ("What is the speed of light?", True),

            # Not cacheable (should return "no")
            ("What is the weather in London right now?", False),
            ("What is the current price of Bitcoin?", False),
            ("Write a funny poem about cats", False),
            ("What is the latest stock price of NVIDIA?", False),
            ("What time is it in Tokyo right now?", False),
        ]

    def test_model(self, model_name: str) -> dict:
        """Test a model."""
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")

        llm = OllamaLLM(
            model=model_name,
            temperature=0.1,
            num_predict=10,
        )

        results = {
            "model": model_name,
            "correct": 0,
            "total": len(self.test_queries),
            "latencies": [],
            "accuracy": 0,
            "avg_latency": 0,
        }

        for i, (query, expected_cacheable) in enumerate(self.test_queries, 1):
            prompt = self.prompt_template.format(question=query)

            start = time.time()
            response = llm.invoke(prompt).strip().lower()
            latency = time.time() - start

            predicted_cacheable = "yes" in response[:10]

            is_correct = predicted_cacheable == expected_cacheable
            if is_correct:
                results["correct"] += 1

            results["latencies"].append(latency)

            status = "✓" if is_correct else "✗"
            print(f"  {status} Query {i}: {query[:40]:40s} ({latency*1000:.0f}ms)")

        # Calculate stats
        results["accuracy"] = results["correct"] / results["total"]
        results["avg_latency"] = sum(results["latencies"]) / len(results["latencies"])

        return results

    def compare_models(self, models: list):
        """Compare multiple models."""
        print("\n" + "╔" + "="*58 + "╗")
        print("║" + " "*58 + "║")
        print("║" + "  Cache Classifier Model Benchmark".center(58) + "║")
        print("║" + " "*58 + "║")
        print("╚" + "="*58 + "╝")

        all_results = []

        for model in models:
            try:
                result = self.test_model(model)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error testing {model}: {e}")
                continue

        # Print comparison table
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)

        print(f"\n{'Model':<25} {'Accuracy':<12} {'Avg Latency':<15} {'Total Time'}")
        print("-" * 70)

        for result in all_results:
            model_name = result["model"].replace("phi3:mini", "Phi-3-mini").replace("llama2", "Llama 2")
            accuracy = result["accuracy"]
            avg_latency = result["avg_latency"] * 1000  # Convert to ms
            total_time = sum(result["latencies"]) * 1000  # Convert to ms

            print(f"{model_name:<25} {accuracy:>6.1%}        {avg_latency:>6.1f}ms        {total_time:>6.0f}ms")

        # Show winner
        if len(all_results) > 1:
            print("\n" + "="*60)
            print("WINNER:")
            print("="*60)

            # Accuracy winner
            best_accuracy = max(all_results, key=lambda x: x["accuracy"])
            print(f"✅ Best Accuracy: {best_accuracy['model']} ({best_accuracy['accuracy']:.1%})")

            # Speed winner
            best_speed = min(all_results, key=lambda x: x["avg_latency"])
            print(f"⚡ Fastest: {best_speed['model']} ({best_speed['avg_latency']*1000:.0f}ms avg)")

            # Best overall (balanced)
            best_overall = max(
                all_results,
                key=lambda x: x["accuracy"] / (1 + x["avg_latency"]/0.1)  # Balance accuracy vs speed
            )
            print(f"🏆 Best Overall: {best_overall['model']}")

        # Projection for full dataset
        print("\n" + "="*60)
        print("PROJECTION: Labeling 50,000 queries")
        print("="*60)

        for result in all_results:
            total_seconds = result["avg_latency"] * 50000
            hours = total_seconds / 3600
            print(f"{result['model']:<25} {hours:>5.1f} hours ({hours*60:>5.0f} minutes)")

        return all_results


def main():
    """Main execution."""
    benchmark = ModelBenchmark()

    # Models to test
    models_to_test = [
        "phi3:mini",      # New recommended
        # "llama2",         # Old baseline (optional, uncomment if you have it)
    ]

    # Check which models are available
    print("\n🔍 Checking available models...")
    available_models = []

    for model in models_to_test:
        try:
            llm = OllamaLLM(model=model, temperature=0.1, num_predict=1)
            llm.invoke("test")  # Quick test
            available_models.append(model)
            print(f"  ✓ {model} available")
        except Exception as e:
            print(f"  ✗ {model} not available ({e})")

    if not available_models:
        print("\n❌ No models available!")
        print("\nTo setup Phi-3-mini:")
        print("  python setup_phi3.py")
        return

    # Run benchmark
    results = benchmark.compare_models(available_models)

    # Save results
    results_file = Path("benchmark_results.json")
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types
        for r in results:
            r["latencies"] = [float(x) for x in r["latencies"]]
        json.dump(results, f, indent=2)

    print(f"\n📊 Results saved to {results_file}")


if __name__ == "__main__":
    main()
