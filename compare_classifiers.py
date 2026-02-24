"""
Comparison Script: BERT Classifier vs. LLM-based Classification
Benchmarks both approaches on the same test set to show speedup and accuracy improvements.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_ollama import OllamaLLM
from inference import CacheClassifier
import yaml


class ComparisonBenchmark:
    """Compare BERT classifier vs LLM-based classification."""

    def __init__(self,
                 bert_model_path: str = "models/cache_classifier/final_model",
                 llm_model_name: str = "hf.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M",
                 prompt_file: str = "prompts.yaml"):
        """Initialize both classifiers."""

        print("Initializing classifiers...")

        # BERT classifier
        print("Loading BERT classifier...")
        self.bert_classifier = CacheClassifier(bert_model_path)

        # LLM classifier
        print("Loading LLM classifier...")
        self.llm = OllamaLLM(
            model=llm_model_name,
            temperature=0.1,
            num_predict=10,
        )

        # Load prompt template
        with open(prompt_file, 'r') as f:
            prompts = yaml.safe_load(f)
            self.llm_prompt = prompts.get('classifier_prompt_1', '')

        print("Both classifiers ready!\n")

    def classify_with_llm(self, query: str) -> Tuple[bool, float]:
        """Classify with LLM."""
        prompt = self.llm_prompt.format(question=query)

        start = time.time()
        response = self.llm.invoke(prompt).strip().lower()
        latency = time.time() - start

        # Parse response
        if "yes" in response[:10]:
            return True, latency
        elif "no" in response[:10]:
            return False, latency
        else:
            return False, latency  # Default to not caching if uncertain

    def classify_with_bert(self, query: str) -> Tuple[bool, float]:
        """Classify with BERT."""
        start = time.time()
        should_cache, confidence = self.bert_classifier.predict(query, return_prob=True)
        latency = time.time() - start

        return should_cache, latency

    def run_comparison(self, test_queries: List[Dict], max_samples: int = 100):
        """
        Run side-by-side comparison.

        Args:
            test_queries: List of dicts with 'text' and 'label' keys
            max_samples: Maximum number of queries to test
        """
        print("="*60)
        print(f"Running Comparison Benchmark ({max_samples} samples)")
        print("="*60)

        # Sample test queries
        if len(test_queries) > max_samples:
            import random
            test_queries = random.sample(test_queries, max_samples)

        # Results storage
        llm_results = {
            'correct': 0,
            'total': 0,
            'latencies': [],
            'predictions': []
        }

        bert_results = {
            'correct': 0,
            'total': 0,
            'latencies': [],
            'predictions': []
        }

        print("\nTesting queries...\n")

        for i, item in enumerate(test_queries):
            query = item['text']
            expected = bool(item['label'])

            print(f"[{i+1}/{len(test_queries)}] {query[:60]}...")

            # Test LLM
            llm_pred, llm_latency = self.classify_with_llm(query)
            llm_results['predictions'].append(llm_pred)
            llm_results['latencies'].append(llm_latency)
            llm_results['total'] += 1
            if llm_pred == expected:
                llm_results['correct'] += 1

            # Test BERT
            bert_pred, bert_latency = self.classify_with_bert(query)
            bert_results['predictions'].append(bert_pred)
            bert_results['latencies'].append(bert_latency)
            bert_results['total'] += 1
            if bert_pred == expected:
                bert_results['correct'] += 1

            # Show comparison
            llm_symbol = "✓" if llm_pred == expected else "✗"
            bert_symbol = "✓" if bert_pred == expected else "✗"

            print(f"  LLM:  {llm_symbol} {llm_pred} ({llm_latency*1000:.1f}ms)")
            print(f"  BERT: {bert_symbol} {bert_pred} ({bert_latency*1000:.1f}ms)")
            print()

        # Calculate metrics
        self._print_comparison_results(llm_results, bert_results)

    def _print_comparison_results(self, llm_results: Dict, bert_results: Dict):
        """Print comparison results."""

        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)

        # Accuracy
        llm_acc = llm_results['correct'] / llm_results['total']
        bert_acc = bert_results['correct'] / bert_results['total']

        print("\nAccuracy:")
        print(f"  LLM:  {llm_acc:.1%} ({llm_results['correct']}/{llm_results['total']})")
        print(f"  BERT: {bert_acc:.1%} ({bert_results['correct']}/{bert_results['total']})")
        print(f"  Improvement: {(bert_acc - llm_acc)*100:+.1f} percentage points")

        # Latency
        llm_latencies = llm_results['latencies']
        bert_latencies = bert_results['latencies']

        llm_avg = np.mean(llm_latencies) * 1000
        llm_p95 = np.percentile(llm_latencies, 95) * 1000
        bert_avg = np.mean(bert_latencies) * 1000
        bert_p95 = np.percentile(bert_latencies, 95) * 1000

        print("\nLatency (Average):")
        print(f"  LLM:  {llm_avg:.1f}ms")
        print(f"  BERT: {bert_avg:.1f}ms")
        print(f"  Speedup: {llm_avg / bert_avg:.1f}x faster")

        print("\nLatency (P95):")
        print(f"  LLM:  {llm_p95:.1f}ms")
        print(f"  BERT: {bert_p95:.1f}ms")
        print(f"  Speedup: {llm_p95 / bert_p95:.1f}x faster")

        # Memory estimate
        print("\nMemory (Inference):")
        print(f"  LLM:  ~8 GB (Llama 3 8B Q4)")
        print(f"  BERT: ~200-600 MB (DistilBERT/ModernBERT)")
        print(f"  Reduction: ~13-40x less memory")

        # Total cost
        total_llm_time = sum(llm_latencies)
        total_bert_time = sum(bert_latencies)

        print(f"\nTotal Time for {llm_results['total']} queries:")
        print(f"  LLM:  {total_llm_time:.1f}s")
        print(f"  BERT: {total_bert_time:.1f}s")
        print(f"  Time saved: {total_llm_time - total_bert_time:.1f}s ({(1 - total_bert_time/total_llm_time)*100:.0f}%)")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"✅ BERT is {llm_avg / bert_avg:.0f}x FASTER")
        print(f"✅ BERT uses {13:.0f}-{40:.0f}x LESS MEMORY")

        if bert_acc >= llm_acc:
            print(f"✅ BERT is MORE ACCURATE (+{(bert_acc - llm_acc)*100:.1f}pp)")
        else:
            print(f"⚠️  BERT is slightly less accurate ({(bert_acc - llm_acc)*100:.1f}pp)")

        print("\n💡 Recommendation: Use BERT classifier for production!")
        print("="*60)


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare BERT vs LLM classification")
    parser.add_argument("--dataset", type=str,
                       default="cache_classifier_data/labeled_dataset.jsonl",
                       help="Test dataset")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples to test (default: 100)")
    parser.add_argument("--bert-model", type=str,
                       default="models/cache_classifier/final_model",
                       help="BERT model path")
    parser.add_argument("--llm-model", type=str,
                       default="hf.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M",
                       help="LLM model name")

    args = parser.parse_args()

    # Load test dataset
    print(f"Loading test dataset from {args.dataset}...")
    test_queries = []
    with open(args.dataset, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get('label') is not None and item['label'] in [0, 1]:
                test_queries.append({
                    'text': item['text'],
                    'label': item['label']
                })

    print(f"Loaded {len(test_queries)} labeled queries")

    # Initialize benchmark
    benchmark = ComparisonBenchmark(
        bert_model_path=args.bert_model,
        llm_model_name=args.llm_model
    )

    # Run comparison
    benchmark.run_comparison(test_queries, max_samples=args.samples)


if __name__ == "__main__":
    main()
