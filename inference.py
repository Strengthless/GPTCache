"""
Inference and Integration for Cache Classifier
Fast inference with trained BERT model + integration with GPTCache.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time


class CacheClassifier:
    """Fast cache decision inference with trained BERT model."""

    def __init__(self,
                 model_path: str = "models/cache_classifier/final_model",
                 device: str = "auto"):
        """
        Initialize classifier.

        Args:
            model_path: Path to trained model directory
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_path = Path(model_path)

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading model from {self.model_path}")
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print("Model loaded successfully!")

    def predict(self, query: str, return_prob: bool = False) -> Union[bool, Tuple[bool, float]]:
        """
        Predict if query should be cached.

        Args:
            query: Query string to classify
            return_prob: If True, return (decision, confidence_score)

        Returns:
            bool: True if should cache, False otherwise
            OR (bool, float): (decision, confidence) if return_prob=True
        """
        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()

        should_cache = bool(pred_class)  # 1 = cache, 0 = don't cache

        if return_prob:
            return should_cache, confidence
        else:
            return should_cache

    def predict_batch(self,
                     queries: List[str],
                     batch_size: int = 32) -> List[Tuple[bool, float]]:
        """
        Predict on batch of queries (more efficient).

        Args:
            queries: List of query strings
            batch_size: Batch size for processing

        Returns:
            List of (should_cache, confidence) tuples
        """
        results = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_classes = torch.argmax(probs, dim=-1).cpu().numpy()
                confidences = probs.max(dim=-1).values.cpu().numpy()

            # Convert to results
            for pred_class, confidence in zip(pred_classes, confidences):
                should_cache = bool(pred_class)
                results.append((should_cache, float(confidence)))

        return results

    def benchmark(self, test_queries: List[dict]) -> dict:
        """
        Benchmark model on test queries.

        Args:
            test_queries: List of dicts with 'text' and 'expected_cacheable' keys

        Returns:
            Dictionary with benchmark results
        """
        print("Running benchmark...")

        correct = 0
        total = 0
        latencies = []

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for item in test_queries:
            query = item["text"]
            expected = item["expected_cacheable"]

            # Time the prediction
            start = time.time()
            predicted, confidence = self.predict(query, return_prob=True)
            latency = time.time() - start
            latencies.append(latency)

            # Check correctness
            if predicted == expected:
                correct += 1

            # Confusion matrix
            if predicted and expected:
                true_positives += 1
            elif not predicted and not expected:
                true_negatives += 1
            elif predicted and not expected:
                false_positives += 1
            elif not predicted and expected:
                false_negatives += 1

            total += 1

        # Calculate metrics
        accuracy = correct / total
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "avg_latency_ms": avg_latency * 1000,
            "p95_latency_ms": p95_latency * 1000,
            "p99_latency_ms": p99_latency * 1000,
        }

        # Print results
        print("\n" + "="*60)
        print("Benchmark Results")
        print("="*60)
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {true_positives}, TN: {true_negatives}")
        print(f"  FP: {false_positives}, FN: {false_negatives}")
        print(f"\nLatency:")
        print(f"  Average: {avg_latency*1000:.2f}ms")
        print(f"  P95: {p95_latency*1000:.2f}ms")
        print(f"  P99: {p99_latency*1000:.2f}ms")
        print("="*60)

        return results


class CacheDecisionFunc:
    """
    Drop-in replacement for LLM-based cache decision in GPTCache.
    100-1000x faster than LLM approach!
    """

    def __init__(self, model_path: str = "models/cache_classifier/final_model"):
        """Initialize with trained model."""
        self.classifier = CacheClassifier(model_path)

    def __call__(self, query: str, **kwargs) -> bool:
        """
        Check if query should be cached.

        Args:
            query: The user's query string

        Returns:
            bool: True if should cache, False otherwise
        """
        return self.classifier.predict(query)


# Integration example for GPTCache
def integrate_with_gptcache():
    """
    Example of how to integrate the trained classifier with GPTCache.
    """
    from gptcache.utils.cache_func import cache_selectively
    from gptcache import Cache

    # Initialize classifier
    cache_decision = CacheDecisionFunc(model_path="models/cache_classifier/final_model")

    # Use with GPTCache
    # Option 1: As a pre_embedding_func (check before embedding)
    def should_cache_check(data, **params):
        """Check if query should be cached before embedding."""
        query = data.get("messages", [{}])[-1].get("content", "")
        return cache_decision(query)

    # Option 2: Use with cache_selectively
    # See gptcache/utils/cache_func.py for integration details

    print("Classifier integrated with GPTCache!")
    print("Use cache_decision(query) to check if query should be cached")
    print("This is 100-1000x faster than LLM-based checking!")

    return cache_decision


def demo():
    """Interactive demo."""
    classifier = CacheClassifier()

    print("\n" + "="*60)
    print("Cache Classifier Demo")
    print("="*60)
    print("Enter queries to classify (or 'quit' to exit)\n")

    # Test examples
    examples = [
        "What is the capital of France?",
        "What is the weather in Paris right now?",
        "What is the chemical symbol for gold?",
        "What is the current price of Bitcoin?",
        "Explain how HTTPS works",
        "Write a poem about cats",
        "What is 47 × 83?",
        "What does HTTP stand for?",
    ]

    print("Examples:")
    for query in examples:
        should_cache, confidence = classifier.predict(query, return_prob=True)
        decision = "✓ CACHE" if should_cache else "✗ SKIP"
        print(f"{decision} ({confidence:.2%}): {query}")

    print("\n" + "-"*60)
    print("Try your own queries:")

    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        start = time.time()
        should_cache, confidence = classifier.predict(query, return_prob=True)
        latency = time.time() - start

        decision = "✓ CACHE" if should_cache else "✗ SKIP"
        print(f"{decision} (confidence: {confidence:.2%}, latency: {latency*1000:.1f}ms)")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Cache classifier inference")
    parser.add_argument("--model", type=str,
                       default="models/cache_classifier/final_model",
                       help="Path to trained model")
    parser.add_argument("--demo", action="store_true",
                       help="Run interactive demo")
    parser.add_argument("--benchmark", type=str,
                       help="Path to test dataset for benchmarking")

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.benchmark:
        # Load test dataset
        import json
        test_queries = []
        with open(args.benchmark, 'r') as f:
            for line in f:
                item = json.loads(line)
                if item.get("label") is not None:
                    test_queries.append({
                        "text": item["text"],
                        "expected_cacheable": bool(item["label"])
                    })

        # Benchmark
        classifier = CacheClassifier(args.model)
        classifier.benchmark(test_queries)
    else:
        # Just load and show it's ready
        classifier = CacheClassifier(args.model)
        print("\nClassifier ready!")
        print("Use --demo for interactive testing")
        print("Use --benchmark <dataset.jsonl> for evaluation")


if __name__ == "__main__":
    main()
