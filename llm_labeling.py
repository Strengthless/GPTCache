"""
LLM-based Labeling for Cache Classification Dataset
Uses local Ollama LLM to label unlabeled queries.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from langchain_ollama import OllamaLLM
import time

class LLMLabeler:
    """Use local LLM to label queries for cache classification."""

    def __init__(self,
                 model_name: str = "phi3:mini",
                 prompt_template_file: str = "prompts.yaml",
                 batch_size: int = 1,
                 confidence_threshold: float = 0.8):
        """
        Initialize LLM labeler.

        Args:
            model_name: Ollama model name (default: phi3:mini - fast, high quality)
                       Other options: llama2, mistral, qwen2:7b, gemma2:9b
            prompt_template_file: Path to YAML file with prompt template
            batch_size: Number of queries to process at once
            confidence_threshold: Minimum confidence to keep label (0-1)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

        # Initialize LLM
        print(f"Initializing Ollama with model: {model_name}")
        print(f"Note: phi3:mini uses ~2-3GB VRAM, 2-3x faster than Llama 3 8B")
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.1,  # Low temperature for consistent classification
            num_predict=10,   # We only need "yes" or "no"
        )

        # Load prompt template
        self.prompt_template = self._load_prompt_template(prompt_template_file)

    def _load_prompt_template(self, template_file: str) -> str:
        """Load prompt template from YAML file."""
        template_path = Path(template_file)

        if template_path.exists():
            with open(template_path, 'r') as f:
                prompts = yaml.safe_load(f)
                return prompts.get('classifier_prompt_1', self._get_default_prompt())
        else:
            print(f"Warning: {template_file} not found, using default prompt")
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Get default classification prompt."""
        return """You are a strict caching classifier. Output only "yes" or "no" (lowercase).

Rule: Say "yes" ONLY if the correct answer is 100% identical and unchanging forever.
Say "no" for anything that can change, needs computation, or is creative.

Critical keyword triggers → immediately "no":
current, today, right now, live, price, stock, weather, time, latest, block height, winning numbers, generate, write code, create, compose, role-play, solve, calculate, compute

Examples (memorize these exact pairs):

Q: What is the capital of France?
A: yes

Q: What is the weather in Paris right now?
A: no

Q: What is the chemical symbol for gold?
A: yes

Q: What is the current price of gold?
A: no

Q: What port does HTTPS use?
A: yes

Q: What is today's exchange rate USD to EUR?
A: no

Q: Explain how HTTPS works
A: yes

Q: Generate a valid JWT token
A: no

Q: What is sin(π/3)?
A: yes

Q: What is 47 × 83?
A: no

Q: Write a Python function to...
A: no

Now classify ONLY this query.
Respond with nothing but "yes" or "no".

Query: {question}
Answer:"""

    def classify_query(self, query: str) -> Tuple[int, str, float]:
        """
        Classify a single query using LLM.

        Args:
            query: The question/query to classify

        Returns:
            Tuple of (label, reason, confidence)
            - label: 1 (cacheable) or 0 (not cacheable) or -1 (uncertain)
            - reason: Explanation
            - confidence: 0-1 score
        """
        # Format prompt
        prompt = self.prompt_template.format(question=query)

        try:
            # Get LLM response
            response = self.llm.invoke(prompt).strip().lower()

            # Parse response - be strict about exact matches
            # Remove any extra whitespace/newlines
            response_cleaned = response.split('\n')[0].strip()  # Take only first line

            # Check for exact "yes" or "no" at the start
            if response_cleaned == "yes" or response_cleaned.startswith("yes"):
                label = 1
                reason = "Static factual query with unchanging answer"
                confidence = 0.95

            elif response_cleaned == "no" or response_cleaned.startswith("no"):
                label = 0
                reason = "Dynamic, time-sensitive, computational, or creative query"
                confidence = 0.95

            else:
                # Could not parse - model didn't follow instructions
                # Check if it's trying to answer the question instead
                if len(response) > 20:
                    # Long response = probably answering instead of classifying
                    label = -1
                    reason = f"Model answered question instead of classifying. Response: {response[:100]}"
                    confidence = 0.0
                else:
                    label = -1
                    reason = f"Could not parse LLM response: {response}"
                    confidence = 0.0

            return label, reason, confidence

        except Exception as e:
            print(f"Error classifying query '{query[:50]}...': {e}")
            return -1, f"Error: {str(e)}", 0.0

    def label_dataset(self,
                     input_file: str,
                     output_file: str,
                     max_queries: Optional[int] = None,
                     skip_labeled: bool = True) -> Dict[str, Any]:
        """
        Label all unlabeled queries in dataset.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            max_queries: Maximum queries to label (None = all)
            skip_labeled: Skip queries that already have labels

        Returns:
            Statistics dictionary
        """
        print("="*60)
        print("Starting LLM Labeling")
        print("="*60)

        # Load dataset
        input_path = Path(input_file)
        output_path = Path(output_file)

        queries = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                queries.append(json.loads(line))

        print(f"Loaded {len(queries)} queries from {input_path}")

        # Filter queries to label
        to_label = []
        already_labeled = []

        for q in queries:
            if q.get("label") is None:
                to_label.append(q)
            else:
                already_labeled.append(q)

        print(f"Already labeled: {len(already_labeled)}")
        print(f"To label: {len(to_label)}")

        if max_queries:
            to_label = to_label[:max_queries]
            print(f"Limiting to {max_queries} queries")

        # Label queries
        labeled_queries = []
        stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "low_confidence": 0,
            "cacheable": 0,
            "non_cacheable": 0,
        }

        print("\nLabeling queries with LLM...")
        for query in tqdm(to_label):
            text = query["text"]

            # Classify
            label, reason, confidence = self.classify_query(text)

            # Update query
            query["label"] = label if label != -1 else None
            query["llm_reason"] = reason
            query["llm_confidence"] = confidence

            stats["total_processed"] += 1

            if label != -1 and confidence >= self.confidence_threshold:
                stats["successful"] += 1
                if label == 1:
                    stats["cacheable"] += 1
                else:
                    stats["non_cacheable"] += 1
            elif label == -1:
                stats["failed"] += 1
            else:
                stats["low_confidence"] += 1

            labeled_queries.append(query)

            # Small delay to avoid overwhelming the LLM
            time.sleep(0.05)

        # Combine all queries
        all_queries = already_labeled + labeled_queries

        # Save labeled dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            for query in all_queries:
                f.write(json.dumps(query, ensure_ascii=False) + '\n')

        print(f"\nSaved {len(all_queries)} queries to {output_path}")

        # Print statistics
        print("\n" + "="*60)
        print("Labeling Statistics")
        print("="*60)
        print(f"Total processed: {stats['total_processed']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Low confidence: {stats['low_confidence']}")
        print(f"Cacheable (yes): {stats['cacheable']}")
        print(f"Non-cacheable (no): {stats['non_cacheable']}")
        print("="*60)

        # Save stats
        stats_path = output_path.parent / "labeling_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def verify_labels(self, dataset_file: str, sample_size: int = 100):
        """
        Verify quality of labels by sampling and manual inspection.

        Args:
            dataset_file: Path to labeled dataset
            sample_size: Number of samples to show
        """
        import random

        with open(dataset_file, 'r', encoding='utf-8') as f:
            queries = [json.loads(line) for line in f]

        # Filter labeled queries
        labeled = [q for q in queries if q.get("label") is not None]

        # Sample
        samples = random.sample(labeled, min(sample_size, len(labeled)))

        print(f"\n{'='*60}")
        print(f"Sample of {len(samples)} labeled queries:")
        print(f"{'='*60}\n")

        for i, q in enumerate(samples[:20], 1):  # Show first 20
            label_str = "✓ CACHE" if q["label"] == 1 else "✗ NO CACHE"
            confidence = q.get("llm_confidence", 0)

            print(f"{i}. [{label_str}] (conf: {confidence:.2f})")
            print(f"   Query: {q['text'][:80]}...")
            print(f"   Reason: {q.get('llm_reason', 'N/A')[:60]}...")
            print()


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Label cache classification dataset with LLM")
    parser.add_argument("--input", type=str, default="cache_classifier_data/raw_dataset.jsonl",
                       help="Input JSONL file")
    parser.add_argument("--output", type=str, default="cache_classifier_data/labeled_dataset.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--max-queries", type=int, default=None,
                       help="Maximum queries to label (default: all)")
    parser.add_argument("--model", type=str,
                       default="phi3:mini",
                       help="Ollama model name (default: phi3:mini, 2-3x faster than Llama 3, ~2-3GB VRAM). "
                            "Other options: llama2, mistral, qwen2:7b, gemma2:9b, phi3:3.8b")
    parser.add_argument("--verify", action="store_true",
                       help="Verify labels after labeling")

    args = parser.parse_args()

    # Initialize labeler
    labeler = LLMLabeler(model_name=args.model)

    # Label dataset
    stats = labeler.label_dataset(
        input_file=args.input,
        output_file=args.output,
        max_queries=args.max_queries
    )

    # Verify if requested
    if args.verify:
        labeler.verify_labels(args.output, sample_size=100)

    print("\nNext step: Run train_classifier.py to fine-tune BERT model")


if __name__ == "__main__":
    main()
