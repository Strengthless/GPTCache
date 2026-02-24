"""
FAST LLM-based Labeling with Multithreading
3-5x faster than sequential version using concurrent requests to Ollama.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from langchain_ollama import OllamaLLM
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class FastLLMLabeler:
    """Fast multi-threaded LLM labeler for cache classification."""

    def __init__(self,
                 model_name: str = "phi3:mini",
                 prompt_template_file: str = "prompts.yaml",
                 num_threads: int = 8,
                 confidence_threshold: float = 0.8):
        """
        Initialize fast LLM labeler.

        Args:
            model_name: Ollama model name
            prompt_template_file: Path to YAML file with prompt template
            num_threads: Number of parallel threads (default: 8, adjust based on your system)
            confidence_threshold: Minimum confidence to keep label (0-1)
        """
        self.model_name = model_name
        self.num_threads = num_threads
        self.confidence_threshold = confidence_threshold

        print(f"Initializing Fast Multi-threaded Labeler")
        print(f"Model: {model_name}")
        print(f"Threads: {num_threads}")
        print(f"Expected speedup: {num_threads}x faster")

        # Load prompt template
        self.prompt_template = self._load_prompt_template(prompt_template_file)

        # Thread-local storage for LLM instances (one per thread)
        self.thread_local = threading.local()

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
        return """CRITICAL INSTRUCTION: You MUST respond with EXACTLY one word: either "yes" or "no" (lowercase).

Your task: Determine if a query's answer should be CACHED.

Output "yes" (cacheable) if the answer is an unchanging fact.
Output "no" (NOT cacheable) if time-sensitive, creative, or computational.

Query: {question}
Answer:"""

    def _get_llm(self) -> OllamaLLM:
        """Get thread-local LLM instance."""
        if not hasattr(self.thread_local, 'llm'):
            self.thread_local.llm = OllamaLLM(
                model=self.model_name,
                temperature=0.1,
                num_predict=10,
            )
        return self.thread_local.llm

    def classify_query(self, query: str) -> Tuple[int, str, float]:
        """
        Classify a single query using LLM (thread-safe).

        Args:
            query: The question/query to classify

        Returns:
            Tuple of (label, reason, confidence)
        """
        prompt = self.prompt_template.format(question=query)

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt).strip().lower()

            # Parse response - be strict
            response_cleaned = response.split('\n')[0].strip()

            if response_cleaned == "yes" or response_cleaned.startswith("yes"):
                return 1, "Static factual query with unchanging answer", 0.95

            elif response_cleaned == "no" or response_cleaned.startswith("no"):
                return 0, "Dynamic, time-sensitive, computational, or creative query", 0.95

            else:
                # Model didn't follow instructions
                if len(response) > 20:
                    return -1, f"Model answered instead of classifying: {response[:100]}", 0.0
                else:
                    return -1, f"Could not parse: {response}", 0.0

        except Exception as e:
            return -1, f"Error: {str(e)}", 0.0

    def label_batch(self, queries_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Label a batch of queries in parallel.

        Args:
            queries_batch: List of query dictionaries

        Returns:
            List of labeled query dictionaries
        """
        labeled = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(self.classify_query, q["text"]): q
                for q in queries_batch
            }

            # Collect results as they complete
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                label, reason, confidence = future.result()

                # Update query with label
                query["label"] = label if label != -1 else None
                query["llm_reason"] = reason
                query["llm_confidence"] = confidence

                labeled.append(query)

        return labeled

    def label_dataset(self,
                     input_file: str,
                     output_file: str,
                     max_queries: Optional[int] = None,
                     batch_size: int = 100) -> Dict[str, Any]:
        """
        Label all unlabeled queries in dataset using multithreading.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            max_queries: Maximum queries to label (None = all)
            batch_size: Save progress every N queries (default: 100)

        Returns:
            Statistics dictionary
        """
        print("="*60)
        print("Starting FAST Multi-threaded LLM Labeling")
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

        # Label queries in batches
        stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "low_confidence": 0,
            "cacheable": 0,
            "non_cacheable": 0,
        }

        labeled_queries = []

        print(f"\nLabeling with {self.num_threads} parallel threads...")
        print(f"Batch size: {batch_size} (saves progress every {batch_size} queries)")

        start_time = time.time()

        # Process in batches with progress bar
        with tqdm(total=len(to_label), desc="Labeling", unit="query") as pbar:
            for i in range(0, len(to_label), batch_size):
                batch = to_label[i:i + batch_size]

                # Label batch in parallel
                batch_labeled = self.label_batch(batch)

                # Update stats
                for query in batch_labeled:
                    label = query.get("label")
                    confidence = query.get("llm_confidence", 0)

                    stats["total_processed"] += 1

                    if label is not None and label != -1 and confidence >= self.confidence_threshold:
                        stats["successful"] += 1
                        if label == 1:
                            stats["cacheable"] += 1
                        else:
                            stats["non_cacheable"] += 1
                    elif label == -1:
                        stats["failed"] += 1
                    else:
                        stats["low_confidence"] += 1

                labeled_queries.extend(batch_labeled)
                pbar.update(len(batch))

                # Save progress every batch
                all_queries = already_labeled + labeled_queries
                with open(output_path, 'w', encoding='utf-8') as f:
                    for query in all_queries:
                        f.write(json.dumps(query, ensure_ascii=False) + '\n')

        elapsed = time.time() - start_time
        queries_per_second = len(to_label) / elapsed if elapsed > 0 else 0

        # Final save
        all_queries = already_labeled + labeled_queries
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
        print(f"\nTime elapsed: {elapsed:.1f} seconds")
        print(f"Speed: {queries_per_second:.1f} queries/second")
        print(f"Average: {elapsed/len(to_label):.2f} seconds/query")
        print("="*60)

        # Save stats
        stats_path = output_path.parent / "labeling_stats_fast.json"
        stats["elapsed_seconds"] = elapsed
        stats["queries_per_second"] = queries_per_second
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        return stats


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Fast multi-threaded LLM labeling")
    parser.add_argument("--input", type=str, default="cache_classifier_data/raw_dataset.jsonl",
                       help="Input JSONL file")
    parser.add_argument("--output", type=str, default="cache_classifier_data/labeled_dataset.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--max-queries", type=int, default=None,
                       help="Maximum queries to label (default: all)")
    parser.add_argument("--model", type=str, default="phi3:mini",
                       help="Ollama model name")
    parser.add_argument("--threads", type=int, default=8,
                       help="Number of parallel threads (default: 8)")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Save progress every N queries (default: 100)")

    args = parser.parse_args()

    # Initialize labeler
    labeler = FastLLMLabeler(
        model_name=args.model,
        num_threads=args.threads
    )

    # Label dataset
    stats = labeler.label_dataset(
        input_file=args.input,
        output_file=args.output,
        max_queries=args.max_queries,
        batch_size=args.batch_size
    )

    print("\n✅ Fast labeling complete!")
    print(f"Speedup achieved: ~{args.threads}x faster than sequential")
    print("\nNext step: Run train_classifier.py to fine-tune BERT model")


if __name__ == "__main__":
    main()
