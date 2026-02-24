#!/usr/bin/env python3
"""
Re-label queries with low confidence or failed labels.
"""

import json
from pathlib import Path
from llm_labeling_fast import FastLLMLabeler
from tqdm import tqdm


def main():
    """Re-label low confidence queries."""

    print("="*60)
    print("Re-labeling Low Confidence Queries")
    print("="*60)

    # Load current dataset
    dataset_file = Path("cache_classifier_data/labeled_dataset.jsonl")

    with open(dataset_file, 'r', encoding='utf-8') as f:
        all_queries = [json.loads(line) for line in f]

    # Find low confidence or failed queries
    low_confidence = []
    good_queries = []

    for q in all_queries:
        confidence = q.get('llm_confidence', 1.0)
        label = q.get('label')

        # Re-label if:
        # 1. Label is None (failed)
        # 2. Confidence < 0.8
        # 3. Label is -1 (parse error)
        if label is None or label == -1 or confidence < 0.8:
            low_confidence.append(q)
        else:
            good_queries.append(q)

    print(f"\nTotal queries: {len(all_queries)}")
    print(f"Good labels: {len(good_queries)}")
    print(f"Low confidence/failed: {len(low_confidence)}")

    if not low_confidence:
        print("\n✅ No low confidence queries to re-label!")
        return

    # Show samples
    print("\n" + "="*60)
    print("Sample of queries to re-label:")
    print("="*60)
    for i, q in enumerate(low_confidence[:10], 1):
        conf = q.get('llm_confidence', 0)
        reason = q.get('llm_reason', 'Unknown')
        print(f"\n{i}. Confidence: {conf:.2f}")
        print(f"   Query: {q['text'][:80]}")
        print(f"   Reason: {reason[:60]}")

    # Ask for confirmation
    print(f"\n{'='*60}")
    response = input(f"\nRe-label {len(low_confidence)} queries? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("Cancelled.")
        return

    # Initialize labeler with more threads for speed
    print("\nInitializing labeler...")
    labeler = FastLLMLabeler(
        model_name="phi3:mini",
        num_threads=16
    )

    # Re-label in batches
    print(f"\nRe-labeling {len(low_confidence)} queries...")
    relabeled = labeler.label_batch(low_confidence)

    # Combine with good queries
    final_dataset = good_queries + relabeled

    # Save
    with open(dataset_file, 'w', encoding='utf-8') as f:
        for q in final_dataset:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    # Stats
    successful = len([q for q in relabeled if q.get('label') in [0, 1]])
    still_failed = len([q for q in relabeled if q.get('label') not in [0, 1]])
    cacheable = len([q for q in relabeled if q.get('label') == 1])
    not_cacheable = len([q for q in relabeled if q.get('label') == 0])

    print("\n" + "="*60)
    print("Re-labeling Results")
    print("="*60)
    print(f"Successfully re-labeled: {successful}")
    print(f"Still failed: {still_failed}")
    print(f"Cacheable (yes): {cacheable}")
    print(f"Non-cacheable (no): {not_cacheable}")
    print(f"\nDataset saved to: {dataset_file}")
    print("\n✅ Done! Run diagnose_labels.py to check results.")


if __name__ == "__main__":
    main()
