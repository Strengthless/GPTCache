#!/usr/bin/env python3
"""
Clean dataset by removing failed/unlabeled queries
"""

import json
from pathlib import Path

# Load dataset
with open('cache_classifier_data/labeled_dataset.jsonl', 'r', encoding='utf-8') as f:
    all_queries = [json.loads(line) for line in f]

# Filter to only successfully labeled
good_queries = [q for q in all_queries if q.get('label') in [0, 1]]
failed_queries = [q for q in all_queries if q.get('label') not in [0, 1]]

print(f"Total queries: {len(all_queries)}")
print(f"Successfully labeled: {len(good_queries)}")
print(f"Failed/unlabeled: {len(failed_queries)}")

# Stats
cacheable = len([q for q in good_queries if q['label'] == 1])
not_cacheable = len([q for q in good_queries if q['label'] == 0])

print(f"\nSuccessful labels:")
print(f"  Cacheable: {cacheable} ({cacheable/len(good_queries)*100:.1f}%)")
print(f"  Not cacheable: {not_cacheable} ({not_cacheable/len(good_queries)*100:.1f}%)")

# Save cleaned version
output_file = Path('cache_classifier_data/labeled_dataset_clean.jsonl')
with open(output_file, 'w', encoding='utf-8') as f:
    for q in good_queries:
        f.write(json.dumps(q, ensure_ascii=False) + '\n')

print(f"\nCleaned dataset saved to: {output_file}")
print(f"Ready to train with {len(good_queries)} high-quality labeled queries!")
