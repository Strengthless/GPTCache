#!/usr/bin/env python3
"""
Quick diagnostic script to check labeled data quality
"""

import json
from collections import Counter

# Load labeled dataset
with open('cache_classifier_data/labeled_dataset.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

print("="*60)
print("LABELED DATA DIAGNOSTICS")
print("="*60)

# Overall stats
total = len(data)
labeled = [d for d in data if d.get('label') is not None]
cacheable = [d for d in labeled if d['label'] == 1]
not_cacheable = [d for d in labeled if d['label'] == 0]

print(f"\nTotal queries: {total}")
print(f"Labeled queries: {len(labeled)}")
print(f"Cacheable (1): {len(cacheable)} ({len(cacheable)/len(labeled)*100:.1f}%)")
print(f"Not cacheable (0): {len(not_cacheable)} ({len(not_cacheable)/len(labeled)*100:.1f}%)")

# Check by source
print("\n" + "="*60)
print("BY SOURCE:")
print("="*60)

sources = {}
for d in labeled:
    source = d.get('source', 'unknown')
    label = d['label']

    if source not in sources:
        sources[source] = {'cacheable': 0, 'not_cacheable': 0}

    if label == 1:
        sources[source]['cacheable'] += 1
    else:
        sources[source]['not_cacheable'] += 1

for source, counts in sorted(sources.items()):
    total_source = counts['cacheable'] + counts['not_cacheable']
    cache_pct = counts['cacheable'] / total_source * 100 if total_source > 0 else 0
    print(f"\n{source}:")
    print(f"  Cacheable: {counts['cacheable']} ({cache_pct:.1f}%)")
    print(f"  Not cacheable: {counts['not_cacheable']} ({100-cache_pct:.1f}%)")

# Show recently labeled by LLM
llm_labeled = [d for d in data if d.get('llm_reason')]

print("\n" + "="*60)
print("RECENTLY LABELED BY LLM (First 20):")
print("="*60)

for i, item in enumerate(llm_labeled[:20], 1):
    label_str = "✓ CACHE" if item['label'] == 1 else "✗ SKIP"
    source = item.get('source', 'unknown')
    print(f"\n{i}. [{label_str}] ({source})")
    print(f"   Query: {item['text'][:80]}")
    if len(item['text']) > 80:
        print(f"          ...{item['text'][-30:]}")

# Show the 18 cacheable ones (if they exist)
llm_cacheable = [d for d in llm_labeled if d['label'] == 1]

if llm_cacheable:
    print("\n" + "="*60)
    print(f"THE {len(llm_cacheable)} QUERIES LABELED AS CACHEABLE:")
    print("="*60)

    for i, item in enumerate(llm_cacheable, 1):
        print(f"\n{i}. {item['text'][:100]}")
        if len(item['text']) > 100:
            print(f"   ...{item['text'][-50:]}")

# Show samples of MS MARCO that were labeled not cacheable (suspicious!)
ms_marco_not_cacheable = [d for d in llm_labeled if d.get('source') == 'ms_marco' and d['label'] == 0]

if ms_marco_not_cacheable:
    print("\n" + "="*60)
    print("MS MARCO LABELED AS NOT CACHEABLE (Sample 10):")
    print("="*60)
    print("(These might be incorrectly labeled!)")

    for i, item in enumerate(ms_marco_not_cacheable[:10], 1):
        print(f"\n{i}. {item['text']}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

if len(llm_cacheable) < 100:
    print("\n⚠️  WARNING: Very few cacheable labels!")
    print("   This suggests Phi-3-mini is being too conservative.")
    print("\nPossible causes:")
    print("  1. Phi-3-mini is defaulting to 'no' for uncertainty")
    print("  2. The prompt might be too strict")
    print("  3. MS MARCO queries might have unexpected characteristics")

    print("\n💡 SOLUTIONS:")
    print("  1. Test Phi-3-mini manually (see suggestions below)")
    print("  2. Adjust the prompt to be less strict")
    print("  3. Use a different model (try qwen2:7b or gemma2:9b)")
    print("  4. Manually verify a few MS MARCO queries")

print("\n" + "="*60)
