#!/usr/bin/env python3
"""
Simple diagnostic without unicode issues
"""

import json
from collections import Counter

with open('cache_classifier_data/labeled_dataset.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

print("="*60)
print("LABELED DATA DIAGNOSTICS")
print("="*60)

# Overall stats
labeled = [d for d in data if d.get('label') is not None and d.get('label') != -1]
cacheable = [d for d in labeled if d['label'] == 1]
not_cacheable = [d for d in labeled if d['label'] == 0]

print(f"\nTotal queries: {len(data)}")
print(f"Labeled queries: {len(labeled)}")
print(f"Cacheable (1): {len(cacheable)} ({len(cacheable)/len(labeled)*100:.1f}%)")
print(f"Not cacheable (0): {len(not_cacheable)} ({len(not_cacheable)/len(labeled)*100:.1f}%)")

# By source
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

# MS MARCO samples (labeled as NOT cacheable)
ms_marco_not_cacheable = [d for d in data if d.get('source') == 'ms_marco' and d.get('label') == 0][:20]

print("\n" + "="*60)
print("MS MARCO LABELED AS NOT CACHEABLE (Sample 20):")
print("="*60)
print("(Check if these should actually be cacheable)")

for i, item in enumerate(ms_marco_not_cacheable, 1):
    print(f"\n{i}. {item['text']}")

# MS MARCO samples (labeled AS cacheable - to verify correct)
ms_marco_cacheable = [d for d in data if d.get('source') == 'ms_marco' and d.get('label') == 1][:20]

print("\n" + "="*60)
print("MS MARCO LABELED AS CACHEABLE (Sample 20):")
print("="*60)
print("(Verify these look correct)")

for i, item in enumerate(ms_marco_cacheable, 1):
    print(f"\n{i}. {item['text']}")

# Low confidence
low_conf = [d for d in data if d.get('llm_confidence', 1.0) < 0.8]
print("\n" + "="*60)
print("LOW CONFIDENCE QUERIES:")
print("="*60)
print(f"Total low confidence: {len(low_conf)}")

if low_conf:
    print("\nSample of low confidence queries:")
    for i, item in enumerate(low_conf[:10], 1):
        conf = item.get('llm_confidence', 0)
        reason = item.get('llm_reason', 'Unknown')
        print(f"\n{i}. Confidence: {conf:.2f}")
        print(f"   Query: {item['text'][:80]}")
        print(f"   Reason: {reason[:60]}")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("="*60)

if len(low_conf) > 0:
    print(f"\n1. Re-label {len(low_conf)} low confidence queries:")
    print("   python relabel_low_confidence.py")

if sources.get('ms_marco', {}).get('cacheable', 0) / (sources.get('ms_marco', {}).get('cacheable', 0) + sources.get('ms_marco', {}).get('not_cacheable', 1)) < 0.3:
    print("\n2. MS MARCO has only 16% cacheable (should be ~40-50%)")
    print("   Phi-3-mini might be too conservative.")
    print("   Options:")
    print("   a) Accept current labels (might be fine for training)")
    print("   b) Try a different model (qwen2:7b, gemma2:9b)")
    print("   c) Manually review sample queries above")

print("\n" + "="*60)
