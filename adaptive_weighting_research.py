#!/usr/bin/env python3
"""
Adaptive weighting research for hybrid retrieval (cosine + BM25).

This script:
1) Builds a hybrid Qdrant index with cacheable (label=1) queries.
2) Runs dense and sparse retrieval independently for each evaluation query.
3) Sweeps fusion weights alpha in [0, 1], where:
      fused = alpha * dense_norm + (1 - alpha) * bm25_norm
4) Calibrates cache-hit threshold on validation split for each alpha.
5) Evaluates on test split and plots metrics vs weight.

Primary outputs:
- output_dir/weight_sweep_metrics.csv
- output_dir/weight_sweep_summary.json
- output_dir/weight_sweep_plot.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from fastembed import SparseTextEmbedding
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from qdrant_client import QdrantClient, models


@dataclass(frozen=True)
class LabeledQuery:
    text: str
    label: int


@dataclass
class QuerySignals:
    label: int
    query_text: str
    top_fused_score: float
    top_doc_text: str
    latency_ms: float


def _question_to_text(question_obj) -> str:
    if question_obj is None:
        return ""
    if isinstance(question_obj, str):
        return question_obj
    if hasattr(question_obj, "content"):
        return str(question_obj.content)
    return str(question_obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive weighting research for hybrid search")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cache_classifier_data/labeled_dataset.jsonl",
        help="Path to labeled dataset jsonl (must include fields: text, label)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/adaptive_weighting",
        help="Directory to save CSV/JSON/plot outputs",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=2500,
        help="Max labeled queries to evaluate (balanced sample)",
    )
    parser.add_argument(
        "--max-corpus",
        type=int,
        default=9000,
        help="Max cacheable (label=1) queries used to build the cache index",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-k to retrieve from each modality (dense and bm25)",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.05,
        help="Sweep step size for cosine weight alpha in [0, 1]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling/splits",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.25,
        help="Validation split ratio (from sampled query set)",
    )
    parser.add_argument(
        "--english-only",
        dest="english_only",
        action="store_true",
        default=True,
        help="Use only English-like queries for testing (default: enabled)",
    )
    parser.add_argument(
        "--no-english-only",
        dest="english_only",
        action="store_false",
        help="Disable English-only filtering",
    )
    return parser.parse_args()


def load_labeled_queries(path: Path) -> List[LabeledQuery]:
    queries: List[LabeledQuery] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            label = item.get("label")
            text = item.get("text")
            if label not in (0, 1):
                continue
            if not text or not isinstance(text, str):
                continue
            queries.append(LabeledQuery(text=text.strip(), label=int(label)))
    if not queries:
        raise ValueError(f"No labeled records found in {path}")
    return queries


def is_english_like(text: str, min_ascii_ratio: float = 0.90, min_ascii_alpha_ratio: float = 0.85) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False

    ascii_chars = sum(1 for ch in compact if ord(ch) < 128)
    ascii_ratio = ascii_chars / len(compact)
    if ascii_ratio < min_ascii_ratio:
        return False

    alpha_chars = [ch for ch in compact if ch.isalpha()]
    if not alpha_chars:
        return False

    ascii_alpha = sum(1 for ch in alpha_chars if ord(ch) < 128)
    ascii_alpha_ratio = ascii_alpha / len(alpha_chars)
    return ascii_alpha_ratio >= min_ascii_alpha_ratio


def filter_english_records(records: Sequence[LabeledQuery]) -> List[LabeledQuery]:
    return [record for record in records if is_english_like(record.text)]


def sample_balanced_queries(
    records: Sequence[LabeledQuery], max_queries: int, seed: int
) -> List[LabeledQuery]:
    positives = [item for item in records if item.label == 1]
    negatives = [item for item in records if item.label == 0]

    if not positives or not negatives:
        raise ValueError("Need both label=1 and label=0 samples for evaluation")

    half = max_queries // 2
    rng = random.Random(seed)

    sampled_pos = positives if len(positives) <= half else rng.sample(positives, half)
    sampled_neg = negatives if len(negatives) <= half else rng.sample(negatives, half)

    sampled = sampled_pos + sampled_neg
    rng.shuffle(sampled)
    return sampled


def split_stratified(
    records: Sequence[LabeledQuery], val_ratio: float, seed: int
) -> Tuple[List[LabeledQuery], List[LabeledQuery]]:
    rng = random.Random(seed)
    positives = [item for item in records if item.label == 1]
    negatives = [item for item in records if item.label == 0]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    pos_val = max(1, int(len(positives) * val_ratio))
    neg_val = max(1, int(len(negatives) * val_ratio))

    val_set = positives[:pos_val] + negatives[:neg_val]
    test_set = positives[pos_val:] + negatives[neg_val:]

    rng.shuffle(val_set)
    rng.shuffle(test_set)
    return val_set, test_set


def build_cache_corpus(
    all_records: Sequence[LabeledQuery], max_corpus: int, seed: int
) -> List[str]:
    positives = [item.text for item in all_records if item.label == 1]
    unique = list(dict.fromkeys(positives))
    rng = random.Random(seed)
    if len(unique) > max_corpus:
        unique = rng.sample(unique, max_corpus)
    return unique


def qdrant_sparse_vector(sparse_embedding_obj) -> models.SparseVector:
    return models.SparseVector(**sparse_embedding_obj.as_object())


def minmax_normalize(score_map: Dict[int, float]) -> Dict[int, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    min_score = min(values)
    max_score = max(values)
    if math.isclose(min_score, max_score):
        return {doc_id: 0.0 for doc_id in score_map}
    denom = max_score - min_score
    return {doc_id: (score - min_score) / denom for doc_id, score in score_map.items()}


def compute_metrics(signals: Sequence[QuerySignals], threshold: float) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    positive_total = 0
    positive_exact_match_hits = 0
    predicted_hits = 0

    for row in signals:
        predicted_hit = row.top_fused_score >= threshold
        actual_positive = row.label == 1

        if predicted_hit:
            predicted_hits += 1

        if actual_positive:
            positive_total += 1
            if predicted_hit and row.top_doc_text == row.query_text:
                positive_exact_match_hits += 1

        if predicted_hit and actual_positive:
            tp += 1
        elif predicted_hit and not actual_positive:
            fp += 1
        elif (not predicted_hit) and (not actual_positive):
            tn += 1
        else:
            fn += 1

    total = max(1, len(signals))
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    cache_hit_rate = predicted_hits / total
    false_hit_rate = fp / (fp + tn) if (fp + tn) else 0.0
    exact_match_hit_rate_positive = (
        positive_exact_match_hits / positive_total if positive_total else 0.0
    )
    latencies = [row.latency_ms for row in signals]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cache_hit_rate": cache_hit_rate,
        "false_hit_rate": false_hit_rate,
        "positive_exact_match_hit_rate": exact_match_hit_rate_positive,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def best_threshold_by_f1(signals: Sequence[QuerySignals]) -> Tuple[float, float]:
    if not signals:
        return 0.5, 0.0
    candidates = sorted({row.top_fused_score for row in signals})
    if len(candidates) == 1:
        threshold = candidates[0]
        return threshold, compute_metrics(signals, threshold)["f1"]

    best_threshold = candidates[0]
    best_f1 = -1.0
    for threshold in candidates:
        f1 = compute_metrics(signals, threshold)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1


def lookup_doc_text(
    data_manager,
    doc_id: int,
    score: float,
    doc_text_cache: Dict[int, str],
) -> str:
    if doc_id in doc_text_cache:
        return doc_text_cache[doc_id]

    cache_data = data_manager.get_scalar_data((score, doc_id))
    text = _question_to_text(cache_data.question) if cache_data is not None else ""
    doc_text_cache[doc_id] = text
    return text


def dense_and_sparse_candidates(
    client: QdrantClient,
    collection_name: str,
    query_dense: np.ndarray,
    query_sparse,
    top_k: int,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, str]]:
    dense_results = client.query_points(
        collection_name=collection_name,
        query=query_dense.tolist(),
        using="embedding",
        limit=top_k,
        with_payload=True,
    ).points

    sparse_results = client.query_points(
        collection_name=collection_name,
        query=qdrant_sparse_vector(query_sparse),
        using="bm25",
        limit=top_k,
        with_payload=True,
    ).points

    dense_scores: Dict[int, float] = {}
    sparse_scores: Dict[int, float] = {}
    text_lookup: Dict[int, str] = {}

    for point in dense_results:
        point_id = int(point.id)
        dense_scores[point_id] = float(point.score)
        payload_text = (point.payload or {}).get("text", "")
        if payload_text:
            text_lookup[point_id] = str(payload_text)

    for point in sparse_results:
        point_id = int(point.id)
        sparse_scores[point_id] = float(point.score)
        payload_text = (point.payload or {}).get("text", "")
        if payload_text:
            text_lookup[point_id] = str(payload_text)

    return dense_scores, sparse_scores, text_lookup


def fuse_scores(
    alpha: float,
    dense_scores: Dict[int, float],
    sparse_scores: Dict[int, float],
) -> Dict[int, float]:
    dense_norm = minmax_normalize(dense_scores)
    sparse_norm = minmax_normalize(sparse_scores)
    doc_ids = set(dense_norm.keys()) | set(sparse_norm.keys())

    fused: Dict[int, float] = {}
    for doc_id in doc_ids:
        d = dense_norm.get(doc_id, 0.0)
        s = sparse_norm.get(doc_id, 0.0)
        fused[doc_id] = alpha * d + (1.0 - alpha) * s
    return fused


def evaluate_weight(
    alpha: float,
    queries: Sequence[LabeledQuery],
    query_embeddings: Dict[str, np.ndarray],
    query_sparse_embeddings,
    data_manager,
    client: QdrantClient,
    collection_name: str,
    top_k: int,
    doc_text_cache: Dict[int, str],
) -> List[QuerySignals]:
    signals: List[QuerySignals] = []

    for item in queries:
        start = time.perf_counter()

        dense_scores, sparse_scores, text_lookup = dense_and_sparse_candidates(
            client=client,
            collection_name=collection_name,
            query_dense=query_embeddings[item.text],
            query_sparse=query_sparse_embeddings[item.text],
            top_k=top_k,
        )
        fused = fuse_scores(alpha=alpha, dense_scores=dense_scores, sparse_scores=sparse_scores)

        if fused:
            top_doc_id, top_score = max(fused.items(), key=lambda pair: pair[1])
            top_doc_text = text_lookup.get(top_doc_id, "")
            if not top_doc_text:
                top_doc_text = lookup_doc_text(
                    data_manager=data_manager,
                    doc_id=int(top_doc_id),
                    score=float(top_score),
                    doc_text_cache=doc_text_cache,
                )
        else:
            top_score = 0.0
            top_doc_text = ""

        latency_ms = (time.perf_counter() - start) * 1000.0

        signals.append(
            QuerySignals(
                label=item.label,
                query_text=item.text,
                top_fused_score=float(top_score),
                top_doc_text=top_doc_text,
                latency_ms=latency_ms,
            )
        )

    return signals


def evaluate_rrf(
    queries: Sequence[LabeledQuery],
    query_embeddings: Dict[str, np.ndarray],
    query_sparse_embeddings,
    data_manager,
    top_k: int,
    doc_text_cache: Dict[int, str],
) -> List[QuerySignals]:
    signals: List[QuerySignals] = []

    for item in queries:
        start = time.perf_counter()
        results = data_manager.search(
            embedding_data=query_embeddings[item.text],
            bm25=[query_sparse_embeddings[item.text]],
            top_k=top_k,
        )

        if results:
            top_score, top_doc_id = results[0]
            top_doc_text = lookup_doc_text(
                data_manager=data_manager,
                doc_id=int(top_doc_id),
                score=float(top_score),
                doc_text_cache=doc_text_cache,
            )
        else:
            top_score = 0.0
            top_doc_text = ""

        latency_ms = (time.perf_counter() - start) * 1000.0
        signals.append(
            QuerySignals(
                label=item.label,
                query_text=item.text,
                top_fused_score=float(top_score),
                top_doc_text=top_doc_text,
                latency_ms=latency_ms,
            )
        )

    return signals


def select_conservative_row(rows: Sequence[Dict[str, float]], rrf_row: Dict[str, float]) -> Dict[str, float]:
    constrained = [
        r
        for r in rows
        if r["false_hit_rate"] <= rrf_row["false_hit_rate"]
        and r["precision"] >= rrf_row["precision"]
    ]
    if constrained:
        return sorted(
            constrained,
            key=lambda r: (r["recall"], r["f1"], r["accuracy"]),
            reverse=True,
        )[0]

    # Fallback: most conservative on false hits, then strongest precision and F1.
    return sorted(
        rows,
        key=lambda r: (r["false_hit_rate"], -r["precision"], -r["f1"]),
    )[0]


def sweep_weights(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading labeled dataset: {dataset_path}")
    all_records_raw = load_labeled_queries(dataset_path)

    if args.english_only:
        all_records = filter_english_records(all_records_raw)
        print(
            f"English-only filter: kept {len(all_records)} / {len(all_records_raw)} "
            f"({(len(all_records) / max(1, len(all_records_raw))):.1%})"
        )
    else:
        all_records = all_records_raw

    if not all_records:
        raise ValueError("No records available after filtering")

    sampled_queries = sample_balanced_queries(all_records, args.max_queries, args.seed)
    val_queries, test_queries = split_stratified(sampled_queries, args.val_ratio, args.seed)

    corpus_texts = build_cache_corpus(all_records, args.max_corpus, args.seed)

    print(f"Sampled queries: {len(sampled_queries)}")
    print(f"  Validation: {len(val_queries)}")
    print(f"  Test:       {len(test_queries)}")
    print(f"Cache corpus size (label=1): {len(corpus_texts)}")

    print("Initializing embedding models...")
    dense_encoder = Onnx()
    sparse_encoder = SparseTextEmbedding("Qdrant/bm25")

    query_texts = list({item.text for item in sampled_queries})

    print("Encoding dense query embeddings...")
    query_embeddings: Dict[str, np.ndarray] = {
        text: np.asarray(dense_encoder.to_embeddings(text), dtype=np.float32)
        for text in query_texts
    }

    print("Encoding sparse query embeddings...")
    query_sparse_embeddings = {
        text: list(sparse_encoder.query_embed(text))[0] for text in query_texts
    }

    print("Encoding corpus embeddings...")
    corpus_dense = [
        np.asarray(dense_encoder.to_embeddings(text), dtype=np.float32).tolist()
        for text in corpus_texts
    ]
    corpus_sparse = [list(sparse_encoder.query_embed(text))[0] for text in corpus_texts]

    print("Building in-memory GPTCache index (Qdrant backend)...")
    qdrant = VectorBase(
        "qdrant",
        top_k=args.top_k,
        dimension=dense_encoder.dimension,
        location=":memory:",
        collection_name="adaptive_weighting_cache",
        hybrid=True,
    )
    data_manager = get_data_manager(CacheBase("sqlite"), qdrant)

    data_manager.import_data(
        questions=corpus_texts,
        answers=[text for text in corpus_texts],
        embedding_datas=[np.asarray(v, dtype=np.float32) for v in corpus_dense],
        bm25_datas=[[sparse_vec] for sparse_vec in corpus_sparse],
        session_ids=[None] * len(corpus_texts),
    )

    client: QdrantClient = data_manager.v._client
    collection_name = data_manager.v._collection_name
    doc_text_cache: Dict[int, str] = {}

    step = args.weight_step
    if step <= 0 or step > 1:
        raise ValueError("weight-step must be in (0, 1]")

    alphas: List[float] = []
    current = 0.0
    while current < 1.0 + 1e-9:
        alphas.append(round(current, 4))
        current += step
    if alphas[-1] != 1.0:
        alphas.append(1.0)

    rows: List[Dict[str, float]] = []

    print("Running weight sweep...")
    for alpha in alphas:
        val_signals = evaluate_weight(
            alpha=alpha,
            queries=val_queries,
            query_embeddings=query_embeddings,
            query_sparse_embeddings=query_sparse_embeddings,
            data_manager=data_manager,
            client=client,
            collection_name=collection_name,
            top_k=args.top_k,
            doc_text_cache=doc_text_cache,
        )
        threshold, val_f1 = best_threshold_by_f1(val_signals)

        test_signals = evaluate_weight(
            alpha=alpha,
            queries=test_queries,
            query_embeddings=query_embeddings,
            query_sparse_embeddings=query_sparse_embeddings,
            data_manager=data_manager,
            client=client,
            collection_name=collection_name,
            top_k=args.top_k,
            doc_text_cache=doc_text_cache,
        )
        test_metrics = compute_metrics(test_signals, threshold)

        row: Dict[str, float] = {
            "alpha_cosine": float(alpha),
            "beta_bm25": float(1.0 - alpha),
            "threshold": float(threshold),
            "val_f1": float(val_f1),
            **test_metrics,
        }
        rows.append(row)

        print(
            f"alpha={alpha:.2f} | thr={threshold:.4f} | "
            f"acc={row['accuracy']:.4f} | f1={row['f1']:.4f} | "
            f"hit_rate={row['cache_hit_rate']:.4f} | false_hit={row['false_hit_rate']:.4f}"
        )

    print("Running baseline: Qdrant RRF fusion via GPTCache search...")
    rrf_val_signals = evaluate_rrf(
        queries=val_queries,
        query_embeddings=query_embeddings,
        query_sparse_embeddings=query_sparse_embeddings,
        data_manager=data_manager,
        top_k=args.top_k,
        doc_text_cache=doc_text_cache,
    )
    rrf_threshold, rrf_val_f1 = best_threshold_by_f1(rrf_val_signals)
    rrf_test_signals = evaluate_rrf(
        queries=test_queries,
        query_embeddings=query_embeddings,
        query_sparse_embeddings=query_sparse_embeddings,
        data_manager=data_manager,
        top_k=args.top_k,
        doc_text_cache=doc_text_cache,
    )
    rrf_metrics = compute_metrics(rrf_test_signals, rrf_threshold)
    rrf_row: Dict[str, float] = {
        "alpha_cosine": -1.0,
        "beta_bm25": -1.0,
        "threshold": float(rrf_threshold),
        "val_f1": float(rrf_val_f1),
        **rrf_metrics,
    }
    print(
        f"RRF baseline | thr={rrf_threshold:.4f} | acc={rrf_row['accuracy']:.4f} | "
        f"f1={rrf_row['f1']:.4f} | hit_rate={rrf_row['cache_hit_rate']:.4f} | "
        f"false_hit={rrf_row['false_hit_rate']:.4f}"
    )

    # Pick best alpha by F1, tie-breaker by accuracy then lower false-hit-rate.
    best_row = sorted(
        rows,
        key=lambda r: (r["f1"], r["accuracy"], -r["false_hit_rate"]),
        reverse=True,
    )[0]
    conservative_row = select_conservative_row(rows=rows, rrf_row=rrf_row)

    csv_path = output_dir / "weight_sweep_metrics.csv"
    json_path = output_dir / "weight_sweep_summary.json"
    plot_path = output_dir / "weight_sweep_plot.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "backend": "gptcache_qdrant",
        "dataset": str(dataset_path),
        "english_only": bool(args.english_only),
        "records_before_filter": len(all_records_raw),
        "records_after_filter": len(all_records),
        "sampled_queries": len(sampled_queries),
        "validation_queries": len(val_queries),
        "test_queries": len(test_queries),
        "corpus_size": len(corpus_texts),
        "top_k": args.top_k,
        "weight_step": args.weight_step,
        "best": best_row,
        "conservative": conservative_row,
        "rrf_baseline": rrf_row,
        "comparison": {
            "best_minus_rrf": {
                "accuracy": best_row["accuracy"] - rrf_row["accuracy"],
                "f1": best_row["f1"] - rrf_row["f1"],
                "precision": best_row["precision"] - rrf_row["precision"],
                "recall": best_row["recall"] - rrf_row["recall"],
                "false_hit_rate": best_row["false_hit_rate"] - rrf_row["false_hit_rate"],
                "cache_hit_rate": best_row["cache_hit_rate"] - rrf_row["cache_hit_rate"],
            },
            "conservative_minus_rrf": {
                "accuracy": conservative_row["accuracy"] - rrf_row["accuracy"],
                "f1": conservative_row["f1"] - rrf_row["f1"],
                "precision": conservative_row["precision"] - rrf_row["precision"],
                "recall": conservative_row["recall"] - rrf_row["recall"],
                "false_hit_rate": conservative_row["false_hit_rate"] - rrf_row["false_hit_rate"],
                "cache_hit_rate": conservative_row["cache_hit_rate"] - rrf_row["cache_hit_rate"],
            },
        },
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot (import lazily so script still runs metrics-only if matplotlib is missing)
    try:
        import matplotlib.pyplot as plt

        xs = [r["alpha_cosine"] for r in rows]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(xs, [r["accuracy"] for r in rows], marker="o", label="Accuracy")
        axes[0].plot(xs, [r["f1"] for r in rows], marker="o", label="F1")
        axes[0].plot(xs, [r["precision"] for r in rows], marker="o", label="Precision")
        axes[0].plot(xs, [r["recall"] for r in rows], marker="o", label="Recall")
        axes[0].plot(xs, [r["cache_hit_rate"] for r in rows], marker="o", label="Cache Hit Rate")
        axes[0].axhline(rrf_row["accuracy"], color="black", linestyle=":", label="RRF Accuracy")
        axes[0].axhline(rrf_row["f1"], color="gray", linestyle=":", label="RRF F1")
        axes[0].set_ylabel("Score")
        axes[0].set_title("Hybrid Weight Sweep: Quality Metrics vs Cosine Weight (alpha)")
        axes[0].set_ylim(0.0, 1.02)
        axes[0].legend(ncol=3, fontsize=9)

        axes[1].plot(xs, [r["false_hit_rate"] for r in rows], marker="o", label="False Hit Rate")
        axes[1].plot(xs, [r["avg_latency_ms"] for r in rows], marker="o", label="Avg Latency (ms)")
        axes[1].plot(
            xs,
            [r["positive_exact_match_hit_rate"] for r in rows],
            marker="o",
            label="Positive Exact-Match Hit Rate",
        )
        axes[1].axhline(rrf_row["false_hit_rate"], color="black", linestyle=":", label="RRF False Hit Rate")
        axes[1].set_xlabel("Cosine Weight alpha (BM25 weight = 1 - alpha)")
        axes[1].set_ylabel("Metric Value")
        axes[1].set_title("Operational Metrics vs Weight")
        axes[1].legend(ncol=3, fontsize=9)

        best_alpha = best_row["alpha_cosine"]
        for ax in axes:
            ax.axvline(best_alpha, color="red", linestyle="--", alpha=0.7)
            ax.text(
                best_alpha,
                ax.get_ylim()[1] * 0.92,
                f"best α={best_alpha:.2f}",
                color="red",
                ha="left",
                va="top",
                fontsize=9,
            )

        conservative_alpha = conservative_row["alpha_cosine"]
        for ax in axes:
            ax.axvline(conservative_alpha, color="green", linestyle="--", alpha=0.7)
            ax.text(
                conservative_alpha,
                ax.get_ylim()[1] * 0.84,
                f"cons α={conservative_alpha:.2f}",
                color="green",
                ha="left",
                va="top",
                fontsize=9,
            )

        fig.tight_layout()
        fig.savefig(plot_path, dpi=220)
        print(f"Saved plot: {plot_path}")

    except ImportError:
        print("matplotlib is not installed; skipping PNG plot generation.")
        print("Install with: pip install matplotlib")

    print(f"Saved metrics CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")
    print("\nBest configuration:")
    print(
        f"  alpha(cosine)={best_row['alpha_cosine']:.2f}, "
        f"beta(bm25)={best_row['beta_bm25']:.2f}, "
        f"threshold={best_row['threshold']:.4f}"
    )
    print(
        f"  accuracy={best_row['accuracy']:.4f}, f1={best_row['f1']:.4f}, "
        f"cache_hit_rate={best_row['cache_hit_rate']:.4f}, false_hit_rate={best_row['false_hit_rate']:.4f}"
    )
    print("\nConservative configuration (adaptive):")
    print(
        f"  alpha(cosine)={conservative_row['alpha_cosine']:.2f}, "
        f"beta(bm25)={conservative_row['beta_bm25']:.2f}, "
        f"threshold={conservative_row['threshold']:.4f}"
    )
    print(
        f"  accuracy={conservative_row['accuracy']:.4f}, f1={conservative_row['f1']:.4f}, "
        f"precision={conservative_row['precision']:.4f}, recall={conservative_row['recall']:.4f}, "
        f"false_hit_rate={conservative_row['false_hit_rate']:.4f}"
    )
    print("\nRRF baseline configuration:")
    print(
        f"  threshold={rrf_row['threshold']:.4f}, accuracy={rrf_row['accuracy']:.4f}, "
        f"f1={rrf_row['f1']:.4f}, false_hit_rate={rrf_row['false_hit_rate']:.4f}"
    )


def main() -> None:
    args = parse_args()
    sweep_weights(args)


if __name__ == "__main__":
    main()
