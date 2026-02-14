import argparse
import itertools
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_list(values: Optional[str], caster):
    if not values:
        return None
    return [caster(item) for item in values.split(",") if item.strip()]


def _write_report(path: str, lines: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handler:
        handler.write("\n".join(lines) + "\n")


def _append_report(path: str, lines: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handler:
        handler.write("\n".join(lines) + "\n")


def _parse_variant_specs(specs: Optional[str]):
    if not specs:
        return None
    variants = []
    for spec in specs.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        parts = {}
        for item in spec.split(","):
            if not item.strip():
                continue
            key, _, value = item.partition("=")
            key = key.strip()
            value = value.strip()
            parts[key] = value
        try:
            variant = {
                "target_rate": float(parts.get("rate", "0.5")),
                "order": parts.get("order", "primed"),
                "prefix_context": _parse_bool(parts.get("prefix", "false")),
                "mixed_per_pair": int(parts.get("mixed", "0")),
            }
        except Exception as exc:
            raise ValueError(f"Invalid variant spec: {spec}") from exc
        variants.append(variant)
    return variants


def _variant_key(dataset: str, target_rate: float, order: str, prefix_context: bool, mixed_per_pair: int):
    return (dataset, f"{target_rate:.3f}", order, bool(prefix_context), int(mixed_per_pair))


def _parse_completed_variants(report_path: str):
    if not os.path.exists(report_path):
        return set()
    completed = set()
    dataset = None
    with open(report_path, "r", encoding="utf-8") as handler:
        for line in handler:
            line = line.strip()
            if line.startswith("Dataset:"):
                dataset = line.split(":", 1)[1].strip()
                continue
            if line.startswith("Variant:") and dataset:
                parts = line.split(":", 1)[1].strip().split(",")
                parsed = {}
                for item in parts:
                    key, _, value = item.strip().partition("=")
                    parsed[key.strip()] = value.strip()
                try:
                    target_rate = float(parsed.get("target_rate", "0.0"))
                    order = parsed.get("order", "primed")
                    prefix_context = _parse_bool(parsed.get("prefix_context", "false"))
                    mixed_per_pair = int(parsed.get("mixed_per_pair", "0"))
                except Exception:
                    continue
                completed.add(_variant_key(dataset, target_rate, order, prefix_context, mixed_per_pair))
    return completed

from gptcache.semantic_forest import (
    SemanticForestChunker,
    SemanticForestDataManager,
    SemanticForestEmbedder,
)


@dataclass
class SessionGroup:
    base: dict
    mut1: Optional[dict] = None
    mut2: Optional[dict] = None


def _load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handler:
        return json.load(handler)


def _index_groups(base: List[dict], mut1: List[dict], mut2: List[dict]) -> Dict[str, SessionGroup]:
    groups: Dict[str, SessionGroup] = {}
    for session in base:
        gid = session.get("group_id") or session.get("session_id")
        if not gid:
            continue
        groups[gid] = SessionGroup(base=session)
    for session in mut1:
        gid = session.get("group_id")
        if gid in groups:
            groups[gid].mut1 = session
    for session in mut2:
        gid = session.get("group_id")
        if gid in groups:
            groups[gid].mut2 = session
    return groups


def _prefix_context(session: dict, max_len: int) -> dict:
    context = session.get("context") or ""
    context_id = session.get("source_id") or ""
    context_tag = context_id if context_id else context
    context_tag = context_tag.replace("\n", " ").strip()
    if max_len > 0:
        context_tag = context_tag[:max_len]
    if not context_tag:
        return session
    prefix = f"[CTX:{context_tag}] "
    for turn in session.get("turns", []):
        question = turn.get("question", "")
        turn["question"] = prefix + question
    return session


def _clone_session(session: dict) -> dict:
    return json.loads(json.dumps(session))


def _make_mixed_session(
    session_a: dict,
    session_b: dict,
    mixed_id: str,
    label: str,
    prefix_context: bool,
    context_prefix_len: int,
) -> dict:
    mixed = {
        "session_id": mixed_id,
        "dataset": session_a.get("dataset"),
        "source_id": f"mix:{session_a.get('group_id')}+{session_b.get('group_id')}",
        "context": f"{session_a.get('context','')}\n---\n{session_b.get('context','')}",
        "turns": [],
        "meta": {
            "session_type": label,
            "mix_of": [session_a.get("group_id"), session_b.get("group_id")],
        },
    }
    turns = []
    for turn in session_a.get("turns", []):
        turns.append({"question": turn.get("question", "")})
    for turn in session_b.get("turns", []):
        turns.append({"question": turn.get("question", "")})
    for idx, turn in enumerate(turns):
        turn["turn_index"] = idx
    mixed["turns"] = turns
    if prefix_context:
        _prefix_context(mixed, context_prefix_len)
    return mixed


def _select_groups(groups: Dict[str, SessionGroup], pair_count: int, triple_count: int, rng: random.Random):
    triple_candidates = [g for g in groups.values() if g.mut1 and g.mut2]
    pair_candidates = [g for g in groups.values() if g.mut1 and not g.mut2]

    rng.shuffle(triple_candidates)
    rng.shuffle(pair_candidates)

    triples = triple_candidates[:triple_count]
    pairs = pair_candidates[:pair_count]
    return pairs, triples


def _assemble_dataset(
    base_sessions: List[dict],
    pairs: List[SessionGroup],
    triples: List[SessionGroup],
    mixed_per_pair: int,
    target_similar_rate: Optional[float],
    prefix_context: bool,
    context_prefix_len: int,
    order: str,
    rng: random.Random,
):
    sessions = []
    used_group_ids = set()

    def add_session(session: dict, session_type: str):
        session = _clone_session(session)
        session.setdefault("meta", {})["session_type"] = session_type
        if prefix_context:
            _prefix_context(session, context_prefix_len)
        sessions.append(session)

    for group in triples:
        used_group_ids.add(group.base.get("group_id"))
        add_session(group.base, "base")
        add_session(group.mut1, "mut1")
        add_session(group.mut2, "mut2")

    for group in pairs:
        used_group_ids.add(group.base.get("group_id"))
        add_session(group.base, "base")
        add_session(group.mut1, "mut1")

    mixed_sessions = []
    if mixed_per_pair > 0 and pairs:
        pair_sessions = [g.base for g in pairs]
        rng.shuffle(pair_sessions)
        for idx in range(0, len(pair_sessions), 2):
            if idx + 1 >= len(pair_sessions):
                break
            if len(mixed_sessions) >= mixed_per_pair * len(pairs):
                break
            session_a = pair_sessions[idx]
            session_b = pair_sessions[idx + 1]
            mixed = _make_mixed_session(
                session_a,
                session_b,
                mixed_id=f"mix-{idx//2}",
                label="mixed",
                prefix_context=prefix_context,
                context_prefix_len=context_prefix_len,
            )
            mixed_sessions.append(mixed)
    sessions.extend(mixed_sessions)

    similar_sessions = len([s for s in sessions if s.get("meta", {}).get("session_type") != "single"])
    if target_similar_rate:
        needed_total = int(round(similar_sessions / target_similar_rate))
        needed_singles = max(0, needed_total - len(sessions))
    else:
        needed_singles = 0

    single_candidates = [s for s in base_sessions if s.get("group_id") not in used_group_ids]
    rng.shuffle(single_candidates)
    for session in single_candidates[:needed_singles]:
        add_session(session, "single")

    if order == "shuffle":
        rng.shuffle(sessions)
    else:
        # primed order: base/mut first, then mixed, then singles
        sessions.sort(key=lambda s: s.get("meta", {}).get("session_type"))

    return sessions


def _simulate_cache(
    sessions: List[dict],
    chunker: SemanticForestChunker,
    embedder: SemanticForestEmbedder,
    manager: SemanticForestDataManager,
    log_every_turns: int = 0,
    log_every_sessions: int = 0,
    label: str = "",
):
    total_turns = 0
    hit_turns = 0
    by_type = {}
    for session_idx, session in enumerate(sessions, start=1):
        session_type = session.get("meta", {}).get("session_type", "unknown")
        stats = by_type.setdefault(session_type, {"turns": 0, "hits": 0})
        messages = []
        for turn in session.get("turns", []):
            messages.append({"role": "user", "content": turn.get("question", "")})
            chunk_text, chunk = chunker({"messages": messages})
            embedding = embedder(chunk)
            res = manager.search(embedding)
            hit = bool(res)
            if hit:
                hit_turns += 1
                stats["hits"] += 1
            manager.save(chunk_text, "cached", embedding)
            total_turns += 1
            stats["turns"] += 1
            if log_every_turns and total_turns % log_every_turns == 0:
                print(f"{label} progress: {total_turns} turns processed")
        if log_every_sessions and session_idx % log_every_sessions == 0:
            print(f"{label} progress: {session_idx} sessions processed")
    return {
        "turns": total_turns,
        "hits": hit_turns,
        "hit_rate": (hit_turns / total_turns) if total_turns else 0.0,
        "by_type": {
            key: {
                **value,
                "hit_rate": (value["hits"] / value["turns"]) if value["turns"] else 0.0,
            }
            for key, value in by_type.items()
        },
    }


def _run_eval(sessions, chunking_enabled, args):
    start = time.time()
    if chunking_enabled:
        chunker = SemanticForestChunker(
            max_chunk_size=args.max_chunk_size,
            drift_threshold=args.drift_threshold,
            window_size=args.window_size,
        )
    else:
        chunker = SemanticForestChunker(
            max_chunk_size=1000,
            drift_threshold=0.0,
            window_size=args.window_size,
        )
    embedder = SemanticForestEmbedder(vector_dim=args.vector_dim)
    manager = SemanticForestDataManager(similarity_threshold=args.node_match_threshold, max_nodes=100000)
    label = "chunking" if chunking_enabled else "no_chunk"
    stats = _simulate_cache(
        sessions,
        chunker,
        embedder,
        manager,
        log_every_turns=args.log_every_turns,
        log_every_sessions=args.log_every_sessions,
        label=label,
    )
    elapsed = time.time() - start
    stats["elapsed_seconds"] = elapsed
    return stats


def _print_report(label: str, stats: dict):
    print(f"{label} total hit rate: {stats['hit_rate']:.3f} ({stats['hits']}/{stats['turns']})")
    for session_type, data in sorted(stats["by_type"].items()):
        print(f"  {session_type:>7}: {data['hit_rate']:.3f} ({data['hits']}/{data['turns']})")


def main():
    parser = argparse.ArgumentParser(description="Semantic chunking benchmark pipeline")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", choices=("coqa", "squad_v2", "both"), default="both")
    parser.add_argument("--pair-count", type=int, default=200)
    parser.add_argument("--triple-count", type=int, default=0)
    parser.add_argument("--mixed-per-pair", type=int, default=1)
    parser.add_argument("--target-similar-rate", type=float, default=0.5)
    parser.add_argument("--prefix-context", action="store_true")
    parser.add_argument("--context-prefix-len", type=int, default=64)
    parser.add_argument("--order", choices=("primed", "shuffle"), default="primed")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--variant-preset", choices=("none", "core"), default="none")
    parser.add_argument("--variant-specs", default=None)
    parser.add_argument("--log-every-turns", type=int, default=0)
    parser.add_argument("--log-every-sessions", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--target-similar-rates", default=None)
    parser.add_argument("--orders", default=None)
    parser.add_argument("--prefix-context-variants", default=None)
    parser.add_argument("--mixed-per-pair-variants", default=None)

    parser.add_argument("--drift-threshold", type=float, default=0.35)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--max-chunk-size", type=int, default=4)
    parser.add_argument("--node-match-threshold", type=float, default=0.88)
    parser.add_argument("--vector-dim", type=int, default=64)

    args = parser.parse_args()
    rng = random.Random(args.seed)

    datasets = [args.dataset] if args.dataset != "both" else ["coqa", "squad_v2"]
    variant_specs = _parse_variant_specs(args.variant_specs)
    if variant_specs:
        target_rates = sorted({v["target_rate"] for v in variant_specs})
        orders = sorted({v["order"] for v in variant_specs})
        prefix_variants = sorted({v["prefix_context"] for v in variant_specs})
        mixed_variants = sorted({v["mixed_per_pair"] for v in variant_specs})
    elif args.variant_preset == "core":
        target_rates = [0.5, 0.3, 0.7]
        orders = ["primed", "shuffle"]
        prefix_variants = [False, True]
        mixed_variants = [0, 1]
    else:
        target_rates = _parse_list(args.target_similar_rates, float) or [args.target_similar_rate]
        orders = _parse_list(args.orders, str) or [args.order]
        prefix_variants = _parse_list(args.prefix_context_variants, _parse_bool) or [args.prefix_context]
        mixed_variants = _parse_list(args.mixed_per_pair_variants, int) or [args.mixed_per_pair]

    report_lines = []
    completed_variants = set()
    if args.report_path:
        if args.resume and os.path.exists(args.report_path):
            completed_variants = _parse_completed_variants(args.report_path)
            _append_report(
                args.report_path,
                [f"Resume run: {time.strftime('%Y-%m-%d %H:%M:%S')}"],
            )
        else:
            report_lines.extend(
                [
                    "Semantic chunking benchmark report",
                    f"Data dir: {args.data_dir}",
                    f"Datasets: {', '.join(datasets)}",
                    f"Pair count: {args.pair_count}",
                    f"Triple count: {args.triple_count}",
                    f"Preset: {args.variant_preset}",
                    "Chunker config:",
                    f"  drift_threshold={args.drift_threshold}",
                    f"  window_size={args.window_size}",
                    f"  max_chunk_size={args.max_chunk_size}",
                    f"  node_match_threshold={args.node_match_threshold}",
                    f"  vector_dim={args.vector_dim}",
                    "Variant grid:",
                    f"  target_rates={target_rates}",
                    f"  orders={orders}",
                    f"  prefix_context={prefix_variants}",
                    f"  mixed_per_pair={mixed_variants}",
                    f"Variant specs: {args.variant_specs or 'none'}",
                    "",
                ]
            )
            _write_report(args.report_path, report_lines)
        report_lines = []

    if variant_specs:
        variant_total = len(datasets) * len(variant_specs)
    else:
        variant_total = len(datasets) * len(target_rates) * len(orders) * len(prefix_variants) * len(mixed_variants)
    variant_index = 0
    for dataset_name in datasets:
        base = _load_json(f"{args.data_dir}/{dataset_name}_base.json")
        mut1 = _load_json(f"{args.data_dir}/{dataset_name}_mut1.json")
        mut2 = _load_json(f"{args.data_dir}/{dataset_name}_mut2.json")
        groups = _index_groups(base, mut1, mut2)
        pairs, triples = _select_groups(groups, args.pair_count, args.triple_count, rng)

        if variant_specs:
            variant_iter = variant_specs
        else:
            variant_iter = [
                {
                    "target_rate": target_rate,
                    "order": order,
                    "prefix_context": prefix_context,
                    "mixed_per_pair": mixed_per_pair,
                }
                for target_rate, order, prefix_context, mixed_per_pair in itertools.product(
                    target_rates, orders, prefix_variants, mixed_variants
                )
            ]

        for variant in variant_iter:
            target_rate = variant["target_rate"]
            order = variant["order"]
            prefix_context = variant["prefix_context"]
            mixed_per_pair = variant["mixed_per_pair"]
            if completed_variants and args.report_path:
                key = _variant_key(dataset_name, target_rate, order, prefix_context, mixed_per_pair)
                if key in completed_variants:
                    print(
                        f"Skipping completed variant for {dataset_name}: rate={target_rate}, order={order}, "
                        f"prefix_context={prefix_context}, mixed_per_pair={mixed_per_pair}"
                    )
                    continue
            variant_index += 1
            sessions = _assemble_dataset(
                base,
                pairs,
                triples,
                mixed_per_pair=mixed_per_pair,
                target_similar_rate=target_rate,
                prefix_context=prefix_context,
                context_prefix_len=args.context_prefix_len,
                order=order,
                rng=rng,
            )

            print(f"\nDataset: {dataset_name}")
            print(f"Sessions: {len(sessions)}")
            print(
                f"Variant {variant_index}/{variant_total}: target_rate={target_rate}, order={order}, "
                f"prefix_context={prefix_context}, mixed_per_pair={mixed_per_pair}"
            )
            stats_no_chunk = _run_eval(sessions, chunking_enabled=False, args=args)
            stats_chunk = _run_eval(sessions, chunking_enabled=True, args=args)
            _print_report("No chunking", stats_no_chunk)
            _print_report("Chunking", stats_chunk)

            if args.report_path:
                report_lines = [
                    f"Dataset: {dataset_name}",
                    f"Sessions: {len(sessions)}",
                    f"Variant: target_rate={target_rate}, order={order}, "
                    f"prefix_context={prefix_context}, mixed_per_pair={mixed_per_pair}",
                    f"No chunking total hit rate: {stats_no_chunk['hit_rate']:.3f} ({stats_no_chunk['hits']}/{stats_no_chunk['turns']})",
                    f"Chunking total hit rate: {stats_chunk['hit_rate']:.3f} ({stats_chunk['hits']}/{stats_chunk['turns']})",
                ]
                for label, stats in (("no_chunk", stats_no_chunk), ("chunk", stats_chunk)):
                    for session_type, data in sorted(stats["by_type"].items()):
                        report_lines.append(
                            f"  {label}:{session_type} {data['hit_rate']:.3f} ({data['hits']}/{data['turns']})"
                        )
                report_lines.append("")
                _append_report(args.report_path, report_lines)


if __name__ == "__main__":
    main()
