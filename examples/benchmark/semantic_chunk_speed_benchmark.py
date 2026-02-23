import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gptcache.semantic_forest import (  # noqa: E402
    SemanticForestChunker,
    SemanticForestDataManager,
    SemanticForestEmbedder,
)


def _load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handler:
        return json.load(handler)


def _index_groups(base: List[dict], mut1: List[dict], mut2: List[dict]):
    groups: Dict[str, dict] = {}
    for session in base:
        gid = session.get("group_id") or session.get("session_id")
        if not gid:
            continue
        groups[gid] = {"base": session, "mut1": None, "mut2": None}
    for session in mut1:
        gid = session.get("group_id")
        if gid in groups:
            groups[gid]["mut1"] = session
    for session in mut2:
        gid = session.get("group_id")
        if gid in groups:
            groups[gid]["mut2"] = session
    return groups


def _select_groups(groups: Dict[str, dict], pair_count: int, triple_count: int, rng: random.Random):
    triple_candidates = [g for g in groups.values() if g["mut1"] and g["mut2"]]
    pair_candidates = [g for g in groups.values() if g["mut1"] and not g["mut2"]]
    rng.shuffle(triple_candidates)
    rng.shuffle(pair_candidates)
    return pair_candidates[:pair_count], triple_candidates[:triple_count]


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


def _make_mixed_session(session_a: dict, session_b: dict, mixed_id: str, prefix_context: bool, context_prefix_len: int):
    mixed = {
        "session_id": mixed_id,
        "dataset": session_a.get("dataset"),
        "source_id": f"mix:{session_a.get('group_id')}+{session_b.get('group_id')}",
        "context": f"{session_a.get('context','')}\n---\n{session_b.get('context','')}",
        "turns": [],
        "meta": {
            "session_type": "mixed",
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


def _assemble_dataset(
    base_sessions: List[dict],
    pairs: List[dict],
    triples: List[dict],
    mixed_per_pair: int,
    target_similar_rate: float,
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
        used_group_ids.add(group["base"].get("group_id"))
        add_session(group["base"], "base")
        add_session(group["mut1"], "mut1")
        add_session(group["mut2"], "mut2")

    for group in pairs:
        used_group_ids.add(group["base"].get("group_id"))
        add_session(group["base"], "base")
        add_session(group["mut1"], "mut1")

    mixed_sessions = []
    if mixed_per_pair > 0 and pairs:
        pair_sessions = [g["base"] for g in pairs]
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
                prefix_context=prefix_context,
                context_prefix_len=context_prefix_len,
            )
            mixed_sessions.append(mixed)
    sessions.extend(mixed_sessions)

    similar_sessions = len([s for s in sessions if s.get("meta", {}).get("session_type") != "single"])
    needed_total = int(round(similar_sessions / target_similar_rate)) if target_similar_rate > 0 else len(sessions)
    needed_singles = max(0, needed_total - len(sessions))

    single_candidates = [s for s in base_sessions if s.get("group_id") not in used_group_ids]
    rng.shuffle(single_candidates)
    for session in single_candidates[:needed_singles]:
        add_session(session, "single")

    if order == "shuffle":
        rng.shuffle(sessions)
    else:
        sessions.sort(key=lambda s: s.get("meta", {}).get("session_type"))

    return sessions


def _record_sample(samples, metrics, nodes, turns):
    window = metrics[-1]
    samples.append(
        {
            "turns": turns,
            "nodes": nodes,
            **window,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Cache growth speed benchmark")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", choices=("coqa", "squad_v2"), default="coqa")
    parser.add_argument("--pair-count", type=int, default=200)
    parser.add_argument("--triple-count", type=int, default=0)
    parser.add_argument("--mixed-per-pair", type=int, default=1)
    parser.add_argument("--target-similar-rate", type=float, default=0.5)
    parser.add_argument("--prefix-context", action="store_true")
    parser.add_argument("--context-prefix-len", type=int, default=64)
    parser.add_argument("--order", choices=("primed", "shuffle"), default="primed")
    parser.add_argument("--seed", type=int, default=13)

    parser.add_argument("--drift-threshold", type=float, default=0.35)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--max-chunk-size", type=int, default=4)
    parser.add_argument("--node-match-threshold", type=float, default=0.88)
    parser.add_argument("--vector-dim", type=int, default=64)
    parser.add_argument("--chunking", action="store_true")

    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", default=None)

    args = parser.parse_args()
    rng = random.Random(args.seed)

    base = _load_json(f"{args.data_dir}/{args.dataset}_base.json")
    mut1 = _load_json(f"{args.data_dir}/{args.dataset}_mut1.json")
    mut2 = _load_json(f"{args.data_dir}/{args.dataset}_mut2.json")
    groups = _index_groups(base, mut1, mut2)
    pairs, triples = _select_groups(groups, args.pair_count, args.triple_count, rng)
    sessions = _assemble_dataset(
        base,
        pairs,
        triples,
        mixed_per_pair=args.mixed_per_pair,
        target_similar_rate=args.target_similar_rate,
        prefix_context=args.prefix_context,
        context_prefix_len=args.context_prefix_len,
        order=args.order,
        rng=rng,
    )

    if args.chunking:
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

    metrics_window = []
    samples = []
    total_turns = 0

    for _ in range(args.repeat):
        for session in sessions:
            messages = []
            for turn in session.get("turns", []):
                if args.max_turns and total_turns >= args.max_turns:
                    break
                messages.append({"role": "user", "content": turn.get("question", "")})

                t0 = time.perf_counter()
                chunk_text, chunk = chunker({"messages": messages})
                t1 = time.perf_counter()
                embedding = embedder(chunk)
                t2 = time.perf_counter()
                _ = manager.search(embedding)
                t3 = time.perf_counter()
                manager.save(chunk_text, "cached", embedding)
                t4 = time.perf_counter()

                metrics_window.append(
                    {
                        "chunk_ms": (t1 - t0) * 1000.0,
                        "embed_ms": (t2 - t1) * 1000.0,
                        "search_ms": (t3 - t2) * 1000.0,
                        "save_ms": (t4 - t3) * 1000.0,
                        "total_ms": (t4 - t0) * 1000.0,
                    }
                )
                if len(metrics_window) > args.window:
                    metrics_window.pop(0)

                total_turns += 1
                if total_turns % args.sample_every == 0:
                    window_avg = {
                        key: sum(m[key] for m in metrics_window) / len(metrics_window)
                        for key in metrics_window[-1].keys()
                    }
                    _record_sample(samples, [window_avg], len(manager.nodes), total_turns)

            if args.max_turns and total_turns >= args.max_turns:
                break
        if args.max_turns and total_turns >= args.max_turns:
            break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handler:
        for sample in samples:
            handler.write(json.dumps(sample) + "\n")

    if args.csv_output:
        Path(args.csv_output).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["turns", "nodes", "chunk_ms", "embed_ms", "search_ms", "save_ms", "total_ms"]
        with open(args.csv_output, "w", encoding="utf-8", newline="") as handler:
            writer = csv.DictWriter(handler, fieldnames=fieldnames)
            writer.writeheader()
            for sample in samples:
                writer.writerow({key: sample.get(key) for key in fieldnames})

    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
