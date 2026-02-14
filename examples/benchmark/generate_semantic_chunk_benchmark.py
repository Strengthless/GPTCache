import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional


def _load_dataset(name, split):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: install the `datasets` package to load benchmark data."
        ) from exc
    return load_dataset(name, split=split)


def _coqa_turns(example):
    questions = example.get("questions") or []
    answers = example.get("answers") or {}

    if isinstance(answers, dict):
        texts = answers.get("input_text") or answers.get("text") or []
        starts = answers.get("span_start") or answers.get("answer_start") or []
        ends = answers.get("span_end") or []
    else:
        texts, starts, ends = [], [], []
        for ans in answers:
            texts.append(ans.get("input_text") or ans.get("text") or ans.get("answer") or "")
            starts.append(ans.get("span_start") or ans.get("answer_start"))
            ends.append(ans.get("span_end"))

    turns = []
    for idx, question in enumerate(questions):
        answer = texts[idx] if idx < len(texts) else ""
        start = starts[idx] if idx < len(starts) else None
        end = ends[idx] if idx < len(ends) else None
        turns.append(
            {
                "turn_index": idx,
                "question": question,
                "answer": answer,
                "answer_start": start,
                "answer_end": end,
            }
        )
    return turns


def _label_shift_by_span(turns, span_jump):
    prev_start = None
    segment_id = -1
    for idx, turn in enumerate(turns):
        start = turn.get("answer_start")
        shift = False
        if idx == 0:
            shift = True
        elif start is not None and prev_start is not None:
            shift = abs(start - prev_start) >= span_jump
        if shift:
            segment_id += 1
        turn["shift"] = shift
        turn["segment_id"] = segment_id if shift or segment_id >= 0 else None
        prev_start = start if start is not None else prev_start


def _tokenize(text):
    return [tok for tok in text.lower().split() if tok]


def _jaccard(lhs, rhs):
    if not lhs or not rhs:
        return 0.0
    inter = lhs & rhs
    union = lhs | rhs
    return (len(inter) / len(union)) if union else 0.0


def _squad_qas_by_context(dataset, include_unanswerable):
    grouped = defaultdict(list)
    for item in dataset:
        answers = item.get("answers") or {}
        texts = answers.get("text") or []
        starts = answers.get("answer_start") or []
        has_answer = len(texts) > 0 and (starts is None or len(starts) > 0)
        if not has_answer and not include_unanswerable:
            continue
        answer_text = texts[0] if texts else ""
        answer_start = starts[0] if starts else None
        question = item.get("question")
        grouped[item.get("context")].append(
            {
                "qa_id": item.get("id"),
                "question": question,
                "answer": answer_text,
                "answer_start": answer_start,
                "context": item.get("context"),
                "title": item.get("title"),
                "tokens": set(_tokenize(question)),
            }
        )
    return grouped


def _build_squad_session(
    rng,
    context_entry,
    all_contexts,
    session_id,
    min_turns,
    max_turns,
    shift_rate,
    cross_context_shift_rate,
    similarity_threshold,
    dissimilarity_threshold,
):
    context_id, context_text, qas = context_entry
    if not qas:
        return None

    num_turns = rng.randint(min_turns, max_turns)
    current_context = context_entry
    current_qa = rng.choice(qas)
    used = set()
    turns = []
    segment_id = -1

    for turn_idx in range(num_turns):
        shift = False
        context_shift = False
        if turn_idx == 0:
            shift = True
        else:
            if rng.random() < shift_rate:
                shift = True
                if cross_context_shift_rate > 0 and rng.random() < cross_context_shift_rate:
                    candidates = [
                        entry for entry in all_contexts if entry[0] != current_context[0]
                    ]
                    if candidates:
                        current_context = rng.choice(candidates)
                        context_id, context_text, qas = current_context
                        context_shift = True

        if shift:
            segment_id += 1

        if not qas:
            continue

        if turn_idx == 0:
            qa = current_qa
        else:
            anchor_tokens = current_qa.get("tokens") or set()
            available = [qa for qa in qas if qa["qa_id"] not in used]
            if not available:
                available = list(qas)
            scored = [(qa, _jaccard(anchor_tokens, qa.get("tokens") or set())) for qa in available]

            if shift:
                candidates = [qa for qa, score in scored if score <= dissimilarity_threshold]
                qa = rng.choice(candidates) if candidates else rng.choice(available)
            else:
                candidates = [qa for qa, score in scored if score >= similarity_threshold]
                qa = rng.choice(candidates) if candidates else rng.choice(available)

        used.add(qa["qa_id"])
        current_qa = qa
        turns.append(
            {
                "turn_index": turn_idx,
                "question": qa["question"],
                "answer": qa["answer"],
                "answer_start": qa.get("answer_start"),
                "answer_end": None,
                "shift": shift,
                "segment_id": segment_id,
                "context": qa["context"],
                "context_id": context_id,
                "qa_id": qa["qa_id"],
                "context_shift": context_shift,
            }
        )

    return {
        "session_id": session_id,
        "dataset": "squad_v2",
        "source_id": context_id,
        "context": context_text,
        "turns": turns,
        "meta": {
            "shift_rate": shift_rate,
            "cross_context_shift_rate": cross_context_shift_rate,
            "similarity_threshold": similarity_threshold,
            "dissimilarity_threshold": dissimilarity_threshold,
        },
    }


def _light_rewrite(text):
    replacements = {
        "what": "which",
        "which": "what",
        "who": "which person",
        "where": "in what place",
        "when": "at what time",
        "why": "for what reason",
        "how": "in what way",
    }
    tokens = text.split()
    for idx, token in enumerate(tokens):
        key = token.lower()
        if key in replacements:
            tokens[idx] = replacements[key]
            return " ".join(tokens)
    return "Please " + text if text else text


def _parse_json_list(payload):
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("[")
        end = payload.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(payload[start : end + 1])


def _llm_rewrite_session(questions, client, model, system_prompt, max_retries=3):
    if not questions:
        return questions
    user_prompt = (
        "Rewrite each question as a close paraphrase without changing meaning. "
        "Return ONLY a JSON array of strings, same length and order. "
        "No extra text. Questions: "
        + json.dumps(questions, ensure_ascii=False)
    )
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            content = response.choices[0].message.content.strip()
            items = _parse_json_list(content)
            if not isinstance(items, list):
                raise ValueError("LLM response is not a JSON list.")
            if len(items) != len(questions):
                raise ValueError("LLM response length mismatch.")
            return [str(item) for item in items]
        except Exception as exc:
            last_exc = exc
            time.sleep(2**attempt)
    raise last_exc


def _mutate_session(session, mode, rng, round_idx, llm_client=None, llm_model=None, llm_prompt=None):
    mutated = copy.deepcopy(session)
    mutated["meta"] = dict(mutated.get("meta") or {})
    mutated["meta"]["mutation_round"] = round_idx
    mutated["meta"]["mutation_mode"] = mode
    mutated["meta"]["mutation_of"] = session.get("session_id")

    if mode == "duplicate":
        return mutated

    if mode == "light_rewrite":
        for turn in mutated.get("turns", []):
            turn["question"] = _light_rewrite(turn.get("question", ""))
        return mutated

    if mode == "llm":
        if llm_client is None:
            raise RuntimeError("LLM mutation requested but no LLM client is configured.")
        questions = [turn.get("question", "") for turn in mutated.get("turns", [])]
        try:
            rewritten = _llm_rewrite_session(questions, llm_client, llm_model, llm_prompt)
            for turn, rewrite in zip(mutated.get("turns", []), rewritten):
                turn["question"] = rewrite
        except Exception as exc:
            mutated["meta"]["llm_error"] = str(exc)
        return mutated

    raise ValueError(f"Unknown mutation mode: {mode}")


def _select_coqa_sessions(dataset, count, rng, min_turns, max_turns, label_shifts, span_jump):
    candidates = []
    for example in dataset:
        turns = _coqa_turns(example)
        if not turns:
            continue
        if len(turns) < min_turns or len(turns) > max_turns:
            continue
        if label_shifts:
            _label_shift_by_span(turns, span_jump)
        session = {
            "session_id": example.get("id"),
            "dataset": "coqa",
            "source_id": example.get("id"),
            "context": example.get("story") or example.get("context") or "",
            "turns": turns,
            "meta": {
                "label_coqa_shifts": label_shifts,
                "coqa_span_jump": span_jump,
            },
        }
        candidates.append(session)

    if len(candidates) < count:
        raise ValueError(f"Not enough CoQA sessions with turns in range to select {count}.")

    rng.shuffle(candidates)
    return candidates[:count]


def _select_squad_sessions(
    dataset,
    count,
    rng,
    min_turns,
    max_turns,
    shift_rate,
    cross_context_shift_rate,
    similarity_threshold,
    dissimilarity_threshold,
):
    grouped = _squad_qas_by_context(dataset, include_unanswerable=False)
    contexts = [(qa_list[0].get("title") or f"context-{idx}", ctx, qa_list)
                for idx, (ctx, qa_list) in enumerate(grouped.items())]
    if not contexts:
        raise ValueError("No SQuAD contexts available to build sessions.")

    rng.shuffle(contexts)
    sessions = []
    attempts = 0
    max_attempts = count * 10
    while len(sessions) < count and attempts < max_attempts:
        context_entry = contexts[attempts % len(contexts)]
        session_id = f"squad-{len(sessions)}"
        session = _build_squad_session(
            rng,
            context_entry,
            contexts,
            session_id,
            min_turns,
            max_turns,
            shift_rate,
            cross_context_shift_rate,
            similarity_threshold,
            dissimilarity_threshold,
        )
        attempts += 1
        if not session or not session.get("turns"):
            continue
        sessions.append(session)

    if len(sessions) < count:
        raise ValueError(f"Failed to build {count} SQuAD sessions. Got {len(sessions)}.")
    return sessions


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handler:
        json.dump(payload, handler, indent=2)


def _prepare_llm_client(mode, model, system_prompt):
    if mode != "llm":
        return None, None, None
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("LLM mutation requires the `openai` package.") from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM mutation.")
    client = OpenAI()
    prompt = system_prompt or (
        "Rewrite the question to be a close paraphrase without changing its meaning."
    )
    return client, model, prompt


def _mutate_sessions_parallel(
    sessions,
    mutation_mode,
    rng,
    round_idx,
    llm_client,
    llm_model,
    llm_prompt,
    parallelism,
):
    if mutation_mode != "llm" or parallelism <= 1:
        return [
            _mutate_session(
                session,
                mutation_mode,
                rng,
                round_idx=round_idx,
                llm_client=llm_client,
                llm_model=llm_model,
                llm_prompt=llm_prompt,
            )
            for session in sessions
        ]
    results = [None] * len(sessions)
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_map = {
            executor.submit(
                _mutate_session,
                session,
                mutation_mode,
                rng,
                round_idx,
                llm_client,
                llm_model,
                llm_prompt,
            ): idx
            for idx, session in enumerate(sessions)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
    return results


def _build_outputs_for_dataset(
    dataset_name,
    base_sessions,
    mutate_count,
    mutate2_count,
    mutation_mode,
    rng,
    llm_client,
    llm_model,
    llm_prompt,
    llm_parallelism,
):
    for idx, session in enumerate(base_sessions):
        session["group_id"] = session.get("session_id") or f"{dataset_name}-{idx}"

    base_mut1 = base_sessions[:mutate_count]
    mut1_raw = _mutate_sessions_parallel(
        base_mut1,
        mutation_mode,
        rng,
        round_idx=1,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_prompt=llm_prompt,
        parallelism=llm_parallelism,
    )
    mut1 = []
    for session, mutated in zip(base_mut1, mut1_raw):
        mutated["group_id"] = session["group_id"]
        mutated["session_id"] = f"{session['group_id']}-m1"
        mut1.append(mutated)

    base_mut2 = mut1[:mutate2_count]
    mut2_raw = _mutate_sessions_parallel(
        base_mut2,
        mutation_mode,
        rng,
        round_idx=2,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_prompt=llm_prompt,
        parallelism=llm_parallelism,
    )
    mut2 = []
    for session, mutated in zip(base_mut2, mut2_raw):
        mutated["group_id"] = session["group_id"]
        mutated["session_id"] = f"{session['group_id']}-m2"
        mut2.append(mutated)

    return base_sessions, mut1, mut2


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark datasets for semantic chunking."
    )
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--datasets", default="coqa,squad_v2", help="Comma-separated dataset names.")
    parser.add_argument("--split", default="train", help="Dataset split name.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")

    parser.add_argument("--base-count", type=int, default=1000, help="Base conversations per dataset.")
    parser.add_argument("--mutate-count", type=int, default=300, help="First-round mutation count.")
    parser.add_argument("--mutate2-count", type=int, default=100, help="Second-round mutation count.")

    parser.add_argument("--min-turns", type=int, default=2, help="Min turns per session.")
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns per session.")

    parser.add_argument("--shift-rate", type=float, default=0.3, help="Probability of a semantic shift (SQuAD only).")
    parser.add_argument("--cross-context-shift-rate", type=float, default=0.0, help="Chance a shift jumps to another context (SQuAD only).")
    parser.add_argument("--similarity-threshold", type=float, default=0.25, help="Jaccard threshold for similar turns (SQuAD only).")
    parser.add_argument("--dissimilarity-threshold", type=float, default=0.1, help="Jaccard threshold for shift turns (SQuAD only).")

    parser.add_argument("--label-coqa-shifts", action="store_true", help="Label CoQA shifts using span jumps.")
    parser.add_argument("--coqa-span-jump", type=int, default=120, help="Char jump to mark CoQA shift.")

    parser.add_argument("--mutation-mode", choices=("duplicate", "light_rewrite", "llm"), default="duplicate")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-system-prompt", default=None)
    parser.add_argument("--llm-parallelism", type=int, default=20)

    args = parser.parse_args()
    rng = random.Random(args.seed)
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]

    if args.mutate2_count > args.mutate_count:
        raise ValueError("mutate2-count must be <= mutate-count")
    if args.mutate_count > args.base_count:
        raise ValueError("mutate-count must be <= base-count")

    llm_client, llm_model, llm_prompt = _prepare_llm_client(
        args.mutation_mode, args.llm_model, args.llm_system_prompt
    )

    for dataset_name in datasets:
        if dataset_name == "coqa":
            dataset = _load_dataset("coqa", args.split)
            base_sessions = _select_coqa_sessions(
                dataset,
                args.base_count,
                rng,
                args.min_turns,
                args.max_turns,
                args.label_coqa_shifts,
                args.coqa_span_jump,
            )
        elif dataset_name in {"squad_v2", "squad"}:
            dataset = _load_dataset(dataset_name, args.split)
            base_sessions = _select_squad_sessions(
                dataset,
                args.base_count,
                rng,
                args.min_turns,
                args.max_turns,
                args.shift_rate,
                args.cross_context_shift_rate,
                args.similarity_threshold,
                args.dissimilarity_threshold,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        base_sessions, mut1, mut2 = _build_outputs_for_dataset(
            dataset_name,
            base_sessions,
            args.mutate_count,
            args.mutate2_count,
            args.mutation_mode,
            rng,
            llm_client,
            llm_model,
            llm_prompt,
            args.llm_parallelism,
        )

        _write_json(
            os.path.join(args.output_dir, f"{dataset_name}_base.json"),
            base_sessions,
        )
        _write_json(
            os.path.join(args.output_dir, f"{dataset_name}_mut1.json"),
            mut1,
        )
        _write_json(
            os.path.join(args.output_dir, f"{dataset_name}_mut2.json"),
            mut2,
        )

        print(f"Wrote {dataset_name} base/mut1/mut2 to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
