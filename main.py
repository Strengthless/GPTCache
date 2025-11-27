import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.utils.ez_integration import init_cache_with_ollama
from langchain_ollama import OllamaLLM
from qdrant_client.models import SparseVector
from gptcache.utils.cache_func import cache_selectively
print("Cache loading.....")

onnx = Onnx()
qdrant = VectorBase(
    "qdrant",
    top_k=10,
    dimension=onnx.dimension,
    location=":memory:",
    hybrid=False # True for hybrid search mode, False for dense only
)
data_manager = get_data_manager(CacheBase("sqlite"), qdrant)
llm = OllamaLLM(
    model="hf.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M",
    validate_model_on_init=True,
    temperature=0.8,
    num_predict=256
)

llm_cache, cached_llm = init_cache_with_ollama(
    llm, 
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation()
)

llm_cache.set_openai_key()
llm_cache.config.similarity_threshold = 0.2  # Set similarity threshold for cache hits

test_dataset = [
    # ==============================================================================
    # 1. EVERGREEN FACTUAL — MUST BE CACHABLE (perfect for semantic cache)
    # ==============================================================================
    {"q": "What is the capital of France?", "a": "Paris", "cacheable": True},
    {"q": "Who wrote 'Pride and Prejudice'?", "a": "Jane Austen", "cacheable": True},
    {"q": "What year was the Python programming language first released?", "a": "1991", "cacheable": True},
    {"q": "What does HTTP stand for?", "a": "HyperText Transfer Protocol", "cacheable": True},
    {"q": "Name the seven layers of the OSI model in order.", "a": "Physical, Data Link, Network, Transport, Session, Presentation, Application", "cacheable": True},
    {"q": "What is the chemical symbol for gold?", "a": "Au", "cacheable": True},
    {"q": "How many bits are in one byte?", "a": "8", "cacheable": True},
    {"q": "What is the speed of light in vacuum in m/s?", "a": "299,792,458 m/s", "cacheable": True},
    {"q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci", "cacheable": True},
    {"q": "What is the largest planet in our solar system?", "a": "Jupiter", "cacheable": True},

    # ==============================================================================
    # 2. NEAR-EVERGREEN (very slow changing) — should be cached with medium-long TTL
    # ==============================================================================
    {"q": "What is the current version of the Python language as of 2025?", "a": "Python 3.13 (stable), Python 3.14 in development", "cacheable": True},
    {"q": "Who is the current CEO of OpenAI in November 2025?", "a": "Sam Altman", "cacheable": True},  # change only every few years
    {"q": "What is the latest stable version of Kubernetes?", "a": "v1.31 (as of late 2025)", "cacheable": True},
    {"q": "Which company acquired GitHub and when?", "a": "Microsoft in 2018 (completed 2020)", "cacheable": True},

    # ==============================================================================
    # 3. TIME-SENSITIVE — MUST NOT BE CACHED
    # ==============================================================================
    {"q": "What is the current price of Bitcoin in USD?", "a": None, "cacheable": False},
    {"q": "What time is it right now in Tokyo?", "a": None, "cacheable": False},
    {"q": "What's the weather like in London today?", "a": None, "cacheable": False},
    {"q": "Who won the last Formula 1 race?", "a": None, "cacheable": False},
    {"q": "What are today's top headlines on Hacker News?", "a": None, "cacheable": False},
    {"q": "What is the current exchange rate EUR to JPY?", "a": None, "cacheable": False},
    {"q": "How many people have visited grok.x.ai today?", "a": None, "cacheable": False},

    # ==============================================================================
    # 4. MATH / EXACT COMPUTATION — MUST NOT BE CACHED
    # ==============================================================================
    {"q": "What is 47 × 83?", "a": None, "cacheable": False},
    {"q": "Solve 3x + 5 = 23 for x.", "a": None, "cacheable": False},
    {"q": "What is the 19th prime number?", "a": None, "cacheable": False},
    {"q": "Calculate the determinant of [[1,2],[3,4]].", "a": None, "cacheable": False},
    {"q": "What is sin(π/3) exactly?", "a": "√3/2", "cacheable": True},  # evergreen math fact
    {"q": "Compute 2^20.", "a": None, "cacheable": False},

    # ==============================================================================
    # 5. LOGICAL PUZZLES / REASONING — MUST NOT BE CACHED (high variance)
    # ==============================================================================
    {"q": "You have 9 balls, one is heavier, using a balance scale in 2 weighings, find it.", "a": None, "cacheable": False},
    {"q": "Einstein's riddle: who owns the fish?", "a": None, "cacheable": False},
    {"q": "Three people check into a hotel room that costs $30...", "a": None, "cacheable": False},
    {"q": "There are 5 houses in a row, each of a different color...", "a": None, "cacheable": False},
    {"q": "Can you solve this Sudoku: [grid]?", "a": None, "cacheable": False},

    # ==============================================================================
    # 6. CREATIVE / OPEN-ENDED / PERSONALIZED — MUST NOT BE CACHED
    # ==============================================================================
    {"q": "Write a funny poem about a cat who loves lasagna.", "a": None, "cacheable": False},
    {"q": "Give me 10 creative name ideas for a coffee shop run by programmers.", "a": None, "cacheable": False},
    {"q": "Help me write a breakup text that's kind but firm.", "a": None, "cacheable": False},
    {"q": "Suggest a 7-day workout plan for a beginner who hates running.", "a": None, "cacheable": False},
    {"q": "Role-play as Elon Musk answering questions about Mars colonization.", "a": None, "cacheable": False},
    {"q": "Generate a bedtime story for a 6-year-old who loves dinosaurs.", "a": None, "cacheable": False},

    # ==============================================================================
    # 7. CODE GENERATION — usually NOT cacheable (high variance)
    # ==============================================================================
    {"q": "Write a FastAPI endpoint that accepts JSON and returns uppercase strings.", "a": None, "cacheable": False},
    {"q": "Implement binary search in Rust.", "a": None, "cacheable": False},
    {"q": "Create a React component for a todo list with drag-and-drop.", "a": None, "cacheable": False},
    {"q": "Write a Docker Compose file for PostgreSQL + Redis + Nginx.", "a": None, "cacheable": False},
    {"q": "What is the syntax for a Python list comprehension?", "a": "[expr for item in iterable if condition]", "cacheable": True},  # trivial syntax → cacheable

    # ==============================================================================
    # 8. TRIVIAL ONE-LINERS — perfect for exact + semantic cache hits
    # ==============================================================================
    {"q": "What port does HTTPS use by default?", "a": "443", "cacheable": True},
    {"q": "What does JSON stand for?", "a": "JavaScript Object Notation", "cacheable": True},
    {"q": "What is the file extension for Python scripts?", "a": ".py", "cacheable": True},
    {"q": "Which HTTP method is idempotent: POST or PUT?", "a": "PUT", "cacheable": True},
    {"q": "What is the time complexity of accessing a hash map?", "a": "O(1) average", "cacheable": True},
    {"q": "What does SQL stand for?", "a": "Structured Query Language", "cacheable": True},
    {"q": "What is the default port for PostgreSQL?", "a": "5432", "cacheable": True},

    # ==============================================================================
    # 9. SEMANTIC EDGE CASES — rephrased versions of the same meaning
    #    → Great for testing embedding similarity threshold in hybrid search
    # ==============================================================================
    {"q": "In which city is the Eiffel Tower located?", "a": "Paris", "cacheable": True},  # same as "capital of France" semantically close
    {"q": "Tell me the capital city of the country famous for the Eiffel Tower.", "a": "Paris", "cacheable": True},
    {"q": "Python was created in what year?", "a": "1991", "cacheable": True},
    {"q": "When did Guido van Rossum release the first version of Python?", "a": "1991", "cacheable": True},
    {"q": "What is the boiling point of water at sea level in Celsius?", "a": "100°C", "cacheable": True},
    {"q": "At standard pressure, water boils at how many degrees Celsius?", "a": "100°C", "cacheable": True},

    # ==============================================================================
    # 10. LONG BUT EVERGREEN EXPLANATIONS — test chunking + semantic retrieval
    # ==============================================================================
    {"q": "Explain how HTTPS works step by step including TLS handshake.", "a": """1. Client hello...""", "cacheable": True},
    {"q": "Give a detailed explanation of how Git rebase differs from merge.", "a": """Rebase rewrites history...""", "cacheable": True},
    {"q": "Describe the CAP theorem and its implications for distributed databases.", "a": """Consistency, Availability, Partition tolerance...""", "cacheable": True},
]

questions = [item["q"] for item in test_dataset]
expected_answers = [item["a"] for item in test_dataset]           # None means should NOT be cached
expected_cacheable = [item["cacheable"] for item in test_dataset]

# Also update load_and_import_mock_data to use real cacheable questions
def load_and_import_mock_data(cache_obj, max_pairs=None):
    # Only import questions that are marked as cacheable=True
    cacheable_items = [item for item in test_dataset if item["cacheable"]]
    if max_pairs:
        cacheable_items = cacheable_items[:max_pairs]
    
    questions_to_cache = [item["q"] for item in cacheable_items]
    answers_to_cache = [item["a"] for item in cacheable_items]
    
    print(f"Pre-populating cache with {len(questions_to_cache)} evergreen questions...")
    cache_obj.import_data(questions=questions_to_cache, answers=answers_to_cache)
    
    return [{"query": q, "expected": a} for q, a in zip(questions_to_cache, answers_to_cache)]

print("\nSLM Classification Accuracy".center(60, "="))
correct = 0
for item in test_dataset:
    answer = llm.invoke(item["q"]).strip()
    predicted_cache = (answer == "yes")
    if predicted_cache == item["cacheable"]:
        correct += 1

accuracy = correct / len(test_dataset)
print(f"SLM Accuracy: {correct}/{len(test_dataset)} → {accuracy:.1%}")
print("=" * 60)

modes = [True, False]
cache_modes = ["all"]
for hybrid_mode in modes:
    for cache_mode in cache_modes:
        print(f"\n--- {'Hybrid' if hybrid_mode else 'Dense'} Mode Performance --- Cache Mode: {cache_mode}")
        
        qdrant = VectorBase(
            "qdrant",
            top_k=10,
            dimension=onnx.dimension,
            location=":memory:",
            hybrid=hybrid_mode
        )
        data_manager = get_data_manager(CacheBase("sqlite"), qdrant)
        llm_cache, cached_llm = init_cache_with_ollama(
            llm, 
            embedding_func=onnx.to_embeddings,
            data_manager=data_manager,
            cache_mode=cache_mode,
            similarity_evaluation=SearchDistanceEvaluation()
        )
        llm_cache.set_openai_key()
        llm_cache.config.similarity_threshold = 0.2
        
        benchmark_pairs = load_and_import_mock_data(llm_cache)
        
        benchmark_hits = 0
        benchmark_misses = 0
        benchmark_latencies = []
        benchmark_failures = 0

        print("Running exact-match benchmark on pre-cached questions...")
        for item in test_dataset:
            if not item["cacheable"]:
                continue  # skip non-cacheable for benchmark
            query = item["q"]
            expected_answer = str(item["a"]).strip()

            try:
                start_time = time.time()
                response = cached_llm(query, cache_obj=llm_cache)
                elapsed = time.time() - start_time
                benchmark_latencies.append(elapsed)
                answer = llm_cache.response_text(response).strip()

                if answer == expected_answer:
                    benchmark_hits += 1
                else:
                    benchmark_misses += 1
                    print(f"MISS MATCH: Q: {query[:60]}... | Got: '{answer}' | Expected: '{expected_answer}'")
            except Exception as e:
                print(f"ERROR on query: {query}\n{e}")
                benchmark_failures += 1

        total_benchmark = benchmark_hits + benchmark_misses
        hit_rate = benchmark_hits / total_benchmark if total_benchmark > 0 else 0
        avg_time = sum(benchmark_latencies) / len(benchmark_latencies) if benchmark_latencies else 0

        print(f"Exact Match Recall (on cached items): {benchmark_hits}/{total_benchmark} ({hit_rate:.1%})")
        print(f"Avg Cache Hit Latency: {avg_time:.3f}s")
        
        print("Evaluating full test suite (cacheable vs non-cacheable detection)...")
        cache_hits = 0
        total_time = []
        results = []

        for idx, item in enumerate(test_dataset):
            question = item["q"]
            should_be_cached = item["cacheable"]

            start_time = time.time()
            response = cached_llm(question, cache_obj=llm_cache)
            elapsed = time.time() - start_time
            total_time.append(elapsed)
            answer = llm_cache.response_text(response)
            is_hit = elapsed < 1.5  # tighter threshold for true cache hit

            if is_hit:
                cache_hits += 1
            results.append((question, answer, is_hit, elapsed))

        print(f"Raw Cache Hit Rate: {cache_hits}/{len(test_dataset)} ({cache_hits/len(test_dataset)*100:.1f}%)")
        print(f"Avg Response Time: {np.mean(total_time):.3f}s ± {np.std(total_time):.3f}s")

        # Proper confusion matrix using ground truth "cacheable" label
        TP = sum(1 for item, (_, _, is_hit, _) in zip(test_dataset, results) if is_hit and item["cacheable"])
        TN = sum(1 for item, (_, _, is_hit, _) in zip(test_dataset, results) if not is_hit and not item["cacheable"])
        FP = sum(1 for item, (_, _, is_hit, _) in zip(test_dataset, results) if is_hit and not item["cacheable"])
        FN = sum(1 for item, (_, _, is_hit, _) in zip(test_dataset, results) if not is_hit and item["cacheable"])

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nSemantic Cache Quality (vs Ground Truth):")
        print(f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}")
        print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
        print(f"{'Hybrid' if hybrid_mode else 'Dense'}-only mode | Threshold: {llm_cache.config.similarity_threshold}")
        print("-" * 60)