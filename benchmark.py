import json
import os
import time
import numpy as np

from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.utils.ez_integration import init_cache_with_ollama
from langchain_ollama import OllamaLLM

# --------------------------- CONFIG ---------------------------
MODEL_NAME = "hf.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M"  # or any Ollama model you have pulled
SIMILARITY_THRESHOLD = 0.95   # Very strict, like in your second script
TOP_K = 10
CACHE_DB_PATH = "sqlite.db"
FAISS_INDEX_PATH = "faiss.index"
MOCK_DATA_FILE = "examples/benchmark/mock_data.json"
# -------------------------------------------------------------

def load_mock_data(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("Loading mock data...")
    mock_data = load_mock_data(MOCK_DATA_FILE)

    # ------------------- Embedding & Vector Store -------------------
    onnx = Onnx()  # uses default onnx.to_embeddings
    vector_base = VectorBase(
        "faiss",
        dimension=onnx.dimension,
        index_path=FAISS_INDEX_PATH,
        top_k=TOP_K,
        location="local",  # persist to disk
    )
    cache_base = CacheBase("sqlite", sqlite_file=CACHE_DB_PATH)

    data_manager = get_data_manager(cache_base, vector_base, max_size=100000)

    # ------------------- LLM & Cache Init -------------------
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

    # Optional: set openai key if you ever want to mix OpenAI calls
    # llm_cache.set_openai_key("dummy")  # not needed for Ollama

    # ------------------- Pre-populate Cache -------------------
    if not (os.path.isfile(CACHE_DB_PATH) and os.path.isfile(FAISS_INDEX_PATH) or len(mock_data) == 0):
        print("Pre-populating cache with mock_data.json...")
        questions = [pair["origin"] for pair in mock_data]
        answers = [pair["id"]]  # we store the ID as the "answer" so we can check exact hit
        start_time = time.time()
        llm_cache.import_data(questions=questions, answers=answers)
        print(f"Cache populated in {time.time() - start_time:.2f}s")

    # ------------------- Test Loop -------------------
    cache_hits = 0
    total_time = []
    correct_hits = 0
    total_queries = len(mock_data)

    print("\nRunning test queries...")
    for idx, pair in enumerate(mock_data):
        query = pair["similar"]  # this is the paraphrased query

        start_time = time.time()
        response = cached_llm.invoke(query, cache_obj=llm_cache)  # invoke for LangChain style
        elapsed = time.time() - start_time

        answer = llm_cache.response_text(response)
        total_time.append(elapsed)

        # Cache hit detection: fast response + has_cache() check
        is_hit = elapsed < 1.5 and llm_cache.has_cache(response)

        if is_hit and answer == pair["id"]:
            cache_hits += 1
            correct_hits += 1
        elif is_hit and answer != pair["id"]:
            # False positive: cached but returned wrong answer (very rare with threshold=0.95)
            print(f"WARNING: False hit on idx {idx}: got '{answer}' instead of '{pair['id']}'")

        #print(f"{idx+1:2d}. Q: {query}")
        #print(f"   A: {answer} | Time: {elapsed:.3:.2f}s {'HIT' if is_hit else 'MISS'}")

    # ------------------- Metrics -------------------
    hit_rate = cache_hits / total_queries * 100
    avg_time = sum(total_queries) / len(total_queries)

    print("\n" + "="*50)
    print(" " * 18 + "FINAL RESULTS")
    print("="*50)
    print(f"Cache Hit Rate      : {cache_hits}/{total_queries} ({hit_rate:.1f}%)")
    print(f"Average Time        : {avg_time:.2f}s")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"Top-1 Accuracy     : {correct_hits}/{total_queries} ({correct_hits/total_queries*100:.1f}%)")

    # Bonus: embedding similarity for semantic accuracy (like your first script)
    print("\nSemantic similarity check on correct hits:")
    semantic_correct = 0
    for pair in mock_data:
        expected = pair.get("origin", pair["similar"])
        emb1 = onnx.to_embeddings(expected)
        emb2 = onnx.to_embeddings(pair["similar"])
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        if sim > 0.8:  # reasonable semantic threshold
            semantic_correct += 1

    print(f"Semantic similarity > 0.8: {semantic_correct}/{total_queries} ({semantic_correct/total_queries*100:.1f}%)")

if __name__ == "__main__":
    main()