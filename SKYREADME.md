## GPTCache: Hybrid Semantic Caching with SLM Filtering (Project Extension for CUHK Final Year Project, Ongoing)

### What is new in this project?

- **SLM-based cacheability filter**: Uses a quantized Small Language Model (Meta-Llama-3-8B-Instruct, 4-bit) to classify queries as cacheable (static/factual) or not (dynamic/time-sensitive/creative) before storing in cache. This prevents cache pollution and improves factual accuracy.
- **Hybrid search**: Combines dense semantic embeddings (ONNX) with sparse BM25 keyword search using Qdrant's hybrid mode. Results are fused using Reciprocal Rank Fusion (RRF) for robust retrieval.
- **Labeled test set generated synthetically**: 59 diverse queries, each labeled as cacheable (with answer) or non-cacheable (answer=None), for realistic benchmarking.
- **Automated benchmarking**: Evaluates cache hit rate, latency, precision, recall, F1, and SLM classification accuracy for both hybrid and dense-only modes.

---

### Quick Setup

```bash
uv pip install langchain_ollama transformers openai==0.28.1 langchain onnxruntime==1.21.1
```

## Local LLM setup (4-bit quantized Llama-3-8B-Instruct, feel) ##
```bash
ollama pull hf.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M # feel free to use another model
```
---

### Example Usage

```python
from langchain_ollama import OllamaLLM
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.utils.ez_integration import init_cache_with_ollama

onnx = Onnx()
qdrant = VectorBase(
    "qdrant",
    top_k=10,
    dimension=onnx.dimension,
    location=":memory:",
    hybrid=True  # Set to False for dense-only search
)
data_manager = get_data_manager(CacheBase("sqlite"), qdrant)

llm = OllamaLLM(
    model="llama3.2:1b",  # Choose your own model, make sure it is installed locally
    validate_model_on_init=True,
    temperature=0.8,
    num_predict=256,
)

llm_cache, cached_llm = init_cache_with_ollama(
    llm,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation()
)

llm_cache.set_openai_key()

questions = [
    "what's github",
    "can you explain what GitHub is",
    "can you tell me more about GitHub",
    "what is the purpose of GitHub"
]

for question in questions:
    start_time = time.time()
    response = cached_llm(
        question, cache_obj=llm_cache
    )
    print(f'Question: {question}')
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f'Answer: {llm_cache.response_text(response)}\n')
```

---

### Project Summary

This project extends GPTCache with two key innovations:

1. **SLM-based cacheability filter**: A quantized Llama-3-8B-Instruct model (4-bit) classifies each query as cacheable or not, using a carefully designed prompt. Only static, factual queries are cached, preventing pollution from time-sensitive or creative queries.
2. **Hybrid retrieval**: Dense ONNX embeddings and sparse BM25 vectors are fused using Reciprocal Rank Fusion (RRF) in Qdrant, improving retrieval precision for both paraphrased and keyword-heavy queries.

**Evaluation**: The system is benchmarked on 108+ labeled queries (cacheable and non-cacheable). Metrics include cache hit rate, latency, precision, recall, F1, and SLM accuracy. Results show improved cache hygiene and retrieval quality over dense-only or naive caching baselines.

---

### How to Run the Full Benchmark (Preliminary, detailed evaluation in future)

```bash
python main.py
```

The script prints detailed results for both hybrid and dense search, as well as SLM classification accuracy.

---
