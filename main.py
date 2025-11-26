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
    hybrid=True # True for hybrid search mode, False for dense only
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


def load_and_import_mock_data(cache_obj, max_pairs=10):
    questions = list(test_questions)
    if max_pairs and max_pairs < len(questions):
        questions = questions[:max_pairs]
    answers = [str(i) for i in range(len(questions))]
    benchmark_rows = [{"query": q, "expected": answers[i]} for i, q in enumerate(questions)]
    cache_obj.import_data(questions=questions, answers=answers)
    return benchmark_rows






test_questions = [
    "What is a TV?",
    "Describe the function of a television.",
    "What is a TV used for?",
    "What are the main features of a Smart TV?",
    "Name common TV display technologies",
    "Explain the difference between LED and OLED",
    "How do streaming services work on a TV?",
    "What is GDPR and who does it protect?",
    "List the core principles of GDPR",
    "When must organizations report a data breach under GDPR?",
    "How do I convert USD to EUR?",
    "Where can I find the current exchange rate?",
    "What is GitHub used for?",
    "How do pull requests work?",
    "Explain the purpose of version control",
    "Give a simple cake recipe",
    "How long should I bake a sponge cake at 180C?",
    "What are basic dog training tips?",
    "How does photosynthesis work in plants?",
    "Who discovered penicillin and when?",
    "What is the capital city of France?",
    "Define HTTP",
    "What are REST API principles?",
    "What is unit testing and why use it?",
    "Show a Python for-loop example",
    "Define machine learning in simple terms",
    "What is supervised learning?",
    "What is the approximate speed of light?",
    "How do I check the current time in London?",
    "How can I check the weather forecast?",
    "What is recursion in programming?",
    "Name common data structures",
    "What does CPU stand for?",
    "List the eight planets in our solar system",
    "How do I reset a forgotten password on a website?",
    "Explain encryption at a high level",
    "How do you brew coffee with a pour-over?",
    "What is a blockchain in one sentence?",
    "What is cryptocurrency?",
    "How can I start a short mindfulness meditation?",
    "List benefits of regular exercise",
    "How do I improve my sleep quality?",
    "What is the purpose of a database index?",
    "Explain difference between SQL and NoSQL",
    "How to safely delete a Git branch?",
    "What is OAuth used for?",
    "How does two-factor authentication (2FA) work?",
    "What is a RESTful resource?",
    "How to measure the accuracy of a model?",
    "Explain precision vs recall",
    "What is overfitting in machine learning?",
    "How to perform a linear regression?",
    "What is cross-validation?",
    "How to create a virtual environment in Python?",
    "What is the Python package manager?",
    "How to install dependencies from requirements.txt?",
    "What is a Docker container?",
    "How to build a simple HTTP server in Python?",
    "What is continuous integration (CI)?",
    "How to write a unit test using pytest?",
    "What is the difference between threads and processes?",
    "How to handle exceptions in Python?",
    "What is a neural network?",
    "How does backpropagation work?",
    "What is tokenization in NLP?",
    "What are embeddings used for?",
    "How to find the mean of a list in Python?",
    "What is an API key and why keep it secret?",
    "How to paginate API responses?",
    "What is latency vs throughput?",
    "How do caches improve performance?",
    "What is eventual consistency?",
    "How to write SQL to join two tables?",
    "Explain primary key vs foreign key",
    "What is HTTPS and why use it?",
    "How to compress images for the web?",
    "What is accessibility (a11y) in web design?",
    "How to center a div horizontally and vertically in CSS?",
    "What is responsive web design?",
    "How to set up a basic React app?",
    "What is version pinning and why does it matter?",
    "Explain the CAP theorem in databases",
    "What is the difference between PUT and PATCH?",
    "How to back up a PostgreSQL database?",
    "What is load balancing?",
    "How to secure sensitive data at rest?",
    "What is a public key vs private key?",
    "How does DNS work at a high level?",
    "Explain the difference between IPv4 and IPv6",
    "What is an SSL/TLS certificate?",
    "How to monitor application health?",
    "What is observability?",
    "How to log events responsibly?",
    "What is feature flagging?",
    "How to reduce cold start time in serverless functions?",
    "What is edge computing?",
    "How does content delivery network (CDN) work?",
    "What is the difference between synchronous and asynchronous code?",
    "How to use map/filter/reduce in Python?",
    "What is the principle of least privilege?",
    "How to perform a rolling deployment?",
    "What is service discovery?",
    "How to encrypt data in transit?",
    "Explain webhook vs polling",
    "What is a message queue and when to use one?",
    "How to test API endpoints automatically?",
    "What is rate limiting and why implement it?",
    "How to design an idempotent API endpoint?",
]

test_answers = [
    "An electronic device for receiving and displaying audio-visual content.",
    "A television displays video and sound for entertainment and information.",
    "Watching shows, movies, news and streaming content.",
    "Smart TVs provide streaming apps, internet access, and apps.",
    "LCD, LED, OLED, and QLED are common TV display types.",
    "OLED uses organic LEDs for per-pixel illumination and deeper blacks.",
    "Streaming services deliver video over the internet to the TV.",
    "GDPR is an EU regulation protecting personal data of individuals.",
    "Core GDPR principles include lawfulness, fairness, and transparency.",
    "Organizations must report certain data breaches to authorities promptly.",
    "Use a live exchange-rate API or financial website to convert currencies.",
    "Exchange rates are published by banks, exchanges, and financial sites.",
    "Hosting, sharing, and collaborating on code repositories.",
    "A pull request proposes code changes and allows review before merging.",
    "Version control records changes to files and enables collaboration.",
    "Mix flour, sugar, eggs, butter; bake until set at about 180°C (350°F).",
    "Typical sponge cake bakes 25–35 minutes at 180°C, depending on pan size.",
    "Be consistent, use positive reinforcement, and short training sessions.",
    "Plants convert light to chemical energy via chlorophyll in photosynthesis.",
    "Alexander Fleming discovered penicillin in 1928.",
    "Paris is the capital of France.",
    "HTTP is the protocol used to transfer web pages and resources.",
    "REST uses stateless operations and standard HTTP verbs for APIs.",
    "Unit testing checks small code units (functions, classes) for correctness.",
    "for item in iterable:\n    print(item)",
    "Machine learning trains models to make predictions from data.",
    "Supervised learning trains on labeled input-output pairs.",
    "About 299,792 kilometers per second in vacuum.",
    "Use world time websites or system timezones to get London time.",
    "Use weather APIs or forecast websites to get a local forecast.",
    "Recursion is when a function calls itself to solve subproblems.",
    "Examples: arrays, linked lists, stacks, queues, trees, graphs.",
    "CPU = Central Processing Unit, the main processor of a computer.",
    "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
    "Follow the website's password-reset flow or contact support.",
    "Encryption converts data into a ciphertext to prevent reading by others.",
    "Pour hot water over grounds in a filter, let brew, then serve.",
    "A blockchain is an append-only distributed ledger of transactions.",
    "Cryptocurrency is a digital currency secured by cryptography.",
    "Sit comfortably, focus on breath for 2–5 minutes, observe thoughts.",
    "Improves cardiovascular health, mood, energy, and sleep.",
    "Maintain a regular schedule, avoid screens, and limit caffeine.",
    "Indexes speed lookups by providing quick pointers to rows in tables.",
    "SQL uses structured tables; NoSQL uses flexible schemas for different data models.",
    "Delete the branch locally, then delete it on the remote if needed.",
    "OAuth provides delegated authorization for applications.",
    "2FA requires two different forms of identity, e.g., password + code.",
    "A RESTful resource is a unique entity identified by a URL and manipulated with HTTP verbs.",
    "Use accuracy, precision, recall, or domain-specific metrics to measure models.",
    "Precision measures relevant results among retrieved; recall measures retrieved among relevant.",
    "Overfitting occurs when a model fits training data too closely and fails to generalize.",
    "Fit a line by minimizing squared errors between predictions and targets.",
    "Split data and validate model performance on unseen folds.",
    "Create a venv: python -m venv .venv",
    "pip is the standard Python package manager.",
    "Run pip install -r requirements.txt to install dependencies.",
    "A Docker container packages an application and its dependencies.",
    "Use Python's http.server or frameworks like Flask for HTTP servers.",
    "CI automates building, testing, and validating code changes.",
    "Use pytest to write test functions and assert expected behavior.",
    "Threads share memory within a process; processes have separate memory spaces.",
    "Use try/except blocks to catch and handle exceptions in Python.",
    "A neural network is a layered model that learns features from data.",
    "Backpropagation computes gradients to update model weights during training.",
    "Tokenization splits text into tokens (words/subwords) for NLP models.",
    "Embeddings map text to numeric vectors capturing semantic relationships.",
    "Use sum(values)/len(values) or numpy.mean for the list mean.",
    "An API key authenticates a client; keep it secret to prevent abuse.",
    "Return limited results and provide links or tokens for subsequent pages.",
    "Latency is delay; throughput is the amount processed per time unit.",
    "Caches store recent results to reduce computation and latency.",
    "Eventual consistency means updates propagate and reach consistent state over time.",
    "Use SQL JOIN clauses (INNER JOIN, LEFT JOIN) with matching keys.",
    "Primary key uniquely identifies a row; foreign key references another table's primary key.",
    "HTTPS encrypts HTTP with TLS to secure web traffic.",
    "Compress images by reducing dimensions or using efficient formats like WebP.",
    "Accessibility ensures people with disabilities can access and use web content.",
    "Use CSS flexbox or grid and centering techniques to align a div.",
    "Responsive design adapts layout to different screen sizes using fluid grids and media queries.",
    "Use create-react-app or Vite to scaffold a basic React application.",
    "Pinning fixes dependency versions to avoid unexpected upgrades.",
    "CAP theorem: you can have at most two of Consistency, Availability, Partition tolerance.",
    "PUT replaces a resource; PATCH applies partial modifications.",
    "Use pg_dump or managed backup tools to back up PostgreSQL.",
    "Load balancing distributes requests over multiple servers to improve availability.",
    "Encrypt disk volumes and restrict access to secure data at rest.",
    "Public key encrypts or verifies; private key decrypts or signs.",
    "DNS maps domain names to IP addresses using resolvers and authoritative servers.",
    "IPv4 uses 32-bit addresses; IPv6 uses 128-bit addresses to increase address space.",
    "An SSL/TLS certificate proves site identity and enables encrypted connections.",
    "Monitor health via metrics, alerts, and health-check endpoints.",
    "Observability is the ability to infer internal state from logs, metrics, and traces.",
    "Log errors, context, and avoid sensitive data; rotate and index logs.",
    "Feature flags let you enable/disable functionality without redeploying.",
    "Keep functions warm, reduce init work, or use provisioned concurrency to reduce cold starts.",
    "Edge computing runs computation close to users to reduce latency.",
    "A CDN caches static content geographically close to users.",
    "Synchronous code blocks until completion; asynchronous code can run concurrently.",
    "Use map for transformations, filter for selecting, reduce for aggregations.",
    "Principle of least privilege grants minimal access needed for tasks.",
    "Rolling deployment updates instances gradually to minimize disruptions.",
    "Service discovery allows services to find each other, often via a registry.",
    "Encrypt data over TLS to protect it in transit.",
    "Webhooks push events to listeners; polling repeatedly asks for updates.",
    "Message queues decouple producers and consumers and buffer workloads.",
    "Use automated integration tests with known inputs and expected outputs.",
    "Rate limiting prevents abuse by capping requests per time period.",
    "Design operations to be idempotent so repeated calls have the same effect.",
]


cache_hits = 0
total_time = []
results = []

for idx, question in enumerate(test_questions):
    start_time = time.time()
    response = cached_llm(
        question, cache_obj=llm_cache
    )
    elapsed = time.time() - start_time
    total_time.append(elapsed)
    answer = llm_cache.response_text(response)
    is_hit = elapsed < 2.0
    if is_hit:
        cache_hits += 1
    #print(f'Question: {question}')
    #print("Time consuming: {:.2f}s".format(elapsed))
    #print(f'Answer: {answer}\n')
    #print("Cache HIT" if is_hit else "Cache MISS")
    results.append((question, answer, is_hit, elapsed))

print("\n--- Hybrid Cache Metrics ---")
print(f"Cache Hit Rate: {cache_hits}/{len(test_questions)} ({cache_hits/len(test_questions)*100:.1f}%)")
print(f"Average Response Time: {sum(total_time)/len(total_time):.2f}s")

# Improved Top-1 accuracy: Use embedding similarity for better evaluation
correct = 0
for i, (question, answer, is_hit, elapsed) in enumerate(results):
    expected = test_answers[i] if i < len(test_answers) else None
    if expected:
        # Compute embedding similarity (cosine similarity)
        emb_expected = onnx.to_embeddings(expected)
        emb_answer = onnx.to_embeddings(answer)
        similarity = np.dot(emb_expected, emb_answer) / (np.linalg.norm(emb_expected) * np.linalg.norm(emb_answer))
        if similarity > 0.5:  # Threshold for "correct" match
            correct += 1
print(f"Top-1 Accuracy: {correct}/{len(test_questions)} ({correct/len(test_questions)*100:.1f}%)")

# Confusion Matrix Metrics
TP = 0  # True Positive: Cache hit and expected answer exists (should be hit)
TN = 0  # True Negative: Cache miss and expected answer is None (should be miss)
FP = 0  # False Positive: Cache hit but expected answer is None (incorrect hit)
FN = 0  # False Negative: Cache miss but expected answer exists (missed hit)

for i, (question, answer, is_hit, elapsed) in enumerate(results):
    expected = test_answers[i] if i < len(test_answers) else None
    if is_hit and expected is not None:
        TP += 1
    elif not is_hit and expected is None:
        TN += 1
    elif is_hit and expected is None:
        FP += 1
    elif not is_hit and expected is not None:
        FN += 1

print(f"\nConfusion Matrix:")
print(f"True Positive (TP): {TP} - Correct cache hits for relevant queries")
print(f"True Negative (TN): {TN} - Correct cache misses for unrelated queries")
print(f"False Positive (FP): {FP} - Incorrect cache hits for unrelated queries")
print(f"False Negative (FN): {FN} - Missed cache hits for relevant queries")

# Derived metrics
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nDerived Metrics:")
print(f"Precision: {precision:.2f} - Accuracy of cache hits")
print(f"Recall: {recall:.2f} - Ability to find relevant cache hits")
print(f"F1 Score: {f1_score:.2f} - Balanced measure of precision and recall")

# print("\n--- SLM Classification Accuracy ---")
# correct_classifications = 0
# for i, question in enumerate(test_questions):
#     should_cache = cache_selectively(prompt=question)
#     expected_cache = test_answers[i] is not None
#     if should_cache == expected_cache:
#         correct_classifications += 1
#     # print(f"Question: {question}")
#     # print(f"SLM Decision: {'Cache' if should_cache else 'Skip'}")
#     # print(f"Expected: {'Cache' if expected_cache else 'Skip'}")
#     # print(f"Correct: {should_cache == expected_cache}")
#     # print()
# print(f"SLM Accuracy: {correct_classifications}/{len(test_questions)} ({correct_classifications/len(test_questions)*100:.1f}%)")

# modes = [True, False]
# for hybrid_mode in modes:
#     print(f"\n--- {'Hybrid' if hybrid_mode else 'Dense'} Mode Performance ---")
    
#     qdrant = VectorBase(
#         "qdrant",
#         top_k=10,
#         dimension=onnx.dimension,
#         location=":memory:",
#         hybrid=hybrid_mode
#     )
#     data_manager = get_data_manager(CacheBase("sqlite"), qdrant)
#     llm_cache, cached_llm = init_cache_with_ollama(
#         llm, 
#         embedding_func=onnx.to_embeddings,
#         data_manager=data_manager,
#         similarity_evaluation=SearchDistanceEvaluation()
#     )
#     llm_cache.set_openai_key()
#     llm_cache.config.similarity_threshold = 0.2
    
#     benchmark_pairs = load_and_import_mock_data(llm_cache)
    
#     print("\n--- Mock Data Benchmark ---")
#     benchmark_hits = 0
#     benchmark_misses = 0
#     benchmark_latencies = []
#     benchmark_failures = 0
    
#     for pair in benchmark_pairs:
#         query = pair["query"]
#         expected = pair["expected"]
#         try:
#             start_time = time.time()
#             response = cached_llm(query, cache_obj=llm_cache)
#             elapsed = time.time() - start_time
#             benchmark_latencies.append(elapsed)
#             answer = llm_cache.response_text(response).strip()
#             if answer == expected:
#                 benchmark_hits += 1
#             else:
#                 benchmark_misses += 1
#         except Exception as exc:
#             benchmark_failures += 1
    
#     total_benchmark = len(benchmark_pairs)
#     avg_benchmark_time = (
#         sum(benchmark_latencies) / len(benchmark_latencies) if benchmark_latencies else 0
#     )
#     print(f"Benchmark Cache Hits: {benchmark_hits}/{total_benchmark}")
#     print(f"Benchmark Avg Response Time: {avg_benchmark_time:.2f}s")
    
#     cache_hits = 0
#     total_time = []
#     results = []
    
#     for idx, question in enumerate(test_questions):
#         start_time = time.time()
#         response = cached_llm(
#             question, cache_obj=llm_cache
#         )
#         elapsed = time.time() - start_time
#         total_time.append(elapsed)
#         answer = llm_cache.response_text(response)
#         is_hit = elapsed < 2.0
#         if is_hit:
#             cache_hits += 1
#         results.append((question, answer, is_hit, elapsed))
    
#     print(f"Cache Hit Rate: {cache_hits}/{len(test_questions)} ({cache_hits/len(test_questions)*100:.1f}%)")
#     print(f"Average Response Time: {sum(total_time)/len(total_time):.2f}s")
    
#     correct = 0
#     for i, (question, answer, is_hit, elapsed) in enumerate(results):
#         expected = test_answers[i] if i < len(test_answers) else None
#         if expected:
#             emb_expected = onnx.to_embeddings(expected)
#             emb_answer = onnx.to_embeddings(answer)
#             similarity = np.dot(emb_expected, emb_answer) / (np.linalg.norm(emb_expected) * np.linalg.norm(emb_answer))
#             if similarity > 0.5:
#                 correct += 1
#     print(f"Top-1 Accuracy: {correct}/{len(test_questions)} ({correct/len(test_questions)*100:.1f}%)")
    
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for i, (question, answer, is_hit, elapsed) in enumerate(results):
#         expected = test_answers[i] if i < len(test_answers) else None
#         if is_hit and expected is not None:
#             TP += 1
#         elif not is_hit and expected is None:
#             TN += 1
#         elif is_hit and expected is None:
#             FP += 1
#         elif not is_hit and expected is not None:
#             FN += 1
    
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
#     print(f"Precision: {precision:.2f}")
#     print(f"Recall: {recall:.2f}")
#     print(f"F1 Score: {f1_score:.2f}")