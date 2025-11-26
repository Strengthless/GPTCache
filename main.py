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
# qdrant = VectorBase(
#     "qdrant",
#     top_k=10,
#     dimension=onnx.dimension,
#     location=":memory:",
#     hybrid=False # True for hybrid search mode, False for dense only
# )
# data_manager = get_data_manager(CacheBase("sqlite"), qdrant)
llm = OllamaLLM(
    model="hf.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M",
    validate_model_on_init=True,
    temperature=0.8,
    num_predict=256
)

# llm_cache, cached_llm = init_cache_with_ollama(
#     llm, 
#     embedding_func=onnx.to_embeddings,
#     data_manager=data_manager,
#     similarity_evaluation=SearchDistanceEvaluation()
# )

# llm_cache.set_openai_key()
# llm_cache.config.similarity_threshold = 0.2  # Set similarity threshold for cache hits


def load_and_import_mock_data(cache_obj, max_pairs=10):
    questions = list(test_questions)
    if max_pairs and max_pairs < len(questions):
        questions = questions[:max_pairs]
    answers = [str(i) for i in range(len(questions))]
    benchmark_rows = [{"query": q, "expected": answers[i]} for i, q in enumerate(questions)]
    cache_obj.import_data(questions=questions, answers=answers)
    return benchmark_rows

test_questions = [
    "What is a smartphone?",
    "Explain how a refrigerator works.",
    "What is the primary use of a microwave oven?",
    "What features does a modern laptop typically have?",
    "Name popular smartphone operating systems.",
    "What's the difference between iOS and Android?",
    "How do video calls work on a phone?",
    "What is HIPAA and what does it regulate?",
    "Outline the key rights under CCPA.",
    "What are the penalties for non-compliance with data privacy laws?",
    "What's the current exchange rate for GBP to USD?",
    "How can I track live currency fluctuations?",
    "What is Bitbucket used for?",
    "How does code branching function in Git?",
    "Why is source control important in software development?",
    "Provide a basic recipe for chocolate chip cookies.",
    "How long to bake brownies at 350F?",
    "Tips for training a cat to use a litter box.",
    "Describe the water cycle in nature.",
    "Who invented the telephone and in what year?",
    "What is the capital of Japan?",
    "What does TCP/IP stand for?",
    "Key principles of GraphQL APIs.",
    "Why is integration testing valuable?",
    "Example of a while loop in JavaScript.",
    "Simple definition of artificial intelligence.",
    "What is unsupervised learning?",
    "What is the value of pi to two decimal places?",
    "What time is it now in New York?",
    "What's the weather like in Sydney today?",
    "What is a stack overflow in programming?",
    "Common sorting algorithms.",
    "What does RAM stand for?",
    "Name the continents of the world.",
    "How to recover a lost email account?",
    "High-level overview of data compression.",
    "Steps to make tea using a teapot.",
    "Define NFT in simple terms.",
    "What is decentralized finance (DeFi)?",
    "Guide to a quick yoga session for beginners.",
    "Benefits of a balanced diet.",
    "Ways to reduce stress in daily life.",
    "Purpose of a foreign key in databases.",
    "Differences between MongoDB and MySQL.",
    "How to merge Git branches safely?",
    "What is JWT used for in authentication?",
    "How does biometric authentication work?",
    "What is a URI in web development?",
    "Metrics for evaluating classification models.",
    "Difference between F1-score and accuracy.",
    "What is underfitting in models?",
    "Steps to perform logistic regression.",
    "What is k-fold cross-validation?",
    "How to activate a Python virtual environment?",
    "What is conda in Python ecosystem?",
    "How to generate a requirements.txt file?",
    "What is Kubernetes used for?",
    "Build a simple web server in Node.js.",
    "What is continuous deployment (CD)?",
    "Writing a test with Jest framework.",
    "Threads vs coroutines: what's the difference?",
    "Error handling in JavaScript.",
    "What is a convolutional neural network?",
    "Explain gradient descent optimization.",
    "What is stemming in text processing?",
    "Uses of vector databases.",
    "How to calculate median in a dataset?",
    "Why secure environment variables?",
    "Implementing infinite scrolling in apps.",
    "Bandwidth vs latency: explain.",
    "How do indexes work in databases?",
    "What is strong consistency?",
    "SQL query to group and aggregate data.",
    "Unique key vs primary key.",
    "What is HTTP/2 and its benefits?",
    "Optimizing videos for streaming.",
    "What is SEO in digital marketing?",
    "How to create a grid layout in CSS?",
    "Principles of mobile-first design.",
    "Setting up a Vue.js project.",
    "What is semantic versioning?",
    "Explain ACID properties in transactions.",
    "POST vs GET: differences.",
    "How to restore a MySQL database?",
    "What is auto-scaling in cloud?",
    "Securing data in motion.",
    "Symmetric vs asymmetric encryption.",
    "How does a VPN work?",
    "Differences between HTTP and HTTPS.",
    "What is a domain registrar?",
    "Strategies for error monitoring.",
    "What is telemetry in software?",
    "Best practices for auditing logs.",
    "What are A/B tests?",
    "Optimizing lambda functions.",
    "What is fog computing?",
    "How do reverse proxies work?",
    "Callback vs promise in async code.",
    "Using list comprehensions in Python.",
    "What is zero-trust security?",
    "Blue-green deployment strategy.",
    "What is a service mesh?",
    "Data masking techniques.",
    "Long polling vs websockets.",
    "When to use a pub/sub system?",
    "Load testing tools.",
    "What is circuit breaking in microservices?",
    "Designing retry mechanisms.",
]

test_answers = [
    "A portable device that combines phone, computer, and camera functions.",
    "A refrigerator cools food using a vapor-compression cycle to preserve freshness.",
    "Heating and cooking food quickly using microwave radiation.",
    "Laptops include processors, RAM, storage, displays, and connectivity options.",
    "iOS, Android, and Windows Mobile are common smartphone OS.",
    "iOS is Apple-exclusive with tight integration; Android is open-source and customizable.",
    "Video calls transmit audio and video over internet protocols like VoIP.",
    "HIPAA regulates protected health information in the US healthcare sector.",
    "CCPA gives California residents rights over their personal data.",
    "Penalties include fines, lawsuits, and reputational damage.",
    None,  # time-sensitive
    None,  # time-sensitive
    "Bitbucket is for code hosting and collaboration using Git or Mercurial.",
    "Branching creates independent lines of development in Git.",
    "Source control tracks changes, prevents conflicts, and supports team work.",
    None,  # creative/recipe
    None,  # creative/time may vary
    "Consistency, patience, and positive reinforcement for litter box use.",
    "Water evaporates, condenses into clouds, precipitates, and collects.",
    "Alexander Graham Bell invented the telephone in 1876.",
    "Tokyo is the capital of Japan.",
    "Transmission Control Protocol/Internet Protocol for networking.",
    "GraphQL allows clients to request specific data with a single endpoint.",
    "Integration testing verifies components work together as expected.",
    "let i = 0;\nwhile (i < 5) {\n  console.log(i);\n  i++;\n}",
    "AI simulates human intelligence in machines for tasks like learning.",
    "Unsupervised learning finds patterns in unlabeled data.",
    "3.14",
    None,  # time-sensitive
    None,  # time-sensitive
    "Stack overflow occurs when call stack exceeds allocated memory.",
    "Bubble sort, quicksort, mergesort, heapsort.",
    "Random Access Memory, temporary storage for running programs.",
    "Africa, Antarctica, Asia, Europe, North America, Oceania, South America.",
    "Use recovery options like security questions or alternate emails.",
    "Compression reduces file size by removing redundancies.",
    None,  # creative/steps may vary
    "NFT is a unique digital asset verified on blockchain.",
    "DeFi uses blockchain for financial services without intermediaries.",
    None,  # creative
    "Supports weight management, nutrient intake, and disease prevention.",
    None,  # creative/advice
    "Foreign keys enforce referential integrity between tables.",
    "MongoDB is document-based NoSQL; MySQL is relational SQL.",
    "Use git merge or rebase after reviewing changes.",
    "JWT securely transmits claims between parties as tokens.",
    "Biometrics use unique physical traits like fingerprints for verification.",
    "URI identifies resources on the web, including URLs and URNs.",
    "Use ROC curves, confusion matrices, precision-recall.",
    "F1-score balances precision and recall; accuracy is overall correctness.",
    "Underfitting is when a model is too simple and performs poorly.",
    "Use sigmoid function to model binary outcomes.",
    "Divide data into k subsets and train/test iteratively.",
    "Source .venv/bin/activate on Unix or .venv\\Scripts\\activate on Windows.",
    "Conda manages environments and packages across languages.",
    "Use pip freeze > requirements.txt",
    "Kubernetes orchestrates containerized applications.",
    None,  # creative/code example
    "CD automates releasing code to production after CI.",
    None,  # creative/code
    "Threads are lightweight; coroutines are user-managed for async.",
    "Use try-catch blocks to manage errors in JS.",
    "CNN uses convolutional layers for image feature extraction.",
    "Gradient descent minimizes loss by adjusting parameters.",
    "Stemming reduces words to root forms for normalization.",
    "Vector databases store and query high-dimensional data efficiently.",
    None,  # math/computation
    "Environment variables hold configs; secure to avoid exposure.",
    None,  # creative/implementation
    "Bandwidth is capacity; latency is delay.",
    "Indexes create data structures for faster queries.",
    "Strong consistency ensures latest data is always read.",
    "Use GROUP BY and functions like SUM, COUNT.",
    "Unique key allows nulls; primary key does not.",
    "HTTP/2 improves speed with multiplexing and compression.",
    "Use codecs like H.264 and reduce bitrate.",
    "SEO optimizes sites for better search engine rankings.",
    None,  # creative/code
    "Mobile-first starts design for small screens then scales up.",
    "Use Vue CLI to initialize a new Vue project.",
    "Semantic versioning uses MAJOR.MINOR.PATCH format.",
    "Atomicity, Consistency, Isolation, Durability for reliable transactions.",
    "POST sends data to create; GET retrieves data.",
    "Use mysqldump for backups and mysql for restores.",
    "Auto-scaling adjusts resources based on demand.",
    "Use encryption protocols like TLS.",
    "Symmetric uses same key; asymmetric uses key pairs.",
    "VPN creates secure tunnels over public networks.",
    "HTTPS adds encryption to HTTP.",
    "Domain registrar manages domain name registrations.",
    "Use logging, metrics, and alerting systems.",
    "Telemetry collects data on system performance.",
    "Audit logs for compliance, avoiding PII.",
    None,  # logic/creative
    None,  # creative/optimization
    "Fog computing extends cloud to IoT devices.",
    "Reverse proxies handle requests on behalf of servers.",
    "Callbacks are functions passed; promises handle async results.",
    None,  # creative/code
    "Zero-trust verifies every access regardless of location.",
    "Blue-green deploys to two environments and switches traffic.",
    "Service mesh manages microservice communications.",
    None,  # creative
    "Long polling holds requests; websockets enable bidirectional.",
    "Pub/sub for broadcasting messages to subscribers.",
    None,  # creative/tools
    "Circuit breaking stops calls to failing services.",
    None,  # logic/design
]


cache_hits = 0
total_time = []
results = []

# for idx, question in enumerate(test_questions):
#     start_time = time.time()
#     response = cached_llm(
#         question, cache_obj=llm_cache
#     )
#     elapsed = time.time() - start_time
#     total_time.append(elapsed)
#     answer = llm_cache.response_text(response)
#     is_hit = elapsed < 2.0
#     if is_hit:
#         cache_hits += 1
#     #print(f'Question: {question}')
#     #print("Time consuming: {:.2f}s".format(elapsed))
#     #print(f'Answer: {answer}\n')
#     #print("Cache HIT" if is_hit else "Cache MISS")
#     results.append((question, answer, is_hit, elapsed))

# print(f"Cache Hit Rate: {cache_hits}/{len(test_questions)} ({cache_hits/len(test_questions)*100:.1f}%)")
# print(f"Average Response Time: {sum(total_time)/len(total_time):.2f}s")

# # Improved Top-1 accuracy: Use embedding similarity for better evaluation
# correct = 0
# for i, (question, answer, is_hit, elapsed) in enumerate(results):
#     expected = test_answers[i] if i < len(test_answers) else None
#     if expected:
#         # Compute embedding similarity (cosine similarity)
#         emb_expected = onnx.to_embeddings(expected)
#         emb_answer = onnx.to_embeddings(answer)
#         similarity = np.dot(emb_expected, emb_answer) / (np.linalg.norm(emb_expected) * np.linalg.norm(emb_answer))
#         if similarity > 0.8:  # Threshold for "correct" match
#             correct += 1
# print(f"Top-1 Accuracy: {correct}/{len(test_questions)} ({correct/len(test_questions)*100:.1f}%)")

# # Confusion Matrix Metrics
# TP = 0  # True Positive: Cache hit and expected answer exists (should be hit)
# TN = 0  # True Negative: Cache miss and expected answer is None (should be miss)
# FP = 0  # False Positive: Cache hit but expected answer is None (incorrect hit)
# FN = 0  # False Negative: Cache miss but expected answer exists (missed hit)

# for i, (question, answer, is_hit, elapsed) in enumerate(results):
#     expected = test_answers[i] 
#     if is_hit and expected is not None:
#         TP += 1
#     elif not is_hit and expected is None:
#         TN += 1
#     elif is_hit and expected is None:
#         FP += 1
#     elif not is_hit and expected is not None:
#         FN += 1

# print(f"\nConfusion Matrix:")
# print(f"True Positive (TP): {TP} - Correct cache hits for relevant queries")
# print(f"True Negative (TN): {TN} - Correct cache misses for unrelated queries")
# print(f"False Positive (FP): {FP} - Incorrect cache hits for unrelated queries")
# print(f"False Negative (FN): {FN} - Missed cache hits for relevant queries")

# # Derived metrics
# precision = TP / (TP + FP) if (TP + FP) > 0 else 0
# recall = TP / (TP + FN) if (TP + FN) > 0 else 0
# f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# print(f"\nDerived Metrics:")
# print(f"Precision: {precision:.2f} - Accuracy of cache hits")
# print(f"Recall: {recall:.2f} - Ability to find relevant cache hits")
# print(f"F1 Score: {f1_score:.2f} - Balanced measure of precision and recall")

# 
    # print(f"Question: {question}")
    # print(f"SLM Decision: {'Cache' if should_cache else 'Skip'}")
    # print(f"Expected: {'Cache' if expected_cache else 'Skip'}")
    # print(f"Correct: {should_cache == expected_cache}")
    # print()
#print(f"SLM Accuracy: {correct_classifications}/{len(test_questions)} ({correct_classifications/len(test_questions)*100:.1f}%)")

modes = [True, False]
cache_modes = ["all", "selecive"]
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
        
        for pair in benchmark_pairs:
            query = pair["query"]
            expected = pair["expected"]
            try:
                start_time = time.time()
                response = cached_llm(query, cache_obj=llm_cache)
                elapsed = time.time() - start_time
                benchmark_latencies.append(elapsed)
                answer = llm_cache.response_text(response).strip()
                if answer == expected:
                    benchmark_hits += 1
                else:
                    benchmark_misses += 1
            except Exception as exc:
                benchmark_failures += 1
        
        total_benchmark = len(benchmark_pairs)
        avg_benchmark_time = (
            sum(benchmark_latencies) / len(benchmark_latencies) if benchmark_latencies else 0
        )
        print(f"Benchmark Cache Hits: {benchmark_hits}/{total_benchmark}")
        print(f"Benchmark Avg Response Time: {avg_benchmark_time:.2f}s")
        
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
            results.append((question, answer, is_hit, elapsed))
        
        print(f"Cache Hit Rate: {cache_hits}/{len(test_questions)} ({cache_hits/len(test_questions)*100:.1f}%)")
        print(f"Average Response Time: {sum(total_time)/len(total_time):.2f}s")
        
        #correct = 0
        # for i, (question, answer, is_hit, elapsed) in enumerate(results):
        #     expected = test_answers[i] if i < len(test_answers) else None
        #     if expected:
        #         emb_expected = onnx.to_embeddings(expected)
        #         emb_answer = onnx.to_embeddings(answer)
        #         similarity = np.dot(emb_expected, emb_answer) / (np.linalg.norm(emb_expected) * np.linalg.norm(emb_answer))
        #         if similarity > 0.65:
        #             correct += 1
        # print(f"Top-1 Accuracy: {correct}/{len(test_questions)} ({correct/len(test_questions)*100:.1f}%)")
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i, (question, answer, is_hit, elapsed) in enumerate(results):
            expected = test_answers[i] 
            if is_hit and expected is not None:
                TP += 1
            elif not is_hit and expected is None:
                TN += 1
            elif is_hit and expected is None:
                FP += 1
            elif not is_hit and expected is not None:
                FN += 1
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")