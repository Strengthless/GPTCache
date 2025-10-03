import time
#from gptcache import cache
#from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.utils.ez_integration import init_cache_with_ollama
from langchain_ollama import OllamaLLM

print("Cache loading.....")

onnx = Onnx()
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=onnx.dimension))
# we will normally do cache.init()
llm = OllamaLLM(
    model="llama3.2:1b",
    validate_model_on_init=True,
    temperature=0.8,
    num_predict=256
)

llm_cache, cached_llm =  init_cache_with_ollama(
    llm, 
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation()
)

llm_cache.set_openai_key()
questions = [
    "What's 1+1?",
    "What's 1+2?",
    "what is github?",
    "what's the usage of github?",
    "what's 1 * 1?"
]

for question in questions:
    start_time = time.time()
   
    response = cached_llm(
        question, cache_obj=llm_cache
    )
   
    print(f'Question: {question}')
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f'Answer: {llm_cache.response_text(response)}\n')