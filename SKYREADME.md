**Easy Integration for ollama and openai**
- easy cache initialization with ollama
- currently supports ollama and opeai chat models

## Quick Setup

`uv pip install langchain_ollama transformers openai==0.28.1 langchain onnxruntime==1.21.1`
```bash
llm = OllamaLLM(
    model="llama3.2:1b", # choose your own model, make sure it has been installed locally
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