from langchain_ollama import OllamaLLM
from prompt import SLM_PROMPT_1
llm = OllamaLLM(
    model="llama3.2:3b",
    validate_model_on_init=True,
    temperature=0.7,
    num_predict= 50
)
def cache_all(*_, **__):
    return True

def cache_selectively(*_, **kwargs):
    if 'prompt' in kwargs:
        question = kwargs['prompt']
        prompt = SLM_PROMPT_1.format(question=question)
        response = llm(prompt)
        response_cleaned = response.lower().replace(" ","").replace("\"","")
        if response_cleaned == "yes":
            #print(f"cached {question}")
            return True
        else:
            #print(f"skipped caching {question}")
            return False
    return False