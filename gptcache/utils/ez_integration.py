from gptcache.core import cache, Cache
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.utils.cache_func import cache_selectively, cache_all
def get_content_func(data, **_):
    return data.get("prompt").split("Question")[-1]

def init_cache_with_ollama(ollama_llm, embedding_func, data_manager, similarity_evaluation, cache_mode="all"):
    llm_cache = Cache()
    llm_cache.init(
        embedding_func=embedding_func,
        data_manager=data_manager,
        cache_enable_func=cache_all if "all" else cache_selectively,
        pre_embedding_func=get_content_func,
        similarity_evaluation=similarity_evaluation,
        llm_provider="ollama",
        bm25=True
    )
    cached_llm  = LangChainLLMs(llm=ollama_llm)
    return llm_cache , cached_llm