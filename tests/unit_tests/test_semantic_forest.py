import os
import tempfile

from gptcache import cache
from gptcache.semantic_forest import (
    SemanticForestChunker,
    SemanticForestDataManager,
    SemanticForestEmbedder,
    enable_semantic_forest,
)
from gptcache.semantic_forest.forest import SemanticChunk, SemanticQuery


def _build_messages(*user_texts):
    messages = []
    for text in user_texts:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": "ack"})
    return messages


def _chunk(chunker, *user_texts):
    return chunker({"messages": _build_messages(*user_texts)})


def _manual_chunk(*texts):
    queries = [SemanticQuery(text=text, turn_index=idx) for idx, text in enumerate(texts)]
    return "\n".join(texts), SemanticChunk(queries)


def test_semantic_forest_cache_hits_read_from_tree():
    chunker = SemanticForestChunker(max_chunk_size=3)
    embedder = SemanticForestEmbedder(vector_dim=32)
    manager = SemanticForestDataManager(similarity_threshold=0.7, max_nodes=128)

    chunk_text, chunk = _chunk(
        chunker,
        "teach me recursion",
        "show a python example",
    )
    chunk_embedding = embedder(chunk)
    manager.save(chunk_text, "cached recursion answer", chunk_embedding)

    search_results = manager.search(chunk_embedding)
    assert len(search_results) == 1
    node_id = search_results[0][1]
    assert node_id in manager.nodes

    cache_data = manager.get_scalar_data(search_results[0])
    assert cache_data.answers[0].answer == "cached recursion answer"
    assert cache_data.question == chunk_text


def test_semantic_forest_cache_miss_updates_tree():
    chunker = SemanticForestChunker(max_chunk_size=4)
    embedder = SemanticForestEmbedder(vector_dim=48)
    manager = SemanticForestDataManager(similarity_threshold=0.8, max_nodes=256)

    base_text, base_chunk = _chunk(
        chunker,
        "optimize sql queries",
        "give indexing tips",
    )
    base_embedding = embedder(base_chunk)
    manager.save(base_text, "use covering indexes", base_embedding)

    extended_text, extended_chunk = _chunk(
        chunker,
        "optimize sql queries",
        "give indexing tips",
        "what about shard strategies",
    )
    extended_embedding = embedder(extended_chunk)
    assert manager.search(extended_embedding) == [], "unknown path should miss"

    manager.save(extended_text, "discuss sharding", extended_embedding)
    hits = manager.search(extended_embedding)
    assert len(hits) == 1

    leaf_node = manager.nodes[hits[0][1]]
    assert leaf_node.depth == len(extended_chunk.queries) - 1
    assert leaf_node.cache_data.answers[0].answer == "discuss sharding"


def test_semantic_forest_prevents_false_positive_from_semantic_dilution():
    embedder = SemanticForestEmbedder(vector_dim=64)
    manager = SemanticForestDataManager(similarity_threshold=0.9, max_nodes=256)

    first_text, first_chunk = _manual_chunk(
        "apple is good for your health",
        "apple is bad for your health",
        "list more reasons",
    )
    first_embedding = embedder(first_chunk)
    manager.save(first_text, "apple is good because... apple is bad because...", first_embedding)

    blended_text, blended_chunk = _manual_chunk(
        "apple is both good and bad for your health",
        "list more reasons",
    )
    blended_embedding = embedder(blended_chunk)
    assert manager.search(blended_embedding) == [], "semantic blend should miss"

    repeated_text, repeated_chunk = _manual_chunk(
        "apple is both good and bad for your health",
        "apple is both good and bad for your health",
        "list more reasons",
    )
    repeated_embedding = embedder(repeated_chunk)
    assert manager.search(repeated_embedding) == [], "repeating blended query should still miss"


def test_semantic_forest_semantic_drift_creates_new_tree():
    chunker = SemanticForestChunker(max_chunk_size=2, drift_threshold=0.9)
    embedder = SemanticForestEmbedder(vector_dim=64)
    manager = SemanticForestDataManager(similarity_threshold=0.9, max_nodes=64)

    tech_text, tech_chunk = _chunk(
        chunker,
        "debug kubernetes crashloop",
        "inspect pod logs",
    )
    manager.save(tech_text, "use kubectl logs", embedder(tech_chunk))
    assert len(manager.root_ids) == 1

    cooking_text, cooking_chunk = _chunk(
        chunker,
        "how to temper chocolate",
        "fix seized ganache",
    )
    cooking_embedding = embedder(cooking_chunk)
    manager.save(cooking_text, "use a double boiler", cooking_embedding)

    assert len(manager.root_ids) == 2, "drastic topic switch should start a new tree"
    hits = manager.search(cooking_embedding)
    assert hits
    assert manager.get_scalar_data(hits[0]).answers[0].answer == "use a double boiler"


def test_enable_semantic_forest_can_reconfigure_cache():
    tmp_dir = tempfile.mkdtemp()
    data_path = os.path.join(tmp_dir, "forest.pkl")
    cache.init()
    manager = enable_semantic_forest(
        cache,
        data_path=data_path,
        chunk_size=2,
        node_match_threshold=0.5,
    )
    assert isinstance(manager, SemanticForestDataManager)
    assert cache.data_manager is manager
