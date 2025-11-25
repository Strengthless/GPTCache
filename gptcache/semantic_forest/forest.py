"""
Semantic forest proof-of-concept.

The semantic forest keeps a set of semantic trees where every node represents a
single user query (with its own embedding). Cache hits require an exact path
match from the root to the most recent chunk of user queries.
"""

from __future__ import annotations

import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from gptcache.manager.data_manager import DataManager
from gptcache.manager.scalar_data.base import CacheData
from gptcache.similarity_evaluation import SimilarityEvaluation
from gptcache.utils.error import ParamError
from gptcache.utils.log import gptcache_log

if TYPE_CHECKING:  # pragma: no cover
    from gptcache.core import Cache

_EPSILON = 1e-12


@dataclass
class SemanticQuery:
    """Represents a single user utterance."""

    text: str
    turn_index: int = 0


@dataclass
class SemanticChunk:
    """Represents a chunk of user queries."""

    queries: List[SemanticQuery]

    def as_text(self) -> str:
        return "\n".join(q.text.strip() for q in self.queries if q.text).strip()

    def __bool__(self) -> bool:
        return bool(self.queries)


@dataclass
class ForestEmbeddingNode:
    text: str
    turn_index: int
    embedding: np.ndarray


@dataclass
class SemanticChunkEmbedding:
    """Couples a chunk with per-query embeddings."""

    chunk: SemanticChunk
    nodes: List[ForestEmbeddingNode]

    def __post_init__(self):
        if len(self.chunk.queries) != len(self.nodes):
            raise ValueError("Chunk/query mismatch while building semantic chunk embedding.")

    @property
    def size(self) -> int:
        return len(self.nodes)


class SemanticForestChunker:
    """Chunk chat history into sequences of user queries."""

    def __init__(self, max_chunk_size: int = 4, drift_threshold: float = 0.35):
        self.max_chunk_size = max(1, max_chunk_size)
        self.drift_threshold = max(0.0, min(1.0, drift_threshold))

    def __call__(self, data: Dict[str, Any], **_: Dict[str, Any]) -> Tuple[str, SemanticChunk]:
        chunk = self.latest_chunk(data)
        return chunk.as_text(), chunk

    def latest_chunk(self, data: Dict[str, Any]) -> SemanticChunk:
        messages = data.get("messages") or []
        user_queries: List[SemanticQuery] = []
        for idx, message in enumerate(messages):
            role = message.get("role", "user")
            if role != "user":
                continue
            user_queries.append(SemanticQuery(text=message.get("content", ""), turn_index=idx))
        if not user_queries and data.get("prompt"):
            user_queries.append(SemanticQuery(text=data["prompt"], turn_index=len(messages)))
        if not user_queries:
            return SemanticChunk([])
        chunks = self._build_chunks(user_queries)
        return chunks[-1]

    def _build_chunks(self, user_queries: Sequence[SemanticQuery]) -> List[SemanticChunk]:
        chunks: List[List[SemanticQuery]] = []
        current_chunk: List[SemanticQuery] = []
        prev_tokens: Optional[set] = None
        for query in user_queries:
            tokens = self._tokenize(query.text)
            if not current_chunk:
                current_chunk.append(query)
                prev_tokens = tokens
                continue
            if len(current_chunk) >= self.max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = [query]
            else:
                similarity = self._token_overlap(prev_tokens, tokens)
                if similarity < self.drift_threshold:
                    chunks.append(current_chunk)
                    current_chunk = [query]
                else:
                    current_chunk.append(query)
            prev_tokens = tokens
        if current_chunk:
            chunks.append(current_chunk)
        return [SemanticChunk(chunk) for chunk in chunks if chunk]

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token)

    @staticmethod
    def _token_overlap(lhs: Optional[set], rhs: Optional[set]) -> float:
        if not lhs or not rhs:
            return 0.0
        inter = len(lhs & rhs)
        union = len(lhs | rhs)
        return float(inter) / float(union or 1)


class SemanticForestEmbedder:
    """Produces per-query embeddings for a chunk."""

    def __init__(
        self,
        base_embedding_func: Optional[Callable[..., Any]] = None,
        vector_dim: int = 64,
    ):
        self.base_embedding = base_embedding_func
        self.vector_dim = max(8, vector_dim)

    def __call__(self, chunk: SemanticChunk, extra_param: Optional[Dict[str, Any]] = None, **_: Dict[str, Any]) -> SemanticChunkEmbedding:
        if not isinstance(chunk, SemanticChunk):
            chunk = SemanticChunk([SemanticQuery(text=str(chunk), turn_index=0)])
        nodes: List[ForestEmbeddingNode] = []
        extra = extra_param or {}
        embedder = self.base_embedding or extra.get("base_embedding_func")
        for query in chunk.queries:
            vector = self._run_embedder(embedder, query.text)
            nodes.append(ForestEmbeddingNode(text=query.text, turn_index=query.turn_index, embedding=vector))
        return SemanticChunkEmbedding(chunk=chunk, nodes=nodes)

    def _run_embedder(self, embedder: Optional[Callable[..., Any]], text: str) -> np.ndarray:
        if callable(embedder):
            vector = embedder(text)
        else:
            vector = self._hash_embed(text)
        return self._as_vector(vector)

    def _as_vector(self, vector: Any) -> np.ndarray:
        if isinstance(vector, np.ndarray):
            arr = vector.astype("float32")
        elif isinstance(vector, (list, tuple)):
            arr = np.asarray(vector, dtype="float32")
        elif vector is None:
            arr = self._hash_embed("")
        else:
            arr = self._hash_embed(str(vector))
        arr = arr.reshape(-1).astype("float32")
        if not arr.size:
            arr = self._hash_embed("")
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return arr
        return arr / norm

    def _hash_embed(self, text: str) -> np.ndarray:
        buckets = np.zeros(self.vector_dim, dtype="float32")
        if not text:
            return buckets
        for token in text.lower().split():
            buckets[hash(token) % self.vector_dim] += 1.0
        norm = np.linalg.norm(buckets)
        return buckets / norm if norm else buckets


@dataclass
class ForestNode:
    node_id: str
    parent_id: Optional[str]
    depth: int
    query_text: str
    embedding: np.ndarray
    children: List[str] = field(default_factory=list)
    cache_data: Optional[CacheData] = None
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    sessions: set = field(default_factory=set)


class SemanticForestDataManager(DataManager):
    """A minimal in-memory (optional persisted) semantic forest data manager."""

    def __init__(self, data_path: Optional[str] = None, similarity_threshold: float = 0.88, max_nodes: Optional[int] = 4096):
        self.data_path = data_path
        self.similarity_threshold = similarity_threshold
        self.max_nodes = max_nodes
        self.nodes: Dict[str, ForestNode] = {}
        self.root_ids: List[str] = []
        self.sessions: Dict[str, set] = {}
        self.last_report: Optional[dict] = None
        self._dirty = False
        self._load()

    # --- Required DataManager interface implementations ---

    def save(self, question, answer, embedding_data, **kwargs):
        chunk_embedding = self._ensure_chunk_embedding(embedding_data)
        if not chunk_embedding.size:
            return
        cache_data = CacheData(
            question=question,
            answers=answer,
            embedding_data=chunk_embedding.nodes[-1].embedding,
        )
        leaf_node, _ = self._upsert_path(chunk_embedding)
        leaf_node.cache_data = cache_data
        leaf_node.last_access = time.time()
        session = kwargs.get("session", None)
        if session:
            self._store_session(leaf_node.node_id, session.name)
        self._dirty = True

    def import_data(self, questions, answers, embedding_datas, session_ids):
        if not (
            len(questions) == len(answers)
            == len(embedding_datas)
            == len(session_ids)
        ):
            raise ParamError("Make sure that all parameters have the same length")
        for question, answer, embedding, session_id in zip(questions, answers, embedding_datas, session_ids):
            self.save(
                question,
                answer,
                embedding,
                session=_SessionStub(session_id) if session_id else None,
            )

    def get_scalar_data(self, res_data, **kwargs) -> Optional[CacheData]:
        if not res_data:
            return None
        node = self.nodes.get(res_data[1])
        if not node or node.cache_data is None:
            return None
        return node.cache_data

    def hit_cache_callback(self, res_data, **kwargs):
        node = self.nodes.get(res_data[1]) if res_data else None
        if node:
            node.last_access = time.time()

    def search(self, embedding_data, **kwargs):
        chunk_embedding = self._ensure_chunk_embedding(embedding_data)
        if not chunk_embedding.size:
            return []
        results = []
        for root_id in list(self.root_ids):
            match = self._match_path(root_id, chunk_embedding)
            if not match:
                continue
            leaf_node, score = match
            if leaf_node.cache_data is None:
                continue
            if score < self.similarity_threshold:
                continue
            results.append((score, leaf_node.node_id))
        results.sort(key=lambda item: item[0], reverse=True)
        top_k = kwargs.get("top_k", -1)
        if isinstance(top_k, int) and top_k > 0:
            results = results[:top_k]
        return results

    def flush(self):
        if not self._dirty or not self.data_path:
            return
        payload = {
            "nodes": self.nodes,
            "root_ids": self.root_ids,
            "sessions": self.sessions,
            "config": {
                "similarity_threshold": self.similarity_threshold,
                "max_nodes": self.max_nodes,
            },
        }
        try:
            with open(self.data_path, "wb") as handler:
                pickle.dump(payload, handler)
            self._dirty = False
        except OSError as err:  # pragma: no cover - best effort logging
            gptcache_log.warning("Failed to persist semantic forest to %s: %s", self.data_path, err)

    def add_session(self, res_data, session_id, pre_embedding_data):
        if not res_data:
            return
        self._store_session(res_data[1], session_id)

    def list_sessions(self, session_id=None, key=None):
        if key:
            node = self.nodes.get(key)
            return list(node.sessions) if node else []
        if session_id:
            return list(self.sessions.get(session_id, set()))
        return list(self.sessions.keys())

    def delete_session(self, session_id):
        node_ids = self.sessions.pop(session_id, set())
        for node_id in node_ids:
            node = self.nodes.get(node_id)
            if node:
                node.sessions.discard(session_id)

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ):
        self.last_report = {
            "user_question": user_question,
            "cache_question": cache_question,
            "cache_question_id": cache_question_id,
            "cache_answer": cache_answer,
            "similarity_value": similarity_value,
            "cache_delta_time": cache_delta_time,
        }

    def close(self):
        self.flush()

    # --- helpers ---

    def _ensure_chunk_embedding(self, embedding_data) -> SemanticChunkEmbedding:
        if isinstance(embedding_data, SemanticChunkEmbedding):
            return embedding_data
        if isinstance(embedding_data, np.ndarray):
            normalized = embedding_data.astype("float32")
            norm = np.linalg.norm(normalized)
            normalized = normalized / norm if norm else normalized
            chunk = SemanticChunk([SemanticQuery(text="", turn_index=0)])
            node = ForestEmbeddingNode(text="", turn_index=0, embedding=normalized)
            return SemanticChunkEmbedding(chunk=chunk, nodes=[node])
        raise ValueError("SemanticForestDataManager expected SemanticChunkEmbedding as embedding payload.")

    def _match_path(self, root_id: str, chunk_embedding: SemanticChunkEmbedding) -> Optional[Tuple[ForestNode, float]]:
        root_node = self.nodes.get(root_id)
        if not root_node:
            return None
        if not chunk_embedding.size:
            return None
        first_vector = chunk_embedding.nodes[0].embedding
        first_score = self._similarity(root_node.embedding, first_vector)
        if first_score < self.similarity_threshold:
            return None
        total_score = first_score
        current_node = root_node
        for payload in chunk_embedding.nodes[1:]:
            current_node, score = self._best_child(current_node, payload.embedding)
            if current_node is None or score < self.similarity_threshold:
                return None
            total_score += score
        return current_node, total_score / float(chunk_embedding.size)

    def _best_child(self, node: ForestNode, vector: np.ndarray) -> Tuple[Optional[ForestNode], float]:
        best_node = None
        best_score = -1.0
        for child_id in node.children:
            child = self.nodes.get(child_id)
            if child is None:
                continue
            score = self._similarity(child.embedding, vector)
            if score > best_score:
                best_score = score
                best_node = child
        return best_node, best_score

    def _upsert_path(self, chunk_embedding: SemanticChunkEmbedding) -> Tuple[ForestNode, float]:
        parent_id: Optional[str] = None
        total_score = 0.0
        traversed = 0
        for depth, payload in enumerate(chunk_embedding.nodes):
            matched_node, score = self._best_match(parent_id, payload.embedding)
            if matched_node and score >= self.similarity_threshold:
                target_node = matched_node
            else:
                target_node = self._create_node(parent_id, depth, payload)
                score = 1.0
            parent_id = target_node.node_id
            total_score += score
            traversed += 1
        leaf_node = self.nodes[parent_id] if parent_id else None
        avg_score = total_score / float(traversed or 1)
        return leaf_node, avg_score

    def _best_match(self, parent_id: Optional[str], vector: np.ndarray) -> Tuple[Optional[ForestNode], float]:
        candidates: Iterable[str]
        if parent_id is None:
            candidates = self.root_ids
        else:
            parent = self.nodes.get(parent_id)
            candidates = parent.children if parent else []
        best_node = None
        best_score = -1.0
        for node_id in candidates:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            score = self._similarity(node.embedding, vector)
            if score > best_score:
                best_score = score
                best_node = node
        return best_node, best_score

    def _create_node(self, parent_id: Optional[str], depth: int, payload: ForestEmbeddingNode) -> ForestNode:
        node_id = uuid.uuid4().hex
        node = ForestNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            query_text=payload.text,
            embedding=self._normalize(payload.embedding),
        )
        self.nodes[node_id] = node
        if parent_id is None:
            self.root_ids.append(node_id)
        else:
            parent = self.nodes.get(parent_id)
            if parent:
                parent.children.append(node_id)
        self._dirty = True
        self._enforce_limits()
        return node

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm <= _EPSILON:
            return vector.astype("float32")
        return (vector / norm).astype("float32")

    def _similarity(self, lhs: np.ndarray, rhs: np.ndarray) -> float:
        if lhs is None or rhs is None:
            return 0.0
        denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
        if denom <= _EPSILON:
            return 0.0
        return float(np.dot(lhs, rhs) / denom)

    def _store_session(self, node_id: str, session_id: str):
        if not session_id:
            return
        node = self.nodes.get(node_id)
        if not node:
            return
        node.sessions.add(session_id)
        mapping = self.sessions.setdefault(session_id, set())
        mapping.add(node_id)

    def _enforce_limits(self):
        if self.max_nodes is None:
            return
        while len(self.nodes) > self.max_nodes and self.root_ids:
            root_id = self.root_ids.pop(0)
            self._delete_subtree(root_id)
            self._dirty = True

    def _delete_subtree(self, node_id: str):
        node = self.nodes.pop(node_id, None)
        if node is None:
            return
        for session_id in list(node.sessions):
            members = self.sessions.get(session_id)
            if members:
                members.discard(node_id)
                if not members:
                    self.sessions.pop(session_id)
        for child_id in list(node.children):
            self._delete_subtree(child_id)
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent:
                parent.children = [cid for cid in parent.children if cid != node_id]

    def _load(self):
        if not self.data_path or not os.path.exists(self.data_path):
            return
        try:
            with open(self.data_path, "rb") as handler:
                payload = pickle.load(handler)
            self.nodes = payload.get("nodes", {})
            self.root_ids = payload.get("root_ids", [])
            self.sessions = payload.get("sessions", {})
            config = payload.get("config", {})
            self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)
            self.max_nodes = config.get("max_nodes", self.max_nodes)
        except (OSError, pickle.UnpicklingError) as err:  # pragma: no cover - i/o best effort
            gptcache_log.warning("Failed to load semantic forest from %s: %s", self.data_path, err)


class SemanticForestEvaluation(SimilarityEvaluation):
    """Reuse the search score provided by the forest traversal."""

    def evaluation(self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs) -> float:
        search_result = cache_dict.get("search_result")
        if not search_result:
            return 0.0
        return float(search_result[0])

    def range(self) -> Tuple[float, float]:
        return 0.0, 1.0


def enable_semantic_forest(
    cache_obj: "Cache",
    *,
    data_path: Optional[str] = None,
    chunk_size: int = 4,
    drift_threshold: float = 0.35,
    node_match_threshold: float = 0.88,
    max_nodes: Optional[int] = 4096,
    base_embedding_func: Optional[Callable[..., Any]] = None,
    vector_dim: int = 64,
) -> SemanticForestDataManager:
    """Helper that wires the semantic forest into an existing cache instance."""

    chunker = SemanticForestChunker(max_chunk_size=chunk_size, drift_threshold=drift_threshold)
    embedder = SemanticForestEmbedder(base_embedding_func=base_embedding_func, vector_dim=vector_dim)
    data_manager = SemanticForestDataManager(
        data_path=data_path,
        similarity_threshold=node_match_threshold,
        max_nodes=max_nodes,
    )
    evaluation = SemanticForestEvaluation()

    if cache_obj.has_init:
        cache_obj.pre_embedding_func = chunker
        cache_obj.embedding_func = embedder
        cache_obj.data_manager = data_manager
        cache_obj.similarity_evaluation = evaluation
    else:
        cache_obj.init(
            pre_embedding_func=chunker,
            embedding_func=embedder,
            data_manager=data_manager,
            similarity_evaluation=evaluation,
        )
    return data_manager


class _SessionStub:
    """Minimal session shim used during import."""

    def __init__(self, name: Optional[str]):
        self.name = name
