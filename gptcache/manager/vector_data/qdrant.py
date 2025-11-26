from typing import List, Optional

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_qdrant
from gptcache.utils.log import gptcache_log

import_qdrant()

# pylint: disable=C0413
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    PointStruct,
    HnswConfigDiff,
    VectorParams,
    OptimizersConfigDiff,
    Distance,
)


class QdrantVectorStore(VectorBase):
    """Qdrant Vector Store"""

    def __init__(
        self,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        collection_name: Optional[str] = "gptcache",
        location: Optional[str] = "./qdrant",
        dimension: int = 0,
        top_k: int = 1,
        flush_interval_sec: int = 5,
        index_params: Optional[dict] = None,
        hybrid: bool = True,
    ):
        if dimension <= 0:
            raise ValueError(
                f"invalid `dim` param: {dimension} in the Qdrant vector store."
            )
        self._client: QdrantClient
        self._collection_name = collection_name
        self._in_memory = location == ":memory:"
        self.dimension = dimension
        self.top_k = top_k
        self.hybrid = hybrid
        
        if self._in_memory or location is not None:
            self._create_local(location)
        else:
            self._create_remote(
                url, port, api_key, timeout, host, grpc_port, prefer_grpc, prefix, https
            )
        self._client.delete_collection(collection_name=self._collection_name)
        self._create_collection(collection_name, flush_interval_sec, index_params)
        
    def _create_local(self, location):
        self._client = QdrantClient(location=location)

    def _create_remote(
        self, url, port, api_key, timeout, host, grpc_port, prefer_grpc, prefix, https
    ):
        self._client = QdrantClient(
            url=url,
            port=port,
            api_key=api_key,
            timeout=timeout,
            host=host,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            prefix=prefix,
            https=https,
        )

    def _create_collection(
        self,
        collection_name: str,
        flush_interval_sec: int,
        index_params: Optional[dict] = None,
    ):
        hnsw_config = HnswConfigDiff(**(index_params or {}))
        vectors_config = {"embedding": VectorParams(
            size=self.dimension, distance=Distance.COSINE, hnsw_config=hnsw_config
        )}
        
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
        optimizers_config = OptimizersConfigDiff(
            deleted_threshold=0.2,
            vacuum_min_vector_number=1000,
            flush_interval_sec=flush_interval_sec,
        )
        # check if the collection exists
        existing_collections = self._client.get_collections()
        for existing_collection in existing_collections.collections:
            if existing_collection.name == collection_name:
                gptcache_log.warning(
                    "The %s collection already exists, and it will be used directly.",
                    collection_name,
                )
                break
        else:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                optimizers_config=optimizers_config,
            )

    def mul_add(self, datas: List[VectorData]):
        
        points = [
            PointStruct(
                id=d.id,
                vector={
                    "embedding": d.data["embedding"],    # Already SparseVector object
                    "bm25": d.data["bm25"]
                }
            )
            for d in datas
        ]
        try:
            self._client.upsert(
                collection_name=self._collection_name, points=points, wait=False
            )
        except Exception as e:
            print("upsert error: ", e)

    def search(self, data: np.ndarray, bm25, top_k: int = -1):
        if self.hybrid:
            if top_k == -1:
                top_k = self.top_k
            reshaped_data = data.reshape(-1).tolist()
            points = self._client.scroll(
                collection_name=self._collection_name,
                limit=10,  # Retrieve up to 10 points
            )
            #print("Points in Collection:", points)
            #print("searching cache: ", bm25[0])
            prefetch = [
                    models.Prefetch(
                        query=reshaped_data,
                        using="embedding",
                        limit=5,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(**bm25[0].as_object()),
                        using="bm25",
                        limit=5,
                    ),
            ]
            search_result = self._client.query_points(
                collection_name=self._collection_name,
                prefetch=prefetch,
                limit=top_k,
                query=  models.FusionQuery(fusion=models.Fusion.RRF)
            )
            #print("result:", search_result)
            if search_result != []:
                return list(map(lambda x: (x.score, x.id), search_result.points))
            return []
        else:
            return self._search(data, bm25, top_k)
    
    ## dense only search
    def _search(self, data: np.ndarray, bm25, top_k: int = -1):
        if top_k == -1:
            top_k = self.top_k
        reshaped_data = data.reshape(-1).tolist()
        search_result = self._client.query_points(
            collection_name=self._collection_name,
            query=reshaped_data,
            using="embedding",
            limit=top_k
        )
        #print("result:", search_result)
        if search_result != []:
            return list(map(lambda x: (x.score, x.id), search_result.points))
        return []

    def delete(self, ids: List[str]):
        self._client.delete(collection_name=self._collection_name, points_selector=ids)

    def rebuild(self, ids=None):  # pylint: disable=unused-argument
        optimizers_config = OptimizersConfigDiff(
            deleted_threshold=0.2, vacuum_min_vector_number=1000
        )
        self._client.update_collection(
            collection_name=self._collection_name, optimizer_config=optimizers_config
        )

    def flush(self):
        # no need to flush manually as qdrant flushes automatically based on the optimizers_config for remote Qdrant
        pass

    def close(self):
        self.flush()
