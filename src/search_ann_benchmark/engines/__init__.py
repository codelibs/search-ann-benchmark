"""Vector search engine implementations."""

from .chroma import ChromaConfig, ChromaEngine
from .elasticsearch import ElasticsearchConfig, ElasticsearchEngine
from .milvus import MilvusConfig, MilvusEngine
from .opensearch import OpenSearchConfig, OpenSearchEngine
from .pgvector import PgvectorConfig, PgvectorEngine
from .qdrant import QdrantConfig, QdrantEngine
from .redisstack import RedisStackConfig, RedisStackEngine
from .vespa import VespaConfig, VespaEngine
from .weaviate import WeaviateConfig, WeaviateEngine

ENGINE_REGISTRY: dict[str, type] = {
    "qdrant": QdrantEngine,
    "elasticsearch": ElasticsearchEngine,
    "opensearch": OpenSearchEngine,
    "milvus": MilvusEngine,
    "weaviate": WeaviateEngine,
    "vespa": VespaEngine,
    "pgvector": PgvectorEngine,
    "chroma": ChromaEngine,
    "redisstack": RedisStackEngine,
}


def get_engine_class(engine_name: str) -> type:
    """Get engine class by name.

    Args:
        engine_name: Name of the engine

    Returns:
        Engine class

    Raises:
        ValueError: If engine not found
    """
    if engine_name not in ENGINE_REGISTRY:
        raise ValueError(f"Unknown engine: {engine_name}. Available: {list(ENGINE_REGISTRY.keys())}")
    return ENGINE_REGISTRY[engine_name]


__all__ = [
    "QdrantEngine",
    "QdrantConfig",
    "ElasticsearchEngine",
    "ElasticsearchConfig",
    "OpenSearchEngine",
    "OpenSearchConfig",
    "MilvusEngine",
    "MilvusConfig",
    "WeaviateEngine",
    "WeaviateConfig",
    "VespaEngine",
    "VespaConfig",
    "PgvectorEngine",
    "PgvectorConfig",
    "ChromaEngine",
    "ChromaConfig",
    "RedisStackEngine",
    "RedisStackConfig",
    "ENGINE_REGISTRY",
    "get_engine_class",
]
