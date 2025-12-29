"""Vector search engine implementations."""

from .qdrant import QdrantEngine, QdrantConfig
from .elasticsearch import ElasticsearchEngine, ElasticsearchConfig
from .opensearch import OpenSearchEngine, OpenSearchConfig
from .milvus import MilvusEngine, MilvusConfig
from .weaviate import WeaviateEngine, WeaviateConfig
from .vespa import VespaEngine, VespaConfig
from .pgvector import PgvectorEngine, PgvectorConfig
from .chroma import ChromaEngine, ChromaConfig
from .lancedb import LanceDBEngine, LancedbConfig

ENGINE_REGISTRY: dict[str, type] = {
    "qdrant": QdrantEngine,
    "elasticsearch": ElasticsearchEngine,
    "opensearch": OpenSearchEngine,
    "milvus": MilvusEngine,
    "weaviate": WeaviateEngine,
    "vespa": VespaEngine,
    "pgvector": PgvectorEngine,
    "chroma": ChromaEngine,
    "lancedb": LanceDBEngine,
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
    "LanceDBEngine",
    "LancedbConfig",
    "ENGINE_REGISTRY",
    "get_engine_class",
]
