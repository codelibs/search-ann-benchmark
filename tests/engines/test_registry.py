"""Tests for engine registry."""

import pytest

from search_ann_benchmark.engines import ENGINE_REGISTRY, get_engine_class


class TestEngineRegistry:
    """Tests for engine registry."""

    def test_all_engines_registered(self) -> None:
        """Test that all expected engines are registered."""
        expected_engines = [
            "qdrant",
            "elasticsearch",
            "opensearch",
            "milvus",
            "weaviate",
            "vespa",
            "pgvector",
            "chroma",
            "clickhouse",
            "lancedb",
            "redisstack",
            "vald",
        ]
        for engine in expected_engines:
            assert engine in ENGINE_REGISTRY, f"Engine {engine} not registered"

    def test_get_engine_class_valid(self) -> None:
        """Test getting a valid engine class."""
        engine_class = get_engine_class("qdrant")
        assert engine_class is not None
        assert engine_class.engine_name == "qdrant"

    def test_get_engine_class_invalid(self) -> None:
        """Test getting an invalid engine raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_engine_class("invalid_engine")
        assert "Unknown engine" in str(exc_info.value)

    def test_engine_classes_have_required_attributes(self) -> None:
        """Test that all engine classes have required attributes."""
        for name, engine_class in ENGINE_REGISTRY.items():
            assert hasattr(engine_class, "engine_name"), f"{name} missing engine_name"
            assert hasattr(engine_class, "get_docker_command"), f"{name} missing get_docker_command"
            assert hasattr(engine_class, "wait_until_ready"), f"{name} missing wait_until_ready"
            assert hasattr(engine_class, "create_index"), f"{name} missing create_index"
            assert hasattr(engine_class, "delete_index"), f"{name} missing delete_index"
            assert hasattr(engine_class, "search"), f"{name} missing search"
            assert hasattr(engine_class, "insert_documents"), f"{name} missing insert_documents"
