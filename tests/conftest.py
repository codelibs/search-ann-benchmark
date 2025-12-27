"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path

from search_ann_benchmark.config import DatasetConfig


@pytest.fixture
def sample_dataset_config() -> DatasetConfig:
    """Create a sample dataset configuration for testing."""
    return DatasetConfig(
        content_path=Path("dataset/passages-c400-jawiki-20230403"),
        embedding_path=Path("dataset/passages-c400-jawiki-20230403/multilingual-e5-base-passage"),
        num_of_docs=5555583,
        index_size=1000,  # Small for testing
        bulk_size=100,
        index_name="test_contents",
        distance="dot_product",
        dimension=768,
        exact=False,
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef=100,
        update_docs_per_sec=0,
        quantization="none",
    )


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample embedding vector."""
    import numpy as np

    np.random.seed(42)
    embedding = np.random.randn(768).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


@pytest.fixture
def sample_documents() -> list[dict]:
    """Create sample documents for testing."""
    return [
        {"page_id": 1, "rev_id": 100, "section": "test_section_1"},
        {"page_id": 2, "rev_id": 101, "section": "test_section_2"},
        {"page_id": 3, "rev_id": 102, "section": "test_section_1"},
    ]
