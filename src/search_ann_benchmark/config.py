"""Configuration classes for benchmark settings."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import os


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    content_path: Path = field(default_factory=lambda: Path("dataset/passages-c400-jawiki-20230403"))
    embedding_path: Path = field(
        default_factory=lambda: Path("dataset/passages-c400-jawiki-20230403/multilingual-e5-base-passage")
    )
    num_of_docs: int = 5555583
    index_size: int = 100000
    bulk_size: int = 1000
    index_name: str = "contents"
    distance: str = "dot_product"
    dimension: int = 768
    exact: bool = False
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef: int = 100
    update_docs_per_sec: int = 0
    quantization: str = "none"


@dataclass
class EngineConfig:
    """Base engine configuration."""

    name: str = ""
    host: str = "localhost"
    port: int = 0
    version: str = ""
    container_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    """Combined benchmark configuration."""

    dataset: DatasetConfig
    engine: EngineConfig
    target_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target_name,
            "dataset": asdict(self.dataset),
            "engine": self.engine.to_dict(),
        }


# Predefined dataset configurations
DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "100k-768-m32-efc200-ef100-ip": {
        "content_path": Path("dataset/passages-c400-jawiki-20230403"),
        "embedding_path": Path("dataset/passages-c400-jawiki-20230403/multilingual-e5-base-passage"),
        "num_of_docs": 5555583,
        "index_size": 100000,
        "bulk_size": 1000,
        "index_name": "contents",
        "distance": "dot_product",
        "dimension": 768,
        "exact": False,
        "hnsw_m": 32,
        "hnsw_ef_construction": 200,
        "hnsw_ef": 100,
        "update_docs_per_sec": 0,
        "quantization": "none",
    },
    "1m-768-m48-efc200-ef100-ip": {
        "content_path": Path("dataset/passages-c400-jawiki-20230403"),
        "embedding_path": Path("dataset/passages-c400-jawiki-20230403/multilingual-e5-base-passage"),
        "num_of_docs": 5555583,
        "index_size": 1000000,
        "bulk_size": 1000,
        "index_name": "contents",
        "distance": "dot_product",
        "dimension": 768,
        "exact": False,
        "hnsw_m": 48,
        "hnsw_ef_construction": 200,
        "hnsw_ef": 100,
        "update_docs_per_sec": 0,
        "quantization": "none",
    },
    "5m-768-m48-efc200-ef100-ip": {
        "content_path": Path("dataset/passages-c400-jawiki-20230403"),
        "embedding_path": Path("dataset/passages-c400-jawiki-20230403/multilingual-e5-base-passage"),
        "num_of_docs": 5555583,
        "index_size": 5000000,
        "bulk_size": 1000,
        "index_name": "contents",
        "distance": "dot_product",
        "dimension": 768,
        "exact": False,
        "hnsw_m": 48,
        "hnsw_ef_construction": 200,
        "hnsw_ef": 100,
        "update_docs_per_sec": 0,
        "quantization": "none",
    },
}


def get_dataset_config(target_name: str) -> DatasetConfig:
    """Get dataset configuration by preset name.

    Args:
        target_name: Name of the preset configuration

    Returns:
        DatasetConfig instance

    Raises:
        ValueError: If preset name is not found
    """
    if target_name not in DATASET_PRESETS:
        raise ValueError(f"Unknown target configuration: {target_name}. Available: {list(DATASET_PRESETS.keys())}")

    settings = DATASET_PRESETS[target_name].copy()
    # Override quantization from environment variable if set
    settings["quantization"] = os.getenv("SETTING_QUANTIZATION", settings["quantization"])
    return DatasetConfig(**settings)
