"""Tests for configuration module."""

import pytest
from pathlib import Path

from search_ann_benchmark.config import (
    DatasetConfig,
    EngineConfig,
    BenchmarkConfig,
    DATASET_PRESETS,
    get_dataset_config,
)


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DatasetConfig()
        assert config.dimension == 768
        assert config.bulk_size == 1000
        assert config.quantization == "none"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = DatasetConfig(
            index_size=500000,
            hnsw_m=48,
            quantization="int8",
        )
        assert config.index_size == 500000
        assert config.hnsw_m == 48
        assert config.quantization == "int8"


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_to_dict(self) -> None:
        """Test configuration to dictionary conversion."""
        config = EngineConfig(
            name="test_engine",
            host="localhost",
            port=9999,
            version="1.0.0",
            container_name="test_container",
        )
        result = config.to_dict()
        assert result["name"] == "test_engine"
        assert result["port"] == 9999
        assert result["version"] == "1.0.0"


class TestDatasetPresets:
    """Tests for dataset presets."""

    def test_presets_exist(self) -> None:
        """Test that expected presets exist."""
        assert "100k-768-m32-efc200-ef100-ip" in DATASET_PRESETS
        assert "1m-768-m48-efc200-ef100-ip" in DATASET_PRESETS
        assert "5m-768-m48-efc200-ef100-ip" in DATASET_PRESETS

    def test_get_dataset_config_valid(self) -> None:
        """Test getting a valid preset."""
        config = get_dataset_config("100k-768-m32-efc200-ef100-ip")
        assert config.index_size == 100000
        assert config.hnsw_m == 32

    def test_get_dataset_config_invalid(self) -> None:
        """Test getting an invalid preset raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_dataset_config("invalid-preset")
        assert "Unknown target configuration" in str(exc_info.value)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_to_dict(self) -> None:
        """Test combined configuration to dictionary."""
        dataset = DatasetConfig(index_size=1000)
        engine = EngineConfig(name="test", port=8080)
        config = BenchmarkConfig(
            dataset=dataset,
            engine=engine,
            target_name="test-target",
        )
        result = config.to_dict()
        assert result["target"] == "test-target"
        assert "dataset" in result
        assert "engine" in result
