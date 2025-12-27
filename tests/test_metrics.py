"""Tests for metrics module."""

import pytest
import tempfile
import gzip
import json
from pathlib import Path

from search_ann_benchmark.core.metrics import (
    MetricsCalculator,
    BenchmarkMetrics,
    ResultsWriter,
    SearchResultsWriter,
)


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_calculate_precision_perfect(self) -> None:
        """Test precision calculation with perfect match."""
        result_ids = [1, 2, 3, 4, 5]
        truth_ids = [1, 2, 3, 4, 5]
        precision = MetricsCalculator.calculate_precision(result_ids, truth_ids, k=5)
        assert precision == 1.0

    def test_calculate_precision_partial(self) -> None:
        """Test precision calculation with partial match."""
        result_ids = [1, 2, 3, 4, 5]
        truth_ids = [1, 2, 6, 7, 8]
        precision = MetricsCalculator.calculate_precision(result_ids, truth_ids, k=5)
        assert precision == 0.4  # 2 out of 5 match

    def test_calculate_precision_no_match(self) -> None:
        """Test precision calculation with no match."""
        result_ids = [1, 2, 3]
        truth_ids = [4, 5, 6]
        precision = MetricsCalculator.calculate_precision(result_ids, truth_ids, k=3)
        assert precision == 0.0

    def test_calculate_precision_empty_truth(self) -> None:
        """Test precision calculation with empty ground truth."""
        result_ids = [1, 2, 3]
        truth_ids: list[int] = []
        precision = MetricsCalculator.calculate_precision(result_ids, truth_ids, k=3)
        assert precision == 0.0

    def test_calculate_statistics(self) -> None:
        """Test statistics calculation."""
        import pandas as pd

        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = MetricsCalculator.calculate_statistics(series)

        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert "50%" in stats
        assert "99%" in stats


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics."""

    def test_to_dict(self) -> None:
        """Test metrics to dictionary conversion."""
        metrics = BenchmarkMetrics(
            num_of_queries=1000,
            took={"mean": 5.0, "std": 1.0},
            hits={"mean": 10.0},
            precision={"mean": 0.95},
        )
        result = metrics.to_dict()

        assert result["num_of_queries"] == 1000
        assert result["took"]["mean"] == 5.0
        assert result["precision"]["mean"] == 0.95


class TestSearchResultsWriter:
    """Tests for SearchResultsWriter."""

    def test_write_results(self) -> None:
        """Test writing search results to gzipped file."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False) as f:
            output_path = f.name

        try:
            with SearchResultsWriter(output_path) as writer:
                writer.write_result(
                    query_id=1,
                    took=5.0,
                    hits=10,
                    ids=[1, 2, 3],
                    scores=[0.9, 0.8, 0.7],
                )
                writer.write_result(
                    query_id=2,
                    took=6.0,
                    hits=10,
                    ids=[4, 5, 6],
                    scores=[0.85, 0.75, 0.65],
                )

            # Verify written content
            with gzip.open(output_path, "rt") as f:
                lines = f.readlines()
                assert len(lines) == 2

                result1 = json.loads(lines[0])
                assert result1["id"] == 1
                assert result1["took"] == 5.0
                assert result1["ids"] == [1, 2, 3]

        finally:
            Path(output_path).unlink(missing_ok=True)


class TestResultsWriter:
    """Tests for ResultsWriter."""

    def test_save_results(self) -> None:
        """Test saving benchmark results."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            writer = ResultsWriter(output_path)
            writer.save(
                target_config="100k-768-m32-efc200-ef100-ip",
                version="1.0.0",
                settings={"test": "value"},
                results={"top_10": {"num_of_queries": 1000}},
                variant="test",
            )

            # Verify written content
            with open(output_path) as f:
                result = json.load(f)

            assert result["target"] == "100k-768-m32-efc200-ef100-ip"
            assert result["version"] == "1.0.0"
            assert result["variant"] == "test"
            assert "timestamp" in result

        finally:
            Path(output_path).unlink(missing_ok=True)
