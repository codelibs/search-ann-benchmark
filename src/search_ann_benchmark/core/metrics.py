"""Metrics calculation utilities."""

import gzip
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""

    num_of_queries: int = 0
    took: dict[str, float] = field(default_factory=dict)
    hits: dict[str, float] = field(default_factory=dict)
    precision: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"num_of_queries": self.num_of_queries}
        if self.took:
            result["took"] = self.took
        if self.hits:
            result["hits"] = self.hits
        if self.precision:
            result["precision"] = self.precision
        return result


class MetricsCalculator:
    """Calculates benchmark metrics from search results."""

    @staticmethod
    def calculate_statistics(series: pd.Series) -> dict[str, float]:
        """Calculate descriptive statistics for a series.

        Args:
            series: Pandas series of values

        Returns:
            Dictionary with mean, std, min, max, and percentiles
        """
        return {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "25%": float(series.quantile(0.25)),
            "50%": float(series.quantile(0.5)),
            "75%": float(series.quantile(0.75)),
            "90%": float(series.quantile(0.9)),
            "99%": float(series.quantile(0.99)),
            "max": float(series.max()),
        }

    @staticmethod
    def calculate_precision(result_ids: list[int], truth_ids: list[int], k: int) -> float:
        """Calculate precision@k.

        Args:
            result_ids: List of result document IDs
            truth_ids: List of ground truth document IDs
            k: Number of results to consider

        Returns:
            Precision value between 0 and 1
        """
        size = min(len(truth_ids), k)
        if size == 0:
            return 0.0
        intersection = len(set(result_ids[:k]).intersection(set(truth_ids[:k])))
        return intersection / size

    def calculate_from_file(
        self,
        results_file: str | Path,
        ground_truth_file: str | Path | None = None,
        k: int = 10,
    ) -> BenchmarkMetrics:
        """Calculate metrics from a results file.

        Args:
            results_file: Path to gzipped JSONL results file
            ground_truth_file: Path to ground truth file for precision calculation
            k: Number of results for precision@k

        Returns:
            BenchmarkMetrics with calculated statistics
        """
        df = pd.read_json(results_file, lines=True)

        metrics = BenchmarkMetrics(
            num_of_queries=len(df),
            took=self.calculate_statistics(df["took"]),
            hits={
                "mean": float(df["hits"].mean()),
                "std": float(df["hits"].std()),
                "min": float(df["hits"].min()),
                "25%": float(df["hits"].quantile(0.25)),
                "50%": float(df["hits"].quantile(0.5)),
                "75%": float(df["hits"].quantile(0.75)),
                "max": float(df["hits"].max()),
            },
        )

        # Calculate precision if ground truth is available
        if ground_truth_file and os.path.exists(ground_truth_file):
            truth_df = pd.read_json(ground_truth_file, lines=True)[["id", "ids"]].rename(columns={"ids": "truth_ids"})
            df = pd.merge(df, truth_df, on="id", how="inner")

            def get_precision(row: pd.Series) -> float:
                return self.calculate_precision(row["ids"], row["truth_ids"], k)

            df["precision"] = df.apply(get_precision, axis=1)
            metrics.precision = self.calculate_statistics(df["precision"])

        # Print summary
        print(df.describe().to_markdown())

        return metrics


class ResultsWriter:
    """Writes benchmark results to file."""

    def __init__(self, output_path: str | Path = "results.json"):
        """Initialize results writer.

        Args:
            output_path: Path to write results JSON
        """
        self.output_path = Path(output_path)

    def save(
        self,
        target_config: str,
        version: str,
        settings: dict[str, Any],
        results: dict[str, Any],
        variant: str = "",
    ) -> None:
        """Save benchmark results to JSON file.

        Args:
            target_config: Configuration name
            version: Engine version
            settings: Configuration settings dictionary
            results: Results dictionary with metrics
            variant: Engine variant (e.g., "int4", "bbq")
        """
        variant = variant or os.getenv("PRODUCT_VARIANT", "")

        output = {
            "variant": variant,
            "target": target_config,
            "version": version,
            "settings": settings,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

        def json_serializer(obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, Path):
                return str(obj)
            return None

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, default=json_serializer, indent=2)

        print(f"Results saved to {self.output_path}")


class SearchResultsWriter:
    """Writes search results to gzipped JSONL files."""

    def __init__(self, output_path: str | Path):
        """Initialize search results writer.

        Args:
            output_path: Path to write gzipped JSONL file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def __enter__(self) -> "SearchResultsWriter":
        self._file = gzip.open(self.output_path, "wt", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._file:
            self._file.close()

    def write_result(
        self,
        query_id: int,
        took: float,
        hits: int,
        ids: list[int],
        scores: list[float],
        total_hits: int = 0,
    ) -> None:
        """Write a single search result.

        Args:
            query_id: Query identifier
            took: Query time in milliseconds
            hits: Number of hits returned
            ids: List of result document IDs
            scores: List of result scores
            total_hits: Total number of matching documents
        """
        if not self._file:
            raise RuntimeError("Writer not opened. Use 'with' statement.")

        result = {
            "id": query_id,
            "took": took,
            "hits": hits,
            "ids": ids,
            "scores": scores,
        }
        if total_hits:
            result["total_hits"] = total_hits

        self._file.write(json.dumps(result, ensure_ascii=False))
        self._file.write("\n")
