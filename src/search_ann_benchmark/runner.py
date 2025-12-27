"""Benchmark runner for vector search engines."""

import os
import re
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

from search_ann_benchmark.config import DatasetConfig, get_dataset_config
from search_ann_benchmark.core.base import VectorSearchEngine
from search_ann_benchmark.core.docker import DockerManager
from search_ann_benchmark.core.embedding import ContentLoader, EmbeddingLoader
from search_ann_benchmark.core.metrics import (
    BenchmarkMetrics,
    MetricsCalculator,
    ResultsWriter,
    SearchResultsWriter,
)


class BenchmarkRunner:
    """Orchestrates benchmark execution for a vector search engine."""

    def __init__(
        self,
        engine: VectorSearchEngine,
        target_config: str = "100k-768-m32-efc200-ef100-ip",
    ):
        """Initialize benchmark runner.

        Args:
            engine: Vector search engine instance
            target_config: Dataset configuration name
        """
        self.engine = engine
        self.target_config = target_config
        self.results: dict[str, Any] = {}
        self.section_values: list[str] = []

        self._docker_manager = DockerManager(engine.engine_config.container_name)
        self._embedding_loader = EmbeddingLoader(engine.dataset_config)
        self._content_loader = ContentLoader(engine.dataset_config)
        self._metrics_calculator = MetricsCalculator()

    def setup(self) -> None:
        """Set up the benchmark environment."""
        print(f"=== Setting up {self.engine.engine_name} benchmark ===")

        # Clean up Docker
        DockerManager.prune()

        # Print engine info
        print(f"<<< {self.engine.engine_name} {self.engine.engine_config.version} >>>")

        # Start container
        docker_cmd = self.engine.get_docker_command()
        if docker_cmd:
            self._docker_manager.run(docker_cmd)
        else:
            # For engines using docker-compose (like Milvus)
            if hasattr(self.engine, "generate_compose_file"):
                compose_file = self.engine.generate_compose_file()
                self._docker_manager.run_compose(compose_file)

        # Wait for engine to be ready
        if not self.engine.wait_until_ready():
            raise RuntimeError(f"Failed to start {self.engine.engine_name}")

        # Print initial stats
        self._print_stats()

    def create_index(self) -> None:
        """Create the search index."""
        print("=== Creating index ===")

        # Create database if needed (e.g., pgvector)
        if hasattr(self.engine, "create_database"):
            self.engine.create_database()

        self.engine.create_index()

        # Wait for index to be ready if needed
        if hasattr(self.engine, "wait_for_index_ready"):
            self.engine.wait_for_index_ready()

        self._print_stats()

    def run_indexing(self) -> dict[str, Any]:
        """Run the indexing benchmark.

        Returns:
            Indexing results dictionary
        """
        print("=== Running indexing ===")
        cfg = self.engine.dataset_config
        start_time = time.time()
        total_process_time = 0.0

        documents: list[dict[str, Any]] = []
        embeddings: list[list[float]] = []
        ids: list[int] = []
        count = 0

        for row, section_vals in self._content_loader.iter_documents(
            max_size=cfg.index_size,
            collect_section_values=True,
        ):
            self.section_values = section_vals

            doc_id, embedding = next(self._embedding_loader.iter_embeddings(
                start_offset=row.id - 1,
                max_size=1,
            ))

            count += 1
            ids.append(count)
            embeddings.append(embedding.tolist())
            documents.append({
                "page_id": int(row.pageid),
                "rev_id": int(row.revid),
                "section": row.section,
            })

            if len(ids) >= cfg.bulk_size:
                t = self.engine.insert_documents(documents, embeddings, ids)
                total_process_time += t
                documents = []
                embeddings = []
                ids = []

        # Insert remaining documents
        if ids:
            t = self.engine.insert_documents(documents, embeddings, ids)
            total_process_time += t

        # Wait for indexing to complete
        self.engine.wait_for_indexing_complete()

        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Execution Time: {int(hours):02d}:{int(minutes):02d}:{seconds:02.2f} ({timedelta(seconds=total_process_time)})")

        indexing_result = {
            "execution_time": execution_time,
            "process_time": total_process_time,
            "container": DockerManager.get_container_stats(),
        }

        self.results["indexing"] = indexing_result
        self._print_stats()

        return indexing_result

    def run_search_benchmark(
        self,
        page_sizes: list[int] | None = None,
        warmup_count: int = 1000,
        query_count: int = 10000,
        with_filter: bool = False,
    ) -> dict[str, BenchmarkMetrics]:
        """Run search benchmarks.

        Args:
            page_sizes: List of result sizes to test (default: [10, 100])
            warmup_count: Number of warmup queries
            query_count: Number of actual queries
            with_filter: Whether to run filtered search

        Returns:
            Dictionary of metrics by page size
        """
        page_sizes = page_sizes or [10, 100]
        cfg = self.engine.dataset_config
        search_results: dict[str, BenchmarkMetrics] = {}

        filter_generator = None
        if with_filter and self.section_values:
            filter_generator = self.engine.get_filter_generator(self.section_values)

        for page_size in page_sizes:
            suffix = "_filtered" if with_filter else ""
            print(f"=== Search benchmark: top_{page_size}{suffix} ===")

            output_filename = self.engine.get_output_filename(f"knn_{page_size}{suffix}")
            truth_filename = self._get_ground_truth_filename(page_size, with_filter)

            # Run warmup
            print(f"Running {warmup_count} warmup queries...")
            self._run_queries(
                output_path=None,
                max_size=warmup_count,
                page_size=page_size,
                offset=0,
                filter_generator=filter_generator,
            )

            # Run actual benchmark
            print(f"Running {query_count} queries...")
            self._run_queries(
                output_path=output_filename,
                max_size=query_count,
                page_size=page_size,
                offset=cfg.index_size,
                filter_generator=filter_generator,
            )

            # Calculate metrics
            metrics = self._metrics_calculator.calculate_from_file(
                output_filename,
                truth_filename,
                k=page_size,
            )

            result_key = f"top_{page_size}{suffix}"
            search_results[result_key] = metrics
            self.results[result_key] = metrics.to_dict()

        if with_filter:
            self.results["num_of_filtered_words"] = len(self.section_values)

        return search_results

    def _run_queries(
        self,
        output_path: str | None,
        max_size: int,
        page_size: int,
        offset: int = 0,
        filter_generator: Generator[dict[str, Any], None, None] | None = None,
    ) -> None:
        """Run search queries.

        Args:
            output_path: Path to write results (None for warmup)
            max_size: Maximum number of queries
            page_size: Number of results per query
            offset: Starting offset in embeddings
            filter_generator: Optional filter query generator
        """
        cfg = self.engine.dataset_config
        start_time = time.time()
        count = 0
        doc_id = 0
        pos = offset
        running = True

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = SearchResultsWriter(output_path)
            writer.__enter__()

        try:
            while running:
                npz_path = cfg.embedding_path / f"{pos}.npz"
                if not npz_path.exists():
                    pos = 0
                    continue

                with np.load(npz_path) as data:
                    embeddings = data["embs"]

                for embedding in embeddings:
                    doc_id += 1
                    if count >= max_size:
                        running = False
                        break

                    embedding = embedding.astype(np.float32)
                    if cfg.distance in ("dot_product", "Dot", "IP", "ip", "<#>", "dotproduct"):
                        embedding = embedding / np.linalg.norm(embedding)

                    filter_query = None
                    if filter_generator:
                        filter_query = next(filter_generator)

                    result = self.engine.search(
                        query_vector=embedding.tolist(),
                        top_k=page_size,
                        filter_query=filter_query,
                    )

                    if result.took_ms == -1:
                        continue

                    if writer:
                        writer.write_result(
                            query_id=doc_id,
                            took=result.took_ms,
                            hits=result.hits,
                            ids=result.ids,
                            scores=result.scores,
                            total_hits=result.total_hits,
                        )

                    count += 1
                    if count % 10000 == 0:
                        print(f"Sent {count}/{max_size} queries.")

                pos += 100000
                if pos > cfg.num_of_docs:
                    pos = 0

        finally:
            if writer:
                writer.__exit__(None, None, None)

        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Execution Time: {int(hours):02d}:{int(minutes):02d}:{seconds:02.2f}")

    def _get_ground_truth_filename(self, page_size: int, with_filter: bool) -> str:
        """Get ground truth filename for precision calculation."""
        base_config = re.sub(r"-m.*", "", self.target_config)
        suffix = "_filtered" if with_filter else ""
        return f"dataset/ground_truth/{base_config}/knn_{page_size}{suffix}.jsonl.gz"

    def save_results(self, output_path: str = "results.json") -> None:
        """Save benchmark results to JSON file.

        Args:
            output_path: Path to write results
        """
        writer = ResultsWriter(output_path)
        writer.save(
            target_config=self.target_config,
            version=self.engine.engine_config.version,
            settings=self._get_settings_dict(),
            results=self.results,
            variant=os.getenv("PRODUCT_VARIANT", ""),
        )

    def _get_settings_dict(self) -> dict[str, Any]:
        """Get combined settings dictionary."""
        cfg = self.engine.dataset_config
        eng_cfg = self.engine.engine_config

        settings = asdict(cfg)
        settings["content_path"] = str(cfg.content_path)
        settings["embedding_path"] = str(cfg.embedding_path)
        settings.update(eng_cfg.to_dict())

        return settings

    def _print_stats(self) -> None:
        """Print Docker and index statistics."""
        print(DockerManager.get_system_df())
        DockerManager.get_container_stats()
        info = self.engine.get_index_info()
        if info:
            print(f"Index info: {info}")

    def cleanup(self) -> None:
        """Clean up after benchmark."""
        print("=== Cleaning up ===")
        self.engine.delete_index()

        docker_cmd = self.engine.get_docker_command()
        if docker_cmd:
            self._docker_manager.stop()
        else:
            if hasattr(self.engine, "milvus_config"):
                compose_file = str(self.engine.milvus_config.compose_yaml_path)
                self._docker_manager.stop_compose(compose_file)

    def run_full_benchmark(self) -> dict[str, Any]:
        """Run the complete benchmark pipeline.

        Returns:
            Complete results dictionary
        """
        try:
            self.setup()
            self.create_index()
            self.run_indexing()
            self.run_search_benchmark(page_sizes=[10, 100])

            # Run filtered search if section values are available
            if self.section_values:
                self.run_search_benchmark(page_sizes=[10, 100], with_filter=True)

            self._print_stats()
            self.save_results()

            return self.results

        finally:
            self.cleanup()


def run_benchmark(
    engine_name: str,
    target_config: str = "100k-768-m32-efc200-ef100-ip",
    **engine_kwargs: Any,
) -> dict[str, Any]:
    """Run benchmark for a specific engine.

    Args:
        engine_name: Name of the engine to benchmark
        target_config: Dataset configuration name
        **engine_kwargs: Engine-specific configuration options

    Returns:
        Benchmark results dictionary
    """
    from search_ann_benchmark.engines import get_engine_class

    # Get dataset config
    dataset_config = get_dataset_config(target_config)

    # Get engine class and create instance
    engine_class = get_engine_class(engine_name)

    # Create engine config if kwargs provided
    engine_config = None
    if engine_kwargs:
        config_class = engine_class.__init__.__annotations__.get("engine_config")
        if config_class:
            engine_config = config_class(**engine_kwargs)

    engine = engine_class(dataset_config, engine_config)

    # Run benchmark
    runner = BenchmarkRunner(engine, target_config)
    return runner.run_full_benchmark()
