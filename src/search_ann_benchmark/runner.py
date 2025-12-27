"""Benchmark runner for vector search engines."""

import logging
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
from search_ann_benchmark.core.logging import get_logger
from search_ann_benchmark.core.metrics import (
    BenchmarkMetrics,
    MetricsCalculator,
    ResultsWriter,
    SearchResultsWriter,
)

logger = get_logger("runner")


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
        logger.info(f"=== Setting up {self.engine.engine_name} benchmark ===")

        # Clean up Docker
        logger.debug("Calling DockerManager.prune()")
        DockerManager.prune()

        # Print engine info
        logger.info(f"<<< {self.engine.engine_name} {self.engine.engine_config.version} >>>")

        # Start container
        docker_cmd = self.engine.get_docker_command()
        if docker_cmd:
            logger.debug(f"Docker command: {' '.join(docker_cmd)}")
            self._docker_manager.run(docker_cmd)
        else:
            # For engines using docker-compose (like Milvus)
            if hasattr(self.engine, "generate_compose_file"):
                compose_file = self.engine.generate_compose_file()
                logger.debug(f"Using docker-compose: {compose_file}")
                self._docker_manager.run_compose(compose_file)

        # Wait for engine to be ready
        logger.debug("Calling wait_until_ready()")
        start = time.time()
        if not self.engine.wait_until_ready():
            logger.error(f"Failed to start {self.engine.engine_name} after {time.time()-start:.1f}s")
            raise RuntimeError(f"Failed to start {self.engine.engine_name}")
        logger.debug(f"Engine ready after {time.time()-start:.1f}s")

    def create_index(self) -> None:
        """Create the search index."""
        logger.info("=== Creating index ===")

        # Create database if needed (e.g., pgvector)
        if hasattr(self.engine, "create_database"):
            logger.debug("Creating database...")
            self.engine.create_database()

        logger.debug("Creating index...")
        start = time.time()
        self.engine.create_index()
        logger.debug(f"Index created in {time.time()-start:.1f}s")

        # Wait for index to be ready if needed
        if hasattr(self.engine, "wait_for_index_ready"):
            logger.debug("Waiting for index to be ready...")
            self.engine.wait_for_index_ready()

        self._print_stats()

    def run_indexing(self) -> dict[str, Any]:
        """Run the indexing benchmark.

        Returns:
            Indexing results dictionary
        """
        logger.info("=== Running indexing ===")
        cfg = self.engine.dataset_config

        # Check if dataset exists
        if not cfg.content_path.exists():
            raise FileNotFoundError(
                f"Content path not found: {cfg.content_path}. Run 'bash scripts/setup.sh' to download the dataset."
            )
        if not cfg.embedding_path.exists():
            raise FileNotFoundError(
                f"Embedding path not found: {cfg.embedding_path}. Run 'bash scripts/setup.sh' to download the dataset."
            )

        logger.debug(f"Loading documents from {cfg.content_path}")
        logger.debug(f"Loading embeddings from {cfg.embedding_path}")
        logger.debug(f"Target index size: {cfg.index_size}, bulk size: {cfg.bulk_size}")

        start_time = time.time()
        total_process_time = 0.0

        documents: list[dict[str, Any]] = []
        embeddings: list[list[float]] = []
        ids: list[int] = []
        count = 0
        batch_num = 0

        for row, section_vals in self._content_loader.iter_documents(
            max_size=cfg.index_size,
            collect_section_values=True,
        ):
            self.section_values = section_vals

            embedding = self._embedding_loader.get_embedding(row.id)

            count += 1
            ids.append(count)
            embeddings.append(embedding.tolist())
            documents.append({
                "page_id": int(row.pageid),
                "rev_id": int(row.revid),
                "section": row.section,
            })

            if len(ids) >= cfg.bulk_size:
                batch_num += 1
                batch_start = time.time()
                logger.debug(f"Inserting batch {batch_num}: {len(ids)} docs (total: {count})")
                t = self.engine.insert_documents(documents, embeddings, ids)
                total_process_time += t
                logger.debug(f"Batch {batch_num} completed in {time.time()-batch_start:.2f}s (process_time: {t:.2f}s)")
                documents = []
                embeddings = []
                ids = []

        # Insert remaining documents
        if ids:
            batch_num += 1
            batch_start = time.time()
            logger.debug(f"Inserting final batch {batch_num}: {len(ids)} docs (total: {count})")
            t = self.engine.insert_documents(documents, embeddings, ids)
            total_process_time += t
            logger.debug(f"Final batch completed in {time.time()-batch_start:.2f}s (process_time: {t:.2f}s)")

        # Wait for indexing to complete
        logger.debug("Calling wait_for_indexing_complete()")
        wait_start = time.time()
        self.engine.wait_for_indexing_complete()
        logger.debug(f"Indexing complete wait finished in {time.time()-wait_start:.1f}s")

        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Indexing complete: {count} docs in {int(hours):02d}:{int(minutes):02d}:{seconds:02.2f} (process_time: {timedelta(seconds=total_process_time)})")

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

        for page_size in page_sizes:
            suffix = "_filtered" if with_filter else ""
            logger.info(f"=== Search benchmark: top_{page_size}{suffix} ===")

            output_filename = self.engine.get_output_filename(f"knn_{page_size}{suffix}")
            truth_filename = self._get_ground_truth_filename(page_size, with_filter)

            # Run warmup with fresh filter generator
            warmup_filter = None
            if with_filter and self.section_values:
                warmup_filter = self.engine.get_filter_generator(self.section_values)
            logger.info(f"Running {warmup_count} warmup queries...")
            warmup_start = time.time()
            self._run_queries(
                output_path=None,
                max_size=warmup_count,
                page_size=page_size,
                offset=0,
                filter_generator=warmup_filter,
            )
            logger.debug(f"Warmup completed in {time.time()-warmup_start:.1f}s")

            # Run actual benchmark with fresh filter generator
            actual_filter = None
            if with_filter and self.section_values:
                actual_filter = self.engine.get_filter_generator(self.section_values)
            logger.info(f"Running {query_count} queries...")
            benchmark_start = time.time()
            self._run_queries(
                output_path=output_filename,
                max_size=query_count,
                page_size=page_size,
                offset=cfg.index_size,
                filter_generator=actual_filter,
            )
            logger.debug(f"Benchmark queries completed in {time.time()-benchmark_start:.1f}s")

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
                    if pos == 0:
                        raise FileNotFoundError(f"Embedding file not found: {npz_path}. Run 'bash scripts/setup.sh' to download the dataset.")
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
                    if logger.isEnabledFor(logging.DEBUG):
                        if count % 1000 == 0:
                            elapsed = time.time() - start_time
                            qps = count / elapsed if elapsed > 0 else 0
                            logger.debug(f"Query progress: {count}/{max_size} ({qps:.1f} qps, elapsed={elapsed:.1f}s)")
                    elif count % 10000 == 0:
                        elapsed = time.time() - start_time
                        qps = count / elapsed if elapsed > 0 else 0
                        logger.info(f"Sent {count}/{max_size} queries ({qps:.1f} qps)")

                pos += 100000
                if pos > cfg.num_of_docs:
                    pos = 0

        finally:
            if writer:
                writer.__exit__(None, None, None)

        execution_time = time.time() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        qps = count / execution_time if execution_time > 0 else 0
        logger.info(f"Queries complete: {count} queries in {int(hours):02d}:{int(minutes):02d}:{seconds:02.2f} ({qps:.1f} qps)")

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
        logger.debug("Getting Docker system info...")
        logger.info(DockerManager.get_system_df())
        DockerManager.get_container_stats()
        info = self.engine.get_index_info()
        if info:
            logger.info(f"Index info: {info}")

    def cleanup(self) -> None:
        """Clean up after benchmark."""
        logger.info("=== Cleaning up ===")
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
