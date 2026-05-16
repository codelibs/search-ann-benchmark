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
from search_ann_benchmark.core.base import SearchResult, VectorSearchEngine
from search_ann_benchmark.core.docker import DockerManager
from search_ann_benchmark.core.embedding import ContentLoader, EmbeddingLoader
from search_ann_benchmark.core.logging import get_logger
from search_ann_benchmark.core.measurement import low_noise_measurement
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
        start = time.perf_counter()
        if not self.engine.wait_until_ready():
            logger.error(f"Failed to start {self.engine.engine_name} after {time.perf_counter()-start:.1f}s")
            raise RuntimeError(f"Failed to start {self.engine.engine_name}")
        logger.debug(f"Engine ready after {time.perf_counter()-start:.1f}s")

    def create_index(self) -> None:
        """Create the search index."""
        logger.info("=== Creating index ===")

        # Create database if needed (e.g., pgvector)
        if hasattr(self.engine, "create_database"):
            logger.debug("Creating database...")
            self.engine.create_database()

        logger.debug("Creating index...")
        start = time.perf_counter()
        self.engine.create_index()
        logger.debug(f"Index created in {time.perf_counter()-start:.1f}s")

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

        start_time = time.perf_counter()
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
                batch_start = time.perf_counter()
                logger.debug(f"Inserting batch {batch_num}: {len(ids)} docs (total: {count})")
                t = self.engine.insert_documents(documents, embeddings, ids)
                total_process_time += t
                logger.debug(f"Batch {batch_num} completed in {time.perf_counter()-batch_start:.2f}s (process_time: {t:.2f}s)")
                documents = []
                embeddings = []
                ids = []

        # Insert remaining documents
        if ids:
            batch_num += 1
            batch_start = time.perf_counter()
            logger.debug(f"Inserting final batch {batch_num}: {len(ids)} docs (total: {count})")
            t = self.engine.insert_documents(documents, embeddings, ids)
            total_process_time += t
            logger.debug(f"Final batch completed in {time.perf_counter()-batch_start:.2f}s (process_time: {t:.2f}s)")

        # Wait for indexing to complete
        logger.debug("Calling wait_for_indexing_complete()")
        wait_start = time.perf_counter()
        self.engine.wait_for_indexing_complete()
        logger.debug(f"Indexing complete wait finished in {time.perf_counter()-wait_start:.1f}s")

        execution_time = time.perf_counter() - start_time
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
            warmup_start = time.perf_counter()
            self._run_queries(
                output_path=None,
                max_size=warmup_count,
                page_size=page_size,
                offset=0,
                filter_generator=warmup_filter,
            )
            logger.debug(f"Warmup completed in {time.perf_counter()-warmup_start:.1f}s")

            # Run actual benchmark with fresh filter generator
            actual_filter = None
            if with_filter and self.section_values:
                actual_filter = self.engine.get_filter_generator(self.section_values)
            logger.info(f"Running {query_count} queries...")
            benchmark_start = time.perf_counter()
            self._run_queries(
                output_path=output_filename,
                max_size=query_count,
                page_size=page_size,
                offset=cfg.index_size,
                filter_generator=actual_filter,
            )
            logger.debug(f"Benchmark queries completed in {time.perf_counter()-benchmark_start:.1f}s")

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

    def _iter_query_batches(
        self,
        offset: int,
        normalize: bool,
    ) -> Generator[tuple[list[int], list[list[float]]], None, None]:
        """Yield successive batches of (doc_ids, preprocessed query vectors).

        All vector preprocessing (dtype cast, normalization, list conversion)
        happens here so the measurement loop only has to call ``engine.search``.
        Embedding shards are yielded in their on-disk order, wrapping back to
        ``0.npz`` once ``num_of_docs`` is exhausted. The consumer is expected
        to stop iterating once it has accumulated enough successful queries.

        Args:
            offset: Starting offset in embeddings
            normalize: Whether to L2-normalize each vector

        Yields:
            ``(doc_ids, query_vectors)`` pairs. Each batch corresponds to one
            ``.npz`` shard.
        """
        cfg = self.engine.dataset_config
        doc_id = 0
        pos = offset

        while True:
            npz_path = cfg.embedding_path / f"{pos}.npz"
            if not npz_path.exists():
                if pos == 0:
                    raise FileNotFoundError(
                        f"Embedding file not found: {npz_path}. "
                        "Run 'bash scripts/setup.sh' to download the dataset."
                    )
                pos = 0
                continue

            with np.load(npz_path) as data:
                raw = data["embs"]

            shard_f32 = raw.astype(np.float32, copy=False)
            if normalize:
                norms = np.linalg.norm(shard_f32, axis=1, keepdims=True)
                # Guard against zero vectors that would otherwise produce NaNs.
                norms = np.where(norms == 0, 1.0, norms)
                shard_f32 = shard_f32 / norms

            doc_ids = list(range(doc_id + 1, doc_id + 1 + len(shard_f32)))
            doc_id += len(shard_f32)

            yield doc_ids, shard_f32.tolist()

            pos += 100000
            if pos > cfg.num_of_docs:
                pos = 0

    def _run_queries(
        self,
        output_path: str | None,
        max_size: int,
        page_size: int,
        offset: int = 0,
        filter_generator: Generator[dict[str, Any], None, None] | None = None,
    ) -> None:
        """Run search queries.

        The per-query measurement loop is intentionally minimal: query vector
        preprocessing, filter generation, result buffering, progress logging,
        and disk writes are deliberately performed outside the search call so
        they do not contribute to per-query latency jitter. Each engine still
        measures its own ``took_ms`` internally; the runner just avoids adding
        noise around that measurement.

        Args:
            output_path: Path to write results (None for warmup)
            max_size: Maximum number of queries
            page_size: Number of results per query
            offset: Starting offset in embeddings
            filter_generator: Optional filter query generator
        """
        cfg = self.engine.dataset_config
        normalize = cfg.distance in (
            "dot_product", "Dot", "IP", "ip", "<#>", "dotproduct",
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Preallocate per-batch result buffers so disk writes can be deferred
        # until after the timed window.
        debug_progress = logger.isEnabledFor(logging.DEBUG)
        progress_every = 1000 if debug_progress else 10000
        engine_search = self.engine.search  # local alias to skip attribute lookup

        start_time = time.perf_counter()
        count = 0
        progress_start = start_time

        writer = SearchResultsWriter(output_path) if output_path else None
        try:
            if writer:
                writer.__enter__()

            with low_noise_measurement():
                batch_iter = self._iter_query_batches(offset=offset, normalize=normalize)
                done = False
                for doc_ids, query_vectors in batch_iter:
                    if done:
                        break

                    # Cap the batch so we never run more queries than needed.
                    remaining = max_size - count
                    if remaining <= 0:
                        break
                    if remaining < len(query_vectors):
                        doc_ids = doc_ids[:remaining]
                        query_vectors = query_vectors[:remaining]

                    # Pre-generate filter queries for the whole batch so the
                    # generator's overhead is outside the timed search call.
                    if filter_generator is not None:
                        filter_queries: list[dict[str, Any] | None] = [
                            next(filter_generator) for _ in query_vectors
                        ]
                    else:
                        filter_queries = [None] * len(query_vectors)

                    batch_results: list[tuple[int, SearchResult]] = []
                    for doc_id, query_vector, filter_query in zip(
                        doc_ids, query_vectors, filter_queries,
                    ):
                        # === Hot measurement region: keep this minimal ===
                        result = engine_search(
                            query_vector=query_vector,
                            top_k=page_size,
                            filter_query=filter_query,
                        )
                        # =================================================
                        batch_results.append((doc_id, result))

                    # All bookkeeping (validation, IO, logging) happens out
                    # of the timed window.
                    for doc_id, result in batch_results:
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
                        if count >= max_size:
                            done = True
                        if count % progress_every == 0:
                            now = time.perf_counter()
                            window = now - progress_start
                            qps = progress_every / window if window > 0 else 0.0
                            msg = (
                                f"Query progress: {count}/{max_size} "
                                f"({qps:.1f} qps over last {progress_every}, "
                                f"window={window:.2f}s)"
                            )
                            if debug_progress:
                                logger.debug(msg)
                            else:
                                logger.info(msg)
                            progress_start = now
        finally:
            if writer:
                writer.__exit__(None, None, None)

        execution_time = time.perf_counter() - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        qps = count / execution_time if execution_time > 0 else 0
        logger.info(
            f"Queries complete: {count} queries in "
            f"{int(hours):02d}:{int(minutes):02d}:{seconds:02.2f} ({qps:.1f} qps)"
        )

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
