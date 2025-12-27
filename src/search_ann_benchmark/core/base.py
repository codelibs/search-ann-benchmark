"""Abstract base class for vector search engines."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Generator, TypeVar

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("base")

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay on each retry
        retryable_exceptions: Tuple of exception types that trigger retry

    Returns:
        Decorated function that retries on failure

    Example:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def insert_batch(documents):
            return requests.post(url, json=documents)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {type(e).__name__}: {e}. Waiting {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries} retries failed for {func.__name__}: {e}"
                        )

            # This should not be reached, but for type safety
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry_with_backoff")

        return wrapper
    return decorator


@dataclass
class SearchResult:
    """Result of a single search query."""

    query_id: int
    took_ms: float
    hits: int
    ids: list[int]
    scores: list[float]
    total_hits: int = 0


@dataclass
class IndexingResult:
    """Result of indexing operation."""

    execution_time: float
    process_time: float
    docs_indexed: int = 0
    container_stats: dict[str, Any] = field(default_factory=dict)


class VectorSearchEngine(ABC):
    """Abstract base class for vector search engines."""

    # Engine identifier used in results and file names
    engine_name: str = ""

    def __init__(self, dataset_config: DatasetConfig, engine_config: EngineConfig):
        """Initialize engine with configuration.

        Args:
            dataset_config: Dataset and benchmark settings
            engine_config: Engine-specific settings
        """
        self.dataset_config = dataset_config
        self.engine_config = engine_config

    @property
    def base_url(self) -> str:
        """Get base URL for the engine API."""
        return f"http://{self.engine_config.host}:{self.engine_config.port}"

    @abstractmethod
    def get_docker_command(self) -> list[str]:
        """Get Docker command to start the engine container.

        Returns:
            List of command arguments for docker run
        """
        pass

    @abstractmethod
    def wait_until_ready(self, timeout: int = 60) -> bool:
        """Wait until the engine is ready to accept requests.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if engine is ready, False if timeout
        """
        pass

    @abstractmethod
    def create_index(self) -> None:
        """Create the search index with configured settings."""
        pass

    @abstractmethod
    def delete_index(self) -> None:
        """Delete the search index."""
        pass

    @abstractmethod
    def get_index_info(self) -> dict[str, Any]:
        """Get information about the current index.

        Returns:
            Dictionary with index metadata (e.g., num_of_docs, index_size)
        """
        pass

    @abstractmethod
    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        """Insert documents with embeddings into the index.

        Args:
            documents: List of document payloads
            embeddings: List of embedding vectors
            ids: List of document IDs

        Returns:
            Time taken in seconds
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Execute a vector search query.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_query: Optional filter conditions

        Returns:
            SearchResult with query results and timing
        """
        pass

    def wait_for_indexing_complete(self, check_interval: float = 1.0, stable_count: int = 30) -> None:
        """Wait for indexing to complete (engine-specific implementation).

        Default implementation checks index status repeatedly until stable.

        Args:
            check_interval: Seconds between status checks
            stable_count: Number of consecutive stable checks required
        """
        logger.debug(f"Waiting for indexing to complete (stable_count={stable_count}, interval={check_interval}s)")
        start = time.time()
        count = 0
        total_checks = 0
        while count < stable_count:
            total_checks += 1
            is_complete = self._is_indexing_complete()
            if is_complete:
                count += 1
            else:
                count = 0
            elapsed = time.time() - start
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Indexing check {total_checks}: complete={is_complete}, stable_count={count}/{stable_count}, elapsed={elapsed:.1f}s")
            else:
                print(".", end="", flush=True)
            time.sleep(check_interval)
        elapsed = time.time() - start
        if not logger.isEnabledFor(logging.DEBUG):
            print(".")
        logger.debug(f"Indexing complete after {total_checks} checks in {elapsed:.1f}s")

    def _is_indexing_complete(self) -> bool:
        """Check if indexing is complete.

        Override in subclass for engine-specific logic.

        Returns:
            True if indexing is complete
        """
        return True

    def get_filter_generator(self, section_values: list[str]) -> Generator[dict[str, Any], None, None]:
        """Generate filter queries for filtered search benchmarks.

        Args:
            section_values: List of section values to filter by

        Yields:
            Filter query dictionaries
        """
        if not section_values:
            return
        while True:
            for section in section_values:
                yield self._build_filter_query(section)

    @abstractmethod
    def _build_filter_query(self, section: str) -> dict[str, Any]:
        """Build a filter query for the given section.

        Args:
            section: Section value to filter by

        Returns:
            Filter query dictionary in engine-specific format
        """
        pass

    def get_output_filename(self, name: str) -> str:
        """Generate output filename for results.

        Args:
            name: Base name for the file (e.g., "knn_10", "knn_100_filtered")

        Returns:
            Full output filename with engine version
        """
        version_str = self.engine_config.version.replace(".", "_")
        filename = f"output/{self.engine_name}{version_str}_{name}"
        if self.dataset_config.exact:
            filename += "_exact"
        filename += ".jsonl.gz"
        return filename

    def normalize_distance(self, distance: str) -> str:
        """Convert distance metric name to engine-specific format.

        Args:
            distance: Standard distance name (e.g., "dot_product", "cosine")

        Returns:
            Engine-specific distance name
        """
        return distance
