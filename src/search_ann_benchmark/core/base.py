"""Abstract base class for vector search engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator
import time

from search_ann_benchmark.config import DatasetConfig, EngineConfig


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
        count = 0
        while count < stable_count:
            if self._is_indexing_complete():
                count += 1
            else:
                count = 0
            print(".", end="", flush=True)
            time.sleep(check_interval)
        print(".")

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
