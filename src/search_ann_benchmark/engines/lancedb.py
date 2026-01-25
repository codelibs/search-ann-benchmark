"""LanceDB vector search engine implementation."""

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("engines.lancedb")


@dataclass
class LancedbConfig(EngineConfig):
    """LanceDB-specific configuration."""

    name: str = "lancedb"
    host: str = "localhost"
    port: int = 0  # Not used for embedded database
    version: str = "0.26.1"
    container_name: str = ""  # Not used for embedded database
    db_path: str = ".lancedb"
    # Fetch multiplier for filtered search - fetch more candidates before filtering
    # to ensure we get enough results after post-filtering is applied
    filter_fetch_multiplier: float = 10.0


class LanceDBEngine(VectorSearchEngine):
    """LanceDB embedded vector database implementation.

    LanceDB is an embedded vector database similar to SQLite.
    It doesn't require a separate server process.
    """

    engine_name = "lancedb"

    def __init__(self, dataset_config: DatasetConfig, engine_config: LancedbConfig | None = None):
        engine_config = engine_config or LancedbConfig()
        super().__init__(dataset_config, engine_config)
        self._db = None
        self._table = None

    @property
    def engine_config(self) -> LancedbConfig:
        return self._engine_config

    @engine_config.setter
    def engine_config(self, value: LancedbConfig) -> None:
        self._engine_config = value

    def _get_db(self) -> Any:
        """Get or create LanceDB database connection."""
        import lancedb

        if self._db is None:
            self._db = lancedb.connect(self.engine_config.db_path)
        return self._db

    def _get_table(self) -> Any:
        """Get the table reference."""
        if self._table is None:
            self._table = self._get_db().open_table(self.dataset_config.index_name)
        return self._table

    def get_docker_command(self) -> list[str]:
        """LanceDB is embedded, no Docker required."""
        return []

    def wait_until_ready(self, timeout: int = 60) -> bool:
        """LanceDB is always ready as an embedded database."""
        logger.info("LanceDB is an embedded database, no startup required [OK]")
        return True

    def create_index(self) -> None:
        """Create the LanceDB table with appropriate schema."""
        cfg = self.dataset_config
        print(f"Creating {cfg.index_name}... ", end="")

        try:
            db = self._get_db()

            # Define schema with PyArrow
            schema = pa.schema([
                pa.field("id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), cfg.dimension)),
                pa.field("page_id", pa.int64()),
                pa.field("rev_id", pa.int64()),
                pa.field("section", pa.utf8()),
            ])

            # Create empty table with schema
            self._table = db.create_table(cfg.index_name, schema=schema)
            print(f"[OK] -> {self._table}")

        except Exception as e:
            print(f"[FAIL] {e}")
            raise

    def _create_vector_index(self) -> None:
        """Create vector index after data insertion for better performance."""
        cfg = self.dataset_config
        table = self._get_table()

        # Determine metric type
        metric = self.normalize_distance(cfg.distance)

        # Determine index type based on quantization
        if cfg.quantization == "pq":
            index_type = "IVF_PQ"
        elif cfg.quantization == "sq" or cfg.quantization == "int8":
            index_type = "IVF_HNSW_SQ"
        else:
            index_type = "IVF_HNSW_SQ"  # Default to HNSW with SQ

        logger.info(f"Creating vector index: type={index_type}, metric={metric}")

        # Calculate num_partitions based on dataset size
        num_partitions = max(1, cfg.index_size // 8192)

        try:
            table.create_index(
                "vector",
                index_type=index_type,
                metric=metric,
                num_partitions=num_partitions,
            )
            logger.info(f"Vector index created [OK]")
        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")

    def delete_index(self) -> None:
        """Delete the LanceDB table."""
        cfg = self.dataset_config
        print(f"Deleting {cfg.index_name}... ", end="")

        try:
            db = self._get_db()
            db.drop_table(cfg.index_name)
            self._table = None
            print("[OK]")
        except Exception as e:
            print(f"[FAIL] {e}")

        # Optionally clean up the database directory
        try:
            db_path = Path(self.engine_config.db_path)
            if db_path.exists():
                shutil.rmtree(db_path)
                logger.debug(f"Cleaned up database directory: {db_path}")
        except Exception as e:
            logger.debug(f"Failed to clean up database directory: {e}")

    def get_index_info(self) -> dict[str, Any]:
        """Get information about the current table."""
        try:
            table = self._get_table()
            return {"num_of_docs": table.count_rows()}
        except Exception as e:
            print(f"[FAIL] {e}")
        return {}

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        """Insert documents with embeddings into the table."""
        logger.debug(f"Preparing insert request: {len(ids)} docs")

        # Prepare data as list of dictionaries
        data = []
        for i, (doc, embedding, doc_id) in enumerate(zip(documents, embeddings, ids)):
            data.append({
                "id": doc_id,
                "vector": embedding,
                "page_id": doc.get("page_id", 0),
                "rev_id": doc.get("rev_id", 0),
                "section": doc.get("section", ""),
            })

        start_time = time.time()
        try:
            self._get_table().add(data)
            elapsed = time.time() - start_time
            logger.debug(f"Insert completed: {len(ids)} docs in {elapsed:.3f}s [OK]")
            print(f"[OK] {elapsed:.3f}s")
            return elapsed
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            print(f"[FAIL] {e}")
            return 0

    def wait_for_indexing_complete(self, check_interval: float = 1.0, stable_count: int = 3) -> None:
        """Create vector index after all data is inserted.

        LanceDB builds indexes after data insertion for optimal performance.
        """
        logger.info("Building vector index...")
        start = time.time()
        self._create_vector_index()
        elapsed = time.time() - start
        logger.info(f"Vector index built in {elapsed:.1f}s")

        # Create scalar index for filtering
        try:
            table = self._get_table()
            table.create_scalar_index("section", index_type="BTREE")
            logger.info("Scalar index on 'section' created [OK]")
        except Exception as e:
            logger.debug(f"Scalar index creation skipped: {e}")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Execute a vector search query."""
        cfg = self.dataset_config

        start_time = time.time()
        try:
            # Build query
            query = self._get_table().search(query_vector)

            # Set distance metric
            query = query.metric(self.normalize_distance(cfg.distance))

            # Apply filter if provided
            # When filtering, fetch more candidates to ensure we get enough results
            # after post-filtering is applied (similar to ClickHouse's fetch multiplier)
            fetch_limit = top_k
            if filter_query:
                where_clause = filter_query.get("where", "")
                if where_clause:
                    query = query.where(where_clause)
                    # Apply fetch multiplier for filtered search
                    fetch_limit = int(top_k * self.engine_config.filter_fetch_multiplier)

            # Set search parameters
            # nprobes controls how many IVF partitions to search
            nprobes = max(1, min(50, cfg.hnsw_ef // 2))
            query = query.nprobes(nprobes)

            # Execute search with potentially higher limit for filtered queries
            results = query.limit(fetch_limit).to_list()

            # Trim results to requested top_k
            results = results[:top_k]
            took_ms = (time.time() - start_time) * 1000

            # Extract results
            ids_list = [int(r.get("id", 0)) for r in results]
            # LanceDB returns _distance field (lower is better for most metrics)
            distances = [float(r.get("_distance", 0)) for r in results]

            # Convert distances to scores (similarity)
            # For L2/Euclidean: score = 1 / (1 + distance)
            # For cosine: score = 1 - distance (cosine distance ranges 0-2)
            # For dot product: score = -distance (negated distance)
            distance_type = cfg.distance
            if distance_type in ("cosine", "Cosine"):
                scores = [1 - d for d in distances]
            elif distance_type in ("dot_product", "Dot", "dot", "ip", "IP"):
                scores = [-d for d in distances]
            else:  # L2/Euclidean
                scores = [1 / (1 + d) for d in distances]

            return SearchResult(
                query_id=0,
                took_ms=took_ms,
                hits=len(ids_list),
                ids=ids_list,
                scores=scores,
            )

        except Exception as e:
            print(f"[FAIL] {e}")
            return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

    def _build_filter_query(self, section: str) -> dict[str, Any]:
        """Build a filter query for the given section.

        LanceDB uses SQL-like WHERE clauses for filtering.
        """
        # Escape single quotes in the section string
        escaped_section = section.replace("'", "''")
        return {"where": f"section = '{escaped_section}'"}

    def normalize_distance(self, distance: str) -> str:
        """Convert distance metric name to LanceDB format.

        LanceDB supports: L2, cosine, dot
        """
        mapping = {
            "dot_product": "dot",
            "Dot": "dot",
            "IP": "dot",
            "ip": "dot",
            "cosine": "cosine",
            "Cosine": "cosine",
            "euclidean": "L2",
            "l2": "L2",
            "L2": "L2",
        }
        return mapping.get(distance, "L2")
