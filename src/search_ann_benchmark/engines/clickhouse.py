"""ClickHouse Vector Search engine implementation."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult, retry_with_backoff
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("engines.clickhouse")


@dataclass
class ClickHouseConfig(EngineConfig):
    """ClickHouse-specific configuration."""

    name: str = "clickhouse"
    host: str = "localhost"
    port: int = 8123
    version: str = "25.8"
    container_name: str = "benchmark_clickhouse"
    database: str = "default"


class ClickHouseEngine(VectorSearchEngine):
    """ClickHouse Vector Search implementation.

    Uses ClickHouse's vector_similarity index with HNSW algorithm
    for approximate nearest neighbor search.
    """

    engine_name = "clickhouse"

    def __init__(self, dataset_config: DatasetConfig, engine_config: ClickHouseConfig | None = None):
        engine_config = engine_config or ClickHouseConfig()
        super().__init__(dataset_config, engine_config)
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        """Get or create HTTP session for connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    def _close_session(self) -> None:
        """Close HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    @property
    def ch_config(self) -> ClickHouseConfig:
        return self.engine_config  # type: ignore

    def get_docker_command(self) -> list[str]:
        cfg = self.ch_config
        return [
            "docker", "run", "-d",
            "--name", cfg.container_name,
            "-p", f"{cfg.port}:8123",
            "-p", "9000:9000",
            "--ulimit", "nofile=262144:262144",
            "-e", "CLICKHOUSE_SKIP_USER_SETUP=1",
            f"clickhouse/clickhouse-server:{cfg.version}",
        ]

    def _execute_query(self, query: str, timeout: int = 60) -> dict[str, Any]:
        """Execute a ClickHouse query via HTTP interface.

        Args:
            query: SQL query to execute
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON response if applicable

        Raises:
            Exception: If query execution fails
        """
        # Add FORMAT JSON for SELECT queries to get structured response
        if query.strip().upper().startswith("SELECT"):
            if "FORMAT" not in query.upper():
                query = query.rstrip().rstrip(";") + " FORMAT JSON"

        response = self._get_session().post(
            f"{self.base_url}/",
            params={"database": self.ch_config.database},
            data=query,
            headers={"Content-Type": "text/plain"},
            timeout=timeout,
        )

        if response.status_code != 200:
            raise Exception(f"ClickHouse query failed: {response.status_code} - {response.text}")

        if response.text.strip():
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {"result": response.text}
        return {}

    def wait_until_ready(self, timeout: int = 60) -> bool:
        logger.info(f"Waiting for {self.engine_config.container_name}...")
        start = time.time()
        for attempt in range(timeout):
            elapsed = time.time() - start
            try:
                logger.debug(f"Health check attempt {attempt+1}/{timeout}, elapsed={elapsed:.1f}s")
                response = requests.get(f"{self.base_url}/ping", timeout=5)
                if response.status_code == 200 and response.text.strip() == "Ok.":
                    logger.info(f"Engine ready after {elapsed:.1f}s [OK]")
                    return True
                logger.debug(f"Health check response: status={response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.debug(f"Health check failed: {type(e).__name__}: {e}")
            if not logger.isEnabledFor(logging.DEBUG):
                print(".", end="", flush=True)
            time.sleep(1)
        if not logger.isEnabledFor(logging.DEBUG):
            print("")
        logger.error(f"Engine not ready after {timeout}s [FAIL]")
        return False

    def create_index(self) -> None:
        cfg = self.dataset_config
        ch_cfg = self.ch_config
        print(f"Creating table {cfg.index_name} with vector_similarity index... ", end="")

        # Map distance metric to ClickHouse function
        distance_function = self.normalize_distance(cfg.distance)

        # Determine quantization type (bf16 is default in ClickHouse)
        quantization = "bf16"
        if cfg.quantization and cfg.quantization != "none":
            quant_map = {
                "int8": "i8",
                "float32": "f32",
                "float16": "f16",
                "bfloat16": "bf16",
            }
            quantization = quant_map.get(cfg.quantization, cfg.quantization)

        # Create table with vector_similarity index
        # ClickHouse vector_similarity index syntax:
        # INDEX name column TYPE vector_similarity('hnsw', 'distance_function', dimensions, quantization, M, ef_construction)
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {cfg.index_name} (
                doc_id UInt64,
                page_id UInt64,
                rev_id UInt64,
                section String,
                title String,
                text String,
                embedding Array(Float32),
                INDEX vec_idx embedding TYPE vector_similarity('hnsw', '{distance_function}', {cfg.dimension}, '{quantization}', {cfg.hnsw_m}, {cfg.hnsw_ef_construction})
            )
            ENGINE = MergeTree()
            ORDER BY doc_id
            SETTINGS index_granularity = 8192
        """

        try:
            self._execute_query(create_table_sql)
            print("[OK]")
        except Exception as e:
            print(f"[FAIL]\n{e}")
            raise

    def delete_index(self) -> None:
        cfg = self.dataset_config
        print(f"Deleting table {cfg.index_name}... ", end="")
        try:
            self._execute_query(f"DROP TABLE IF EXISTS {cfg.index_name}")
            print("[OK]")
        except Exception as e:
            print(f"[FAIL]\n{e}")
        self._close_session()

    def get_index_info(self) -> dict[str, Any]:
        cfg = self.dataset_config
        try:
            result = self._execute_query(f"SELECT count() as cnt FROM {cfg.index_name}")
            count = result.get("data", [{}])[0].get("cnt", 0)
            return {"num_of_docs": int(count)}
        except Exception as e:
            logger.warning(f"Failed to get index info: {e}")
            return {}

    @retry_with_backoff(max_retries=3, initial_delay=1.0, retryable_exceptions=(Exception,))
    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        cfg = self.dataset_config
        print(f"Sending {len(ids)} docs... ", end="")

        # Build INSERT query with JSONEachRow format
        rows = []
        for doc, embedding, doc_id in zip(documents, embeddings, ids):
            row = {
                "doc_id": doc_id,
                "page_id": doc.get("page_id", 0) or 0,
                "rev_id": doc.get("rev_id", 0) or 0,
                "section": doc.get("section", "") or "",
                "title": doc.get("title", "") or "",
                "text": doc.get("text", "") or "",
                "embedding": embedding,
            }
            rows.append(json.dumps(row))

        insert_data = "\n".join(rows)
        insert_sql = f"INSERT INTO {cfg.index_name} FORMAT JSONEachRow"

        start_time = time.time()
        try:
            response = self._get_session().post(
                f"{self.base_url}/",
                params={"database": self.ch_config.database, "query": insert_sql},
                data=insert_data,
                headers={"Content-Type": "application/json"},
                timeout=300,
            )
            if response.status_code != 200:
                raise Exception(f"Insert failed: {response.status_code} - {response.text}")

            elapsed = time.time() - start_time
            print(f"[OK] {elapsed:.3f}s")
            return elapsed
        except Exception as e:
            print(f"[FAIL] {e}")
            raise

    def wait_for_indexing_complete(self, check_interval: float = 1.0, stable_count: int = 10) -> None:
        """Wait for ClickHouse to optimize parts and complete indexing.

        ClickHouse merges parts asynchronously. We wait for the merge to complete
        and for the index to be fully built.
        """
        cfg = self.dataset_config
        logger.info("Waiting for ClickHouse indexing to complete...")

        # Force optimize to merge all parts
        print("Optimizing table (merging parts)... ", end="")
        try:
            self._execute_query(f"OPTIMIZE TABLE {cfg.index_name} FINAL", timeout=600)
            print("[OK]")
        except Exception as e:
            print(f"[WARN] {e}")

        # Wait for parts to stabilize
        super().wait_for_indexing_complete(check_interval=check_interval, stable_count=stable_count)

    def _is_indexing_complete(self) -> bool:
        """Check if all parts have been merged."""
        cfg = self.dataset_config
        try:
            result = self._execute_query(
                f"SELECT count() as part_count FROM system.parts WHERE table = '{cfg.index_name}' AND active = 1"
            )
            part_count = int(result.get("data", [{}])[0].get("part_count", 0))
            # Consider indexing complete when there are only a few active parts (ideally 1)
            return part_count <= 2
        except Exception:
            return False

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        cfg = self.dataset_config

        distance_function = self.normalize_distance(cfg.distance)
        vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

        # Build WHERE clause for filtering
        where_clause = ""
        if filter_query and "section" in filter_query:
            section = filter_query["section"].replace("'", "\\'")
            where_clause = f"WHERE section = '{section}'"

        # Use the same distance function as the index for efficient ANN search
        # For dot_product with normalized vectors, cosineDistance gives equivalent ranking
        # cosineDistance = 1 - cosine_similarity = 1 - dot_product (for unit vectors)
        distance_expr = f"{distance_function}(embedding, {vector_str})"

        # Build SETTINGS clause
        # When using filtered search, ClickHouse uses post-filtering by default, which
        # may return fewer results than requested. To improve precision, we use
        # vector_search_index_fetch_multiplier to fetch more candidates before filtering.
        settings_parts = [f"hnsw_candidate_list_size_for_search = {cfg.hnsw_ef}"]
        if filter_query:
            # Fetch 10x more candidates when filtering to ensure we get enough results
            # after post-filtering is applied
            settings_parts.append("vector_search_index_fetch_multiplier = 10.0")

        settings_clause = ", ".join(settings_parts)

        query = f"""
            SELECT
                doc_id,
                {distance_expr} AS distance
            FROM {cfg.index_name}
            {where_clause}
            ORDER BY distance ASC
            LIMIT {top_k}
            SETTINGS
                {settings_clause}
        """

        start_time = time.time()
        try:
            result = self._execute_query(query, timeout=30)
            took_ms = (time.time() - start_time) * 1000

            data = result.get("data", [])
            ids = [int(row.get("doc_id", 0)) for row in data]
            # Convert distance to score (negate distance so higher is better)
            # For cosineDistance: 0 = identical, 2 = opposite
            # Score = -distance (or 1 - distance for normalized range)
            scores = [-float(row.get("distance", 0)) for row in data]

            return SearchResult(
                query_id=0,
                took_ms=took_ms,
                hits=len(data),
                ids=ids,
                scores=scores,
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

    def _build_filter_query(self, section: str) -> dict[str, Any]:
        return {"section": section}

    def normalize_distance(self, distance: str) -> str:
        """Convert distance metric name to ClickHouse format."""
        mapping = {
            "dot_product": "cosineDistance",  # For HNSW index, use cosine; equivalent for normalized vectors
            "cosine": "cosineDistance",
            "l2": "L2Distance",
            "euclidean": "L2Distance",
        }
        return mapping.get(distance, distance)


# Alias for CLI compatibility (cli.py uses engine.capitalize() + "Config")
ClickhouseConfig = ClickHouseConfig
