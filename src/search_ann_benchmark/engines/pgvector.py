"""pgvector (PostgreSQL) vector search engine implementation."""

import logging
import time
from dataclasses import dataclass
from typing import Any

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("engines.pgvector")


@dataclass
class PgvectorConfig(EngineConfig):
    """pgvector-specific configuration."""

    name: str = "pgvector"
    host: str = "localhost"
    port: int = 5433
    version: str = "0.8.1-pg17"
    container_name: str = "benchmark_pgvector"
    password: str = "ann!test123"
    dbname: str = "vectordb"

    @property
    def conninfo(self) -> str:
        return f"user=postgres password={self.password} host={self.host} port={self.port}"


class PgvectorEngine(VectorSearchEngine):
    """PostgreSQL with pgvector extension implementation."""

    engine_name = "pgvector"

    def __init__(self, dataset_config: DatasetConfig, engine_config: PgvectorConfig | None = None):
        engine_config = engine_config or PgvectorConfig()
        super().__init__(dataset_config, engine_config)
        self._conn: Any = None

    def _get_connection(self) -> Any:
        """Get or create a database connection for search operations."""
        if self._conn is None:
            import psycopg
            from pgvector.psycopg import register_vector

            pg_cfg = self.pg_config
            self._conn = psycopg.connect(f"dbname={pg_cfg.dbname} {pg_cfg.conninfo}")
            register_vector(self._conn)
        return self._conn

    def _close_connection(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def pg_config(self) -> PgvectorConfig:
        return self.engine_config  # type: ignore

    def get_docker_command(self) -> list[str]:
        cfg = self.pg_config
        return [
            "docker", "run", "-d",
            "--name", cfg.container_name,
            "-p", f"{cfg.port}:5432",
            "-e", f"POSTGRES_PASSWORD={cfg.password}",
            f"pgvector/pgvector:{cfg.version}",
        ]

    def wait_until_ready(self, timeout: int = 60) -> bool:
        import psycopg

        logger.info(f"Waiting for {self.engine_config.container_name}...")
        start = time.time()
        for attempt in range(timeout):
            elapsed = time.time() - start
            try:
                logger.debug(f"Health check attempt {attempt+1}/{timeout}, elapsed={elapsed:.1f}s")
                with psycopg.connect(self.pg_config.conninfo) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        if cur.fetchone()[0] == 1:
                            logger.info(f"Engine ready after {elapsed:.1f}s [OK]")
                            return True
            except Exception as e:
                logger.debug(f"Health check failed: {type(e).__name__}: {e}")
            if not logger.isEnabledFor(logging.DEBUG):
                print(".", end="", flush=True)
            time.sleep(1)
        if not logger.isEnabledFor(logging.DEBUG):
            print("")
        logger.error(f"Engine not ready after {timeout}s [FAIL]")
        return False

    def create_database(self) -> None:
        import psycopg

        cfg = self.pg_config
        print(f"Creating {cfg.dbname} database", end="")
        with psycopg.connect(cfg.conninfo) as conn:
            conn.autocommit = True
            conn.execute(f"CREATE DATABASE {cfg.dbname}")
            print(" [OK]")

    def create_index(self) -> None:
        import psycopg

        cfg = self.dataset_config
        pg_cfg = self.pg_config
        vector_type = cfg.quantization if cfg.quantization and cfg.quantization != "none" else "vector"
        print(f"Creating {cfg.index_name} table with {vector_type}... ", end="")

        # Create table without HNSW index (index will be created after data insertion)
        with psycopg.connect(f"dbname={pg_cfg.dbname} {pg_cfg.conninfo}") as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE {cfg.index_name} (
                    doc_id integer PRIMARY KEY,
                    page_id integer,
                    rev_id integer,
                    section text,
                    embedding {vector_type}({cfg.dimension})
                );
            """)
            print(" [OK]")

    def create_hnsw_index(self) -> None:
        """Create HNSW index after all data has been inserted."""
        import psycopg

        cfg = self.dataset_config
        pg_cfg = self.pg_config
        print(f"Creating HNSW index on {cfg.index_name}... ", end="")

        distance = self.normalize_distance(cfg.distance)
        vector_type = cfg.quantization if cfg.quantization and cfg.quantization != "none" else "vector"
        vector_ops = f"{vector_type}_ip_ops" if distance == "<#>" else f"{vector_type}_cosine_ops"

        if cfg.quantization == "halfvec":
            create_idx = f"CREATE INDEX ON {cfg.index_name} USING hnsw ((embedding::{vector_type}({cfg.dimension})) {vector_ops}) WITH (m = {cfg.hnsw_m}, ef_construction = {cfg.hnsw_ef_construction});"
        else:
            create_idx = f"CREATE INDEX ON {cfg.index_name} USING hnsw (embedding {vector_ops}) WITH (m = {cfg.hnsw_m}, ef_construction = {cfg.hnsw_ef_construction});"

        start_time = time.time()
        with psycopg.connect(f"dbname={pg_cfg.dbname} {pg_cfg.conninfo}") as conn:
            conn.execute(create_idx)
        elapsed = time.time() - start_time
        print(f"[OK] {elapsed:.2f}s")

    def wait_for_indexing_complete(self) -> None:
        """Create HNSW index after all documents have been inserted."""
        self.create_hnsw_index()

    def delete_index(self) -> None:
        import psycopg

        # Close search connection first
        self._close_connection()

        cfg = self.dataset_config
        pg_cfg = self.pg_config
        print(f"Deleting {cfg.index_name}... ", end="")
        with psycopg.connect(f"dbname={pg_cfg.dbname} {pg_cfg.conninfo}") as conn:
            conn.execute(f"DROP TABLE {cfg.index_name}")
            print(" [OK]")

    def get_index_info(self) -> dict[str, Any]:
        import psycopg

        cfg = self.dataset_config
        pg_cfg = self.pg_config
        try:
            with psycopg.connect(f"dbname={pg_cfg.dbname} {pg_cfg.conninfo}") as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT count(*) FROM {cfg.index_name}")
                    count = cur.fetchone()[0]
                    return {"num_of_docs": count}
        except Exception as e:
            print(f"count: FAILED {e}")
        return {}

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        import psycopg
        from io import StringIO

        cfg = self.dataset_config
        pg_cfg = self.pg_config
        print(f"Sending {len(ids)} docs... ", end="")

        # Build COPY data in tab-separated format
        buffer = StringIO()
        for doc, embedding, doc_id in zip(documents, embeddings, ids):
            page_id = doc.get("page_id") or "\\N"
            rev_id = doc.get("rev_id") or "\\N"
            section = (doc.get("section") or "\\N").replace("\t", " ").replace("\n", " ")
            # Format embedding as PostgreSQL array literal: [1.0,2.0,3.0]
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
            buffer.write(f"{doc_id}\t{page_id}\t{rev_id}\t{section}\t{embedding_str}\n")
        buffer.seek(0)

        start_time = time.time()
        try:
            with psycopg.connect(f"dbname={pg_cfg.dbname} {pg_cfg.conninfo}") as conn:
                with conn.cursor() as cur:
                    with cur.copy(f"COPY {cfg.index_name} (doc_id, page_id, rev_id, section, embedding) FROM STDIN") as copy:
                        copy.write(buffer.read())
                    conn.commit()
                    elapsed = time.time() - start_time
                    print(f"[OK] {elapsed:.3f}s")
                    return elapsed
        except Exception as e:
            print(f"[FAIL] {e}")
            logger.error(f"COPY failed: {e}")
            return 0

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        cfg = self.dataset_config

        distance = self.normalize_distance(cfg.distance)
        if cfg.quantization == "halfvec":
            column_name = f"embedding::{cfg.quantization}({cfg.dimension})"
        else:
            column_name = "embedding"

        # Cast query vector to appropriate type for pgvector operators
        if cfg.quantization == "halfvec":
            query_cast = f"%s::vector::{cfg.quantization}({cfg.dimension})"
        else:
            query_cast = "%s::vector"

        # Build parameterized query to prevent SQL injection
        params: list[Any] = [query_vector]
        if filter_query and "section" in filter_query:
            where = "WHERE section = %s"
            params.append(filter_query["section"])
        else:
            where = ""

        query = f"""
            SELECT
                doc_id,
                {column_name} {distance} {query_cast} AS distance
            FROM {cfg.index_name}
            {where}
            ORDER BY distance
            LIMIT {top_k};
        """

        start_time = time.time()
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("BEGIN;")
                cur.execute(f"SET LOCAL hnsw.ef_search = {cfg.hnsw_ef};")
                cur.execute(query, tuple(params))
                docs = cur.fetchall()
                took_ms = (time.time() - start_time) * 1000
                cur.execute("COMMIT;")

                return SearchResult(
                    query_id=0,
                    took_ms=took_ms,
                    hits=len(docs),
                    ids=[x[0] for x in docs],
                    scores=[-x[1] for x in docs],  # Negate for inner product
                )
        except Exception as e:
            print(f"[FAIL] {e}")
            return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

    def _build_filter_query(self, section: str) -> dict[str, Any]:
        # Return structured dict for parameterized query (prevents SQL injection)
        return {"section": section}

    def normalize_distance(self, distance: str) -> str:
        mapping = {
            "dot_product": "<#>",
            "cosine": "<=>",
        }
        return mapping.get(distance, distance)
