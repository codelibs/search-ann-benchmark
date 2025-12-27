"""pgvector (PostgreSQL) vector search engine implementation."""

import time
from dataclasses import dataclass
from typing import Any

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult


@dataclass
class PgvectorConfig(EngineConfig):
    """pgvector-specific configuration."""

    name: str = "pgvector"
    host: str = "localhost"
    port: int = 5433
    version: str = "0.8.0-pg17"
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

        print(f"Waiting for {self.engine_config.container_name}", end="")
        for _ in range(timeout):
            try:
                with psycopg.connect(self.pg_config.conninfo) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        if cur.fetchone()[0] == 1:
                            print(" [OK]")
                            return True
            except Exception:
                pass
            print(".", end="", flush=True)
            time.sleep(1)
        print(" [FAIL]")
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
        print(f"Creating {cfg.index_name} with {cfg.quantization}... ", end="")

        vector_type = cfg.quantization if cfg.quantization else "vector"
        vector_ops = f"{vector_type}_ip_ops" if cfg.distance == "<#>" else f"{vector_type}_cosine_ops"

        if cfg.quantization == "halfvec":
            create_idx = f"CREATE INDEX ON {cfg.index_name} USING hnsw ((embedding::{vector_type}({cfg.dimension})) {vector_ops}) WITH (m = {cfg.hnsw_m}, ef_construction = {cfg.hnsw_ef_construction});"
        else:
            create_idx = f"CREATE INDEX ON {cfg.index_name} USING hnsw (embedding {vector_ops}) WITH (m = {cfg.hnsw_m}, ef_construction = {cfg.hnsw_ef_construction});"

        with psycopg.connect(f"dbname={pg_cfg.dbname} {pg_cfg.conninfo}") as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE {cfg.index_name} (
                    doc_id integer PRIMARY KEY,
                    page_id integer,
                    rev_id integer,
                    section character(128),
                    embedding {vector_type}({cfg.dimension})
                );
                {create_idx}
            """)
            print(" [OK]")

    def delete_index(self) -> None:
        import psycopg

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

        cfg = self.dataset_config
        pg_cfg = self.pg_config
        print(f"Sending {len(ids)} docs... ", end="")

        query = f"INSERT INTO {cfg.index_name} (doc_id, page_id, rev_id, section, embedding) VALUES (%s, %s, %s, %s, %s)"
        docs = []
        for doc, embedding, doc_id in zip(documents, embeddings, ids):
            docs.append((
                doc_id,
                doc.get("page_id"),
                doc.get("rev_id"),
                doc.get("section"),
                embedding,
            ))

        start_time = time.time()
        try:
            with psycopg.connect(f"dbname={pg_cfg.dbname} {pg_cfg.conninfo}") as conn:
                with conn.cursor() as cur:
                    cur.executemany(query, docs)
                    conn.commit()
                    elapsed = time.time() - start_time
                    print(f"[OK] {elapsed}")
                    return elapsed
        except Exception as e:
            print(f"[FAIL] {e}")
            return 0

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        import psycopg
        from pgvector.psycopg import register_vector

        cfg = self.dataset_config
        pg_cfg = self.pg_config

        where = f"WHERE {filter_query}" if filter_query else ""
        if cfg.quantization == "halfvec":
            column_name = f"embedding::{cfg.quantization}({cfg.dimension})"
        else:
            column_name = "embedding"

        query = f"""
            SELECT
                doc_id,
                {column_name} {cfg.distance} %s AS distance
            FROM {cfg.index_name}
            {where}
            ORDER BY distance
            LIMIT {top_k};
        """

        start_time = time.time()
        try:
            with psycopg.connect(f"dbname={pg_cfg.dbname} {pg_cfg.conninfo}") as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute("BEGIN;")
                    cur.execute(f"SET LOCAL hnsw.ef_search = {cfg.hnsw_ef};")
                    cur.execute(query, (query_vector,))
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
        return f"section = '{section}'"  # type: ignore

    def normalize_distance(self, distance: str) -> str:
        mapping = {
            "dot_product": "<#>",
            "cosine": "<=>",
        }
        return mapping.get(distance, distance)
