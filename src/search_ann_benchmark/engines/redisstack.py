"""Redis Stack vector search engine implementation."""

import logging
import time
from dataclasses import dataclass
from typing import Any

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import SearchResult, VectorSearchEngine
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("engines.redisstack")


@dataclass
class RedisStackConfig(EngineConfig):
    """Redis Stack-specific configuration."""

    name: str = "redisstack"
    host: str = "localhost"
    port: int = 6379
    version: str = "7.4.2-v2"
    container_name: str = "benchmark_redisstack"


class RedisStackEngine(VectorSearchEngine):
    """Redis Stack vector database implementation using RediSearch."""

    engine_name = "redisstack"

    def __init__(self, dataset_config: DatasetConfig, engine_config: RedisStackConfig | None = None):
        engine_config = engine_config or RedisStackConfig()
        super().__init__(dataset_config, engine_config)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create Redis client."""
        if self._client is None:
            import redis
            self._client = redis.Redis(
                host=self.engine_config.host,
                port=self.engine_config.port,
                decode_responses=False,
            )
        return self._client

    def _close_client(self) -> None:
        """Close Redis client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def redis_config(self) -> RedisStackConfig:
        return self.engine_config  # type: ignore

    def get_docker_command(self) -> list[str]:
        cfg = self.redis_config
        return [
            "docker", "run", "-d",
            "--name", cfg.container_name,
            "-p", f"{cfg.port}:6379",
            f"redis/redis-stack-server:{cfg.version}",
        ]

    def wait_until_ready(self, timeout: int = 60) -> bool:
        import redis

        logger.info(f"Waiting for {self.engine_config.container_name}...")
        start = time.time()
        for attempt in range(timeout):
            elapsed = time.time() - start
            try:
                logger.debug(f"Health check attempt {attempt+1}/{timeout}, elapsed={elapsed:.1f}s")
                client = redis.Redis(
                    host=self.engine_config.host,
                    port=self.engine_config.port,
                )
                if client.ping():
                    # Also check that RediSearch module is loaded
                    modules = client.module_list()
                    module_names = [m[b"name"].decode() if isinstance(m[b"name"], bytes) else m[b"name"] for m in modules]
                    if "search" in module_names:
                        logger.info(f"Engine ready after {elapsed:.1f}s [OK]")
                        client.close()
                        return True
                    logger.debug(f"RediSearch module not yet loaded, modules: {module_names}")
                client.close()
            except Exception as e:
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
        print(f"Creating Index {cfg.index_name}... ", end="")

        client = self._get_client()

        # Build FT.CREATE command for vector index
        # Schema: doc_id (TAG), page_id (NUMERIC), rev_id (NUMERIC), section (TAG), embedding (VECTOR)
        distance_type = self.normalize_distance(cfg.distance)

        # HNSW vector field parameters
        vector_params = [
            "TYPE", "FLOAT32",
            "DIM", str(cfg.dimension),
            "DISTANCE_METRIC", distance_type,
            "M", str(cfg.hnsw_m),
            "EF_CONSTRUCTION", str(cfg.hnsw_ef_construction),
        ]

        try:
            # FT.CREATE idx ON HASH PREFIX 1 doc: SCHEMA ...
            client.execute_command(
                "FT.CREATE", cfg.index_name,
                "ON", "HASH",
                "PREFIX", "1", "doc:",
                "SCHEMA",
                "page_id", "NUMERIC", "SORTABLE",
                "rev_id", "NUMERIC", "SORTABLE",
                "section", "TAG",
                "embedding", "VECTOR", "HNSW", str(len(vector_params)), *vector_params,
            )
            print("[OK]")
        except Exception as e:
            print(f"[FAIL] {e}")
            logger.error(f"Failed to create index: {e}")

    def delete_index(self) -> None:
        cfg = self.dataset_config
        print(f"Deleting Index {cfg.index_name}... ", end="")

        try:
            client = self._get_client()
            # FT.DROPINDEX with DD option to also delete the documents
            client.execute_command("FT.DROPINDEX", cfg.index_name, "DD")
            print("[OK]")
        except Exception as e:
            print(f"[FAIL] {e}")
            logger.error(f"Failed to delete index: {e}")
        finally:
            self._close_client()

    def get_index_info(self) -> dict[str, Any]:
        cfg = self.dataset_config
        try:
            client = self._get_client()
            info = client.execute_command("FT.INFO", cfg.index_name)
            # Parse FT.INFO response (alternating key-value pairs)
            info_dict = {}
            for i in range(0, len(info), 2):
                key = info[i].decode() if isinstance(info[i], bytes) else info[i]
                info_dict[key] = info[i + 1]
            num_docs = int(info_dict.get("num_docs", 0))
            return {"num_of_docs": num_docs}
        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            return {}

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        import struct

        print(f"Sending {len(ids)} docs... ", end="")

        client = self._get_client()
        pipeline = client.pipeline(transaction=False)

        start_time = time.time()
        try:
            for doc, embedding, doc_id in zip(documents, embeddings, ids):
                key = f"doc:{doc_id}"
                # Convert embedding to bytes (FLOAT32)
                embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

                fields: dict[str, Any] = {
                    "embedding": embedding_bytes,
                }
                if doc.get("page_id") is not None:
                    fields["page_id"] = doc["page_id"]
                if doc.get("rev_id") is not None:
                    fields["rev_id"] = doc["rev_id"]
                if doc.get("section"):
                    fields["section"] = doc["section"]

                pipeline.hset(key, mapping=fields)

            pipeline.execute()
            elapsed = time.time() - start_time
            print(f"[OK] {elapsed:.3f}s")
            return elapsed
        except Exception as e:
            print(f"[FAIL] {e}")
            logger.error(f"Insert failed: {e}")
            return 0

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        import struct

        cfg = self.dataset_config
        client = self._get_client()

        # Convert query vector to bytes
        query_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)

        # Build query string
        # Base KNN query: *=>[KNN $K @embedding $BLOB EF_RUNTIME $EF]
        if filter_query and "section" in filter_query:
            # Filtered query: @section:{value}=>[KNN ...]
            section = filter_query["section"].replace("-", "\\-").replace(":", "\\:")
            query_str = f"@section:{{{section}}}=>[KNN $K @embedding $BLOB EF_RUNTIME $EF]"
        else:
            query_str = "*=>[KNN $K @embedding $BLOB EF_RUNTIME $EF]"

        start_time = time.time()
        try:
            # FT.SEARCH idx query PARAMS 6 K top_k BLOB bytes EF ef SORTBY __embedding_score
            result = client.execute_command(
                "FT.SEARCH", cfg.index_name, query_str,
                "PARAMS", "6",
                "K", str(top_k),
                "BLOB", query_bytes,
                "EF", str(cfg.hnsw_ef),
                "SORTBY", "__embedding_score",
                "LIMIT", "0", str(top_k),
                "DIALECT", "2",
            )
            took_ms = (time.time() - start_time) * 1000

            # Parse result: [total_hits, doc_key1, [field1, value1, ...], doc_key2, ...]
            total_hits = result[0]
            doc_ids = []
            scores = []

            i = 1
            while i < len(result):
                doc_key = result[i]
                if isinstance(doc_key, bytes):
                    doc_key = doc_key.decode()
                # Extract doc_id from key (format: "doc:123")
                doc_id = int(doc_key.split(":")[1])
                doc_ids.append(doc_id)

                # Parse fields to get score
                if i + 1 < len(result) and isinstance(result[i + 1], list):
                    fields = result[i + 1]
                    for j in range(0, len(fields), 2):
                        field_name = fields[j]
                        if isinstance(field_name, bytes):
                            field_name = field_name.decode()
                        if field_name == "__embedding_score":
                            score = float(fields[j + 1])
                            scores.append(score)
                            break
                    else:
                        scores.append(0.0)
                    i += 2
                else:
                    scores.append(0.0)
                    i += 1

            return SearchResult(
                query_id=0,
                took_ms=took_ms,
                hits=len(doc_ids),
                ids=doc_ids,
                scores=scores,
                total_hits=total_hits,
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

    def _is_indexing_complete(self) -> bool:
        """Check if indexing is complete by checking if index is not being updated."""
        cfg = self.dataset_config
        try:
            client = self._get_client()
            info = client.execute_command("FT.INFO", cfg.index_name)
            # Parse FT.INFO response
            info_dict = {}
            for i in range(0, len(info), 2):
                key = info[i].decode() if isinstance(info[i], bytes) else info[i]
                info_dict[key] = info[i + 1]

            # Check indexing status - when indexing is 0, it's complete
            indexing = int(info_dict.get("indexing", 0))
            return indexing == 0
        except Exception as e:
            logger.debug(f"Index status check failed: {e}")
            return False

    def _build_filter_query(self, section: str) -> dict[str, Any]:
        return {"section": section}

    def normalize_distance(self, distance: str) -> str:
        """Convert distance metric to Redis vector search format."""
        mapping = {
            "dot_product": "IP",  # Inner Product
            "cosine": "COSINE",
        }
        return mapping.get(distance, distance)
