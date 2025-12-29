"""Elasticsearch vector search engine implementation."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("engines.elasticsearch")


@dataclass
class ElasticsearchConfig(EngineConfig):
    """Elasticsearch-specific configuration."""

    name: str = "elasticsearch"
    host: str = "localhost"
    port: int = 9211
    version: str = "9.2.3"
    container_name: str = "benchmark_es"
    heap: str = "2g"


class ElasticsearchEngine(VectorSearchEngine):
    """Elasticsearch vector database implementation."""

    engine_name = "elasticsearch"

    def __init__(self, dataset_config: DatasetConfig, engine_config: ElasticsearchConfig | None = None):
        engine_config = engine_config or ElasticsearchConfig()
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
    def es_config(self) -> ElasticsearchConfig:
        return self.engine_config  # type: ignore

    def get_docker_command(self) -> list[str]:
        return [
            "docker", "run", "-d",
            "--name", self.engine_config.container_name,
            "-p", f"{self.engine_config.port}:9200",
            "-e", "discovery.type=single-node",
            "-e", "bootstrap.memory_lock=true",
            "-e", "xpack.security.enabled=false",
            "-e", f"ES_JAVA_OPTS=-Xms{self.es_config.heap}",
            f"docker.elastic.co/elasticsearch/elasticsearch:{self.engine_config.version}",
        ]

    def wait_until_ready(self, timeout: int = 60) -> bool:
        logger.info(f"Waiting for {self.engine_config.container_name}...")
        start = time.time()
        for attempt in range(timeout):
            elapsed = time.time() - start
            try:
                logger.debug(f"Health check attempt {attempt+1}/{timeout}, elapsed={elapsed:.1f}s")
                response = requests.get(f"{self.base_url}/", timeout=5)
                if response.status_code == 200:
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

    def _get_knn_type(self) -> str:
        cfg = self.dataset_config
        if cfg.exact:
            return "flat"
        if cfg.quantization == "int8":
            return "int8_hnsw"
        elif cfg.quantization == "int4":
            return "int4_hnsw"
        elif cfg.quantization == "bbq":
            return "bbq_hnsw"
        return "hnsw"

    def create_index(self, number_of_shards: int = 1, number_of_replicas: int = 0) -> None:
        cfg = self.dataset_config
        knn_type = self._get_knn_type()
        logger.info(f"Creating index {cfg.index_name} with {knn_type}...")

        response = requests.put(
            f"{self.base_url}/{cfg.index_name}",
            headers={"Content-Type": "application/json"},
            json={
                "mappings": {
                    "_source": {"excludes": ["embedding"]},
                    "properties": {
                        "page_id": {"type": "integer"},
                        "rev_id": {"type": "integer"},
                        "title": {"type": "text"},
                        "section": {"type": "keyword"},
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": cfg.dimension,
                            "index": True,
                            "similarity": cfg.distance,
                            "index_options": {
                                "type": knn_type,
                                "m": cfg.hnsw_m,
                                "ef_construction": cfg.hnsw_ef_construction,
                            },
                        },
                    },
                },
                "settings": {
                    "index": {
                        "number_of_shards": number_of_shards,
                        "number_of_replicas": number_of_replicas,
                    },
                },
            },
        )
        if response.status_code == 200:
            logger.info(f"Index {cfg.index_name} created [OK]")
        else:
            logger.error(f"Failed to create index: {response.text}")

    def delete_index(self) -> None:
        cfg = self.dataset_config
        logger.info(f"Deleting index {cfg.index_name}...")
        response = requests.delete(f"{self.base_url}/{cfg.index_name}", timeout=10)
        if response.status_code == 200:
            logger.info(f"Index {cfg.index_name} deleted [OK]")
        else:
            logger.error(f"Failed to delete index: {response.text}")
        self._close_session()

    def get_index_info(self) -> dict[str, Any]:
        cfg = self.dataset_config
        response = requests.get(f"{self.base_url}/_cat/indices?v")
        for line in response.text.split("\n"):
            values = line.split()
            if len(values) >= 9 and values[2] == cfg.index_name:
                return {
                    "num_of_docs": values[6],
                    "index_size": values[8],
                }
        return {}

    def _generate_bulk_ndjson(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> str:
        """Generate NDJSON bulk request body efficiently."""
        cfg = self.dataset_config
        lines = []
        for doc, embedding, doc_id in zip(documents, embeddings, ids):
            lines.append(json.dumps({"index": {"_index": cfg.index_name, "_id": doc_id}}))
            lines.append(json.dumps({**doc, "embedding": embedding}))
        return "\n".join(lines) + "\n"

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        logger.debug(f"Preparing bulk request: {len(ids)} docs")

        bulk_body = self._generate_bulk_ndjson(documents, embeddings, ids)

        start = time.time()
        logger.debug(f"POST {self.base_url}/_bulk starting")
        response = requests.post(
            f"{self.base_url}/_bulk",
            headers={"Content-Type": "application/x-ndjson"},
            data=bulk_body,
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            server_took = result.get("took", 0) / 1000

            # Check for partial failures in bulk response
            if result.get("errors", False):
                failed_items = [
                    item for item in result.get("items", [])
                    if item.get("index", {}).get("status", 200) >= 400
                ]
                if failed_items:
                    logger.warning(
                        f"Partial bulk failure: {len(failed_items)}/{len(ids)} docs failed. "
                        f"First error: {failed_items[0]}"
                    )

            logger.debug(f"Bulk insert completed: {len(ids)} docs, server_took={server_took:.3f}s, total={elapsed:.3f}s [OK]")
            return server_took
        else:
            logger.error(f"Bulk insert failed: {response.status_code} {response.text[:500]}")
            return 0

    def flush_index(self) -> None:
        cfg = self.dataset_config
        logger.info(f"Flushing {cfg.index_name}...")
        start = time.time()
        response = requests.post(f"{self.base_url}/{cfg.index_name}/_flush", timeout=600)
        elapsed = time.time() - start
        if response.status_code == 200:
            logger.info(f"Flush completed in {elapsed:.1f}s [OK]")
        else:
            logger.error(f"Flush failed after {elapsed:.1f}s: {response.text}")

    def refresh_index(self) -> None:
        cfg = self.dataset_config
        logger.info(f"Refreshing {cfg.index_name}...")
        start = time.time()
        response = requests.post(f"{self.base_url}/{cfg.index_name}/_refresh", timeout=600)
        elapsed = time.time() - start
        if response.status_code == 200:
            logger.info(f"Refresh completed in {elapsed:.1f}s [OK]")
        else:
            logger.error(f"Refresh failed after {elapsed:.1f}s: {response.text}")

    def close_index(self) -> None:
        cfg = self.dataset_config
        logger.info(f"Closing {cfg.index_name}...")
        start = time.time()
        response = requests.post(f"{self.base_url}/{cfg.index_name}/_close", timeout=600)
        elapsed = time.time() - start
        if response.status_code == 200:
            logger.info(f"Close completed in {elapsed:.1f}s [OK]")
        else:
            logger.error(f"Close failed after {elapsed:.1f}s: {response.text}")

    def open_index(self) -> None:
        cfg = self.dataset_config
        logger.info(f"Opening {cfg.index_name}...")
        start = time.time()
        response = requests.post(f"{self.base_url}/{cfg.index_name}/_open", timeout=600)
        elapsed = time.time() - start
        if response.status_code == 200:
            logger.info(f"Open completed in {elapsed:.1f}s [OK]")
        else:
            logger.error(f"Open failed after {elapsed:.1f}s: {response.text}")

    def forcemerge_index(self) -> None:
        cfg = self.dataset_config
        logger.info(f"Merging {cfg.index_name}...")
        start = time.time()
        response = requests.post(f"{self.base_url}/{cfg.index_name}/_forcemerge?max_num_segments=1", timeout=3600)
        elapsed = time.time() - start
        if response.status_code == 200:
            logger.info(f"Merge completed in {elapsed:.1f}s [OK]")
        else:
            logger.error(f"Merge failed after {elapsed:.1f}s: {response.text}")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        cfg = self.dataset_config
        num_candidates = max(cfg.hnsw_ef, top_k)

        query: dict[str, Any] = {
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "num_candidates": num_candidates,
            }
        }

        if filter_query:
            query["knn"]["filter"] = filter_query

        query_dsl = {
            "query": query,
            "size": top_k,
            "_source": False,
            "sort": [{"_score": "desc"}],
        }

        response = self._get_session().post(
            f"{self.base_url}/{cfg.index_name}/_search?request_cache=false",
            json=query_dsl,
            timeout=10,
        )

        if response.status_code == 200:
            obj = response.json()
            if obj.get("timed_out"):
                print("[TIMEOUT]")
                return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

            hits = obj.get("hits", {}).get("hits", [])
            return SearchResult(
                query_id=0,
                took_ms=obj.get("took", 0),
                hits=len(hits),
                total_hits=obj.get("hits", {}).get("total", {}).get("value", 0),
                ids=[int(x.get("_id")) for x in hits],
                scores=[x.get("_score") for x in hits],
            )

        print(f"[FAIL][{response.status_code}] {response.text}")
        return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

    def _build_filter_query(self, section: str) -> dict[str, Any]:
        return {"term": {"section": section}}

    def _is_index_ready(self) -> bool:
        """Check if index is in ready state (green/yellow status)."""
        cfg = self.dataset_config
        try:
            response = requests.get(
                f"{self.base_url}/_cluster/health/{cfg.index_name}",
                params={"timeout": "1s"},
                timeout=5,
            )
            if response.status_code == 200:
                status = response.json().get("status")
                return status in ("green", "yellow")
        except requests.exceptions.RequestException:
            pass
        return False

    def wait_for_indexing_complete(self, check_interval: float = 1.0, stable_count: int = 30) -> None:
        """Wait for indexing to complete using flush/close/open/refresh cycle."""
        logger.debug("Starting indexing complete sequence (flush/close/open/refresh)")
        self.flush_index()
        self.close_index()

        # Wait for index to be closed (poll instead of fixed sleep)
        logger.debug("Waiting for index to be closed...")
        for _ in range(30):
            time.sleep(1)
            # Index should not be ready when closed
            if not self._is_index_ready():
                break

        self.open_index()

        # Wait for index to be ready after reopening
        logger.debug("Waiting for index to be ready...")
        for _ in range(60):
            if self._is_index_ready():
                break
            time.sleep(1)

        self.refresh_index()
        logger.debug("Indexing complete sequence finished")
