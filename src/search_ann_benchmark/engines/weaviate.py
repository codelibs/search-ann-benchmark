"""Weaviate vector search engine implementation."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("engines.weaviate")


@dataclass
class WeaviateConfig(EngineConfig):
    """Weaviate-specific configuration."""

    name: str = "weaviate"
    host: str = "localhost"
    port: int = 8091
    version: str = "1.35.1"
    container_name: str = "benchmark_weaviate"


class WeaviateEngine(VectorSearchEngine):
    """Weaviate vector database implementation."""

    engine_name = "weaviate"

    def __init__(self, dataset_config: DatasetConfig, engine_config: WeaviateConfig | None = None):
        engine_config = engine_config or WeaviateConfig()
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

    def _get_class_name(self) -> str:
        """Get Weaviate class name (must start with uppercase)."""
        name = self.dataset_config.index_name
        return name[0].upper() + name[1:] if name else "Content"

    def get_docker_command(self) -> list[str]:
        return [
            "docker", "run", "-d",
            "--name", self.engine_config.container_name,
            "-p", f"{self.engine_config.port}:8080",
            "-e", "ASYNC_INDEXING=true",
            f"cr.weaviate.io/semitechnologies/weaviate:{self.engine_config.version}",
        ]

    def wait_until_ready(self, timeout: int = 60) -> bool:
        logger.info(f"Waiting for {self.engine_config.container_name}...")
        start = time.time()
        for attempt in range(timeout):
            elapsed = time.time() - start
            try:
                logger.debug(f"Health check attempt {attempt+1}/{timeout}, elapsed={elapsed:.1f}s")
                response = requests.get(f"{self.base_url}/v1/nodes", timeout=5)
                if response.status_code == 200:
                    obj = response.json()
                    nodes = obj.get("nodes", [])
                    if nodes:
                        status = nodes[0].get("status")
                        logger.debug(f"Health check response: node_status={status}")
                        if status == "HEALTHY":
                            logger.info(f"Engine ready after {elapsed:.1f}s [OK]")
                            return True
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
        class_name = self._get_class_name()
        print(f"Creating {class_name} with {cfg.quantization}... ", end="")

        schema: dict[str, Any] = {
            "class": class_name,
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "distance": self.normalize_distance(cfg.distance),
                "maxConnections": cfg.hnsw_m,
                "ef": cfg.hnsw_ef,
                "efConstruction": cfg.hnsw_ef_construction,
            },
            "properties": [
                {"name": "doc_id", "dataType": ["int"]},
                {"name": "page_id", "dataType": ["int"]},
                {"name": "rev_id", "dataType": ["int"]},
                {"name": "section", "dataType": ["string"], "indexInverted": True},
                {"name": "text", "dataType": ["text"], "indexInverted": True},
                {"name": "title", "dataType": ["text"], "indexInverted": True},
            ],
        }

        if cfg.quantization == "pq":
            schema["vectorIndexConfig"]["pq"] = {
                "enabled": True,
                "trainingLimit": cfg.index_size - 10000,
            }

        response = requests.post(
            f"{self.base_url}/v1/schema",
            headers={"Content-Type": "application/json"},
            json=schema,
        )
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def delete_index(self) -> None:
        class_name = self._get_class_name()
        print(f"Deleting {class_name}... ", end="")
        response = requests.delete(f"{self.base_url}/v1/schema/{class_name}", timeout=10)
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")
        self._close_session()

    def get_index_info(self) -> dict[str, Any]:
        class_name = self._get_class_name()
        response = requests.post(
            f"{self.base_url}/v1/graphql",
            headers={"Content-Type": "application/json"},
            json={"query": f"{{ Aggregate {{ {class_name} {{ meta {{ count }} }} }} }}"},
        )
        if response.status_code == 200:
            obj = response.json()
            count = obj.get("data", {}).get("Aggregate", {}).get(class_name, [{}])[0].get("meta", {}).get("count", 0)
            return {"num_of_docs": count}
        return {}

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        class_name = self._get_class_name()
        logger.debug(f"Preparing batch request: {len(ids)} docs")

        # Build batch objects
        objects = [
            {
                "class": class_name,
                "properties": {"doc_id": doc_id, **doc},
                "vector": embedding,
            }
            for doc, embedding, doc_id in zip(documents, embeddings, ids)
        ]

        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/v1/batch/objects",
            headers={"Content-Type": "application/json"},
            json={"objects": objects},
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            # Check for partial failures in batch response
            result = response.json()
            if isinstance(result, list):
                failed_items = [item for item in result if item.get("result", {}).get("errors")]
                if failed_items:
                    logger.warning(
                        f"Partial batch failure: {len(failed_items)}/{len(ids)} docs failed. "
                        f"First error: {failed_items[0].get('result', {}).get('errors')}"
                    )
            logger.debug(f"Batch insert completed: {len(ids)} docs in {elapsed:.3f}s [OK]")
            return elapsed
        else:
            logger.error(f"Batch insert failed: {response.status_code} {response.text[:500]}")
            return 0

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        class_name = self._get_class_name()

        where_clause = ""
        if filter_query:
            where_clause = filter_query.get("where", "")

        query = f"""{{
  Get {{
    {class_name} (
      limit: {top_k}
      nearVector: {{
        vector: {json.dumps(query_vector)}
      }}
{where_clause}
    ) {{
      doc_id
      section
      _additional {{
        distance
      }}
    }}
  }}
}}"""

        start_time = time.time()
        response = self._get_session().post(
            f"{self.base_url}/v1/graphql",
            json={"query": query},
            timeout=10,
        )
        took_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            obj = response.json()
            results = obj.get("data", {}).get("Get", {}).get(class_name, [])
            return SearchResult(
                query_id=0,
                took_ms=took_ms,
                hits=len(results),
                ids=[x.get("doc_id") for x in results],
                scores=[x.get("_additional", {}).get("distance") for x in results],
            )

        print(f"[FAIL][{response.status_code}] {response.text}")
        return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

    def _build_filter_query(self, section: str) -> dict[str, Any]:
        # Escape quotes to prevent GraphQL injection
        escaped_section = section.replace("\\", "\\\\").replace('"', '\\"')
        return {
            "where": f"""
      where: {{
        operator: Equal
        path: ["section"]
        valueString: "{escaped_section}"
      }}
"""
        }

    def normalize_distance(self, distance: str) -> str:
        mapping = {
            "dot_product": "dot",
            "cosine": "cosine",
        }
        return mapping.get(distance, distance)

    def _is_indexing_complete(self) -> bool:
        """Check if Weaviate shard is ready."""
        class_name = self._get_class_name()
        try:
            response = requests.get(f"{self.base_url}/v1/schema/{class_name}/shards", timeout=5)
            if response.status_code == 200:
                obj = response.json()
                if obj and len(obj) > 0:
                    status = obj[0].get("status")
                    logger.debug(f"Shard status: {status}")
                    return status == "READY"
        except requests.exceptions.RequestException as e:
            logger.debug(f"Indexing check failed: {type(e).__name__}: {e}")
        return False
