"""Qdrant vector search engine implementation."""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("engines.qdrant")


@dataclass
class QdrantConfig(EngineConfig):
    """Qdrant-specific configuration."""

    name: str = "qdrant"
    host: str = "localhost"
    port: int = 6344
    version: str = "1.16.2"
    container_name: str = "benchmark_qdrant"


class QdrantEngine(VectorSearchEngine):
    """Qdrant vector database implementation."""

    engine_name = "qdrant"

    def __init__(self, dataset_config: DatasetConfig, engine_config: QdrantConfig | None = None):
        engine_config = engine_config or QdrantConfig()
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

    def get_docker_command(self) -> list[str]:
        return [
            "docker", "run", "-d",
            "--name", self.engine_config.container_name,
            "-p", f"{self.engine_config.port}:6333",
            f"qdrant/qdrant:v{self.engine_config.version}",
        ]

    def wait_until_ready(self, timeout: int = 60) -> bool:
        logger.info(f"Waiting for {self.engine_config.container_name}...")
        start = time.time()
        for attempt in range(timeout):
            elapsed = time.time() - start
            try:
                logger.debug(f"Health check attempt {attempt+1}/{timeout}, elapsed={elapsed:.1f}s")
                response = requests.get(f"{self.base_url}/cluster", timeout=5)
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

    def create_index(self) -> None:
        cfg = self.dataset_config
        print(f"Creating Collection {cfg.index_name} with {cfg.quantization}... ", end="")

        # HNSW config is set at vectors level only (not duplicated at top level)
        schema: dict[str, Any] = {
            "vectors": {
                "size": cfg.dimension,
                "distance": self.normalize_distance(cfg.distance),
                "hnsw_config": {
                    "m": cfg.hnsw_m,
                    "ef_construct": cfg.hnsw_ef_construction,
                },
            },
        }

        if cfg.quantization == "int8":
            schema["quantization_config"] = {
                "scalar": {
                    "type": "int8",
                    "quantile": 0.99,
                    "always_ram": True,
                }
            }

        response = requests.put(
            f"{self.base_url}/collections/{cfg.index_name}",
            headers={"Content-Type": "application/json"},
            json=schema,
        )

        if response.status_code == 200:
            print("[OK]")
        else:
            print(f"[FAIL]\n{response.text}")
            return

        # Create payload indices for filtering
        self._create_payload_indices(cfg.index_name)

    def _create_payload_indices(self, collection_name: str) -> None:
        """Create payload indices for filtering and search."""
        # Integer indices
        for field_name in ["page_id", "rev_id"]:
            logger.debug(f"Creating payload index: integer:{field_name}")
            response = requests.put(
                f"{self.base_url}/collections/{collection_name}/index",
                headers={"Content-Type": "application/json"},
                json={"field_name": field_name, "field_schema": "integer"},
            )
            if response.status_code != 200:
                logger.warning(f"Failed to create index for {field_name}: {response.text}")

        # Keyword indices
        for field_name in ["section"]:
            logger.debug(f"Creating payload index: keyword:{field_name}")
            response = requests.put(
                f"{self.base_url}/collections/{collection_name}/index",
                headers={"Content-Type": "application/json"},
                json={"field_name": field_name, "field_schema": "keyword"},
            )
            if response.status_code != 200:
                logger.warning(f"Failed to create index for {field_name}: {response.text}")

        # Text indices for full-text search
        for field_name in ["title", "text"]:
            logger.debug(f"Creating payload index: text:{field_name}")
            response = requests.put(
                f"{self.base_url}/collections/{collection_name}/index",
                headers={"Content-Type": "application/json"},
                json={
                    "field_name": field_name,
                    "field_schema": {
                        "type": "text",
                        "tokenizer": "word",
                        "min_token_len": 2,
                        "max_token_len": 40,  # Reasonable max token length for words
                        "lowercase": True,
                    },
                },
            )
            if response.status_code != 200:
                logger.warning(f"Failed to create index for {field_name}: {response.text}")

    def delete_index(self) -> None:
        cfg = self.dataset_config
        print(f"Deleting Collection {cfg.index_name}... ", end="")
        response = requests.delete(f"{self.base_url}/collections/{cfg.index_name}", timeout=10)
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")
        self._close_session()

    def get_index_info(self) -> dict[str, Any]:
        cfg = self.dataset_config
        response = requests.get(f"{self.base_url}/collections/{cfg.index_name}")
        obj = response.json()
        return {
            "num_of_docs": obj.get("result", {}).get("points_count", 0),
        }

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        cfg = self.dataset_config
        print(f"Sending {len(ids)} docs... ", end="")

        response = requests.put(
            f"{self.base_url}/collections/{cfg.index_name}/points",
            headers={"Content-Type": "application/json"},
            params={"wait": "true"},
            data=json.dumps({
                "batch": {
                    "ids": ids,
                    "vectors": embeddings,
                    "payloads": documents,
                }
            }),
        )

        if response.status_code == 200:
            t = response.json().get("time", 0)
            print(f"[OK] {t}")
            return t
        else:
            print(f"[FAIL] {response.status_code} {response.text}")
            return 0

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        cfg = self.dataset_config
        query: dict[str, Any] = {
            "vector": query_vector,
            "limit": top_k,
            "params": {
                "hnsw_ef": cfg.hnsw_ef,
                "exact": cfg.exact,
            },
        }

        if filter_query:
            query["filter"] = filter_query

        response = self._get_session().post(
            f"{self.base_url}/collections/{cfg.index_name}/points/search",
            headers={"Content-Type": "application/json"},
            json=query,
            timeout=10,
        )

        if response.status_code == 200:
            obj = response.json()
            if obj.get("status") != "ok":
                print(f"[FAIL] {response.text}")
                return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

            results = obj.get("result", [])
            return SearchResult(
                query_id=0,
                took_ms=obj.get("time", 0) * 1000,
                hits=len(results),
                ids=[x.get("id") for x in results],
                scores=[x.get("score") for x in results],
            )

        print(f"[FAIL][{response.status_code}] {response.text}")
        return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

    def _is_indexing_complete(self) -> bool:
        cfg = self.dataset_config
        response = requests.get(f"{self.base_url}/collections/{cfg.index_name}")
        obj = response.json()
        return obj.get("result", {}).get("status") == "green"

    def _build_filter_query(self, section: str) -> dict[str, Any]:
        return {
            "must": [
                {
                    "key": "section",
                    "match": {"value": section},
                }
            ]
        }

    def normalize_distance(self, distance: str) -> str:
        mapping = {
            "dot_product": "Dot",
            "cosine": "Cosine",
        }
        return mapping.get(distance, distance)
