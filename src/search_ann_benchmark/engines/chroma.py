"""Chroma vector search engine implementation."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("engines.chroma")


@dataclass
class ChromaConfig(EngineConfig):
    """Chroma-specific configuration."""

    name: str = "chroma"
    host: str = "localhost"
    port: int = 8008
    version: str = "1.4.0"
    container_name: str = "benchmark_chroma"


class ChromaEngine(VectorSearchEngine):
    """Chroma embedding database implementation."""

    engine_name = "chroma"

    def __init__(self, dataset_config: DatasetConfig, engine_config: ChromaConfig | None = None):
        engine_config = engine_config or ChromaConfig()
        super().__init__(dataset_config, engine_config)
        self._client = None
        self._collection = None

    def get_docker_command(self) -> list[str]:
        cfg = self.engine_config
        return [
            "docker", "run", "-d",
            "--name", cfg.container_name,
            "-p", f"{cfg.port}:8000",
            "-e", "IS_PERSISTENT=TRUE",
            "-e", "PERSIST_DIRECTORY=/chroma/chroma",
            f"ghcr.io/chroma-core/chroma:{cfg.version}",
        ]

    def wait_until_ready(self, timeout: int = 60) -> bool:
        logger.info(f"Waiting for {self.engine_config.container_name}...")
        start = time.time()
        for attempt in range(timeout):
            elapsed = time.time() - start
            try:
                logger.debug(f"Health check attempt {attempt+1}/{timeout}, elapsed={elapsed:.1f}s")
                response = requests.get(f"{self.base_url}/api/v1/heartbeat", timeout=5)
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

    def _get_client(self) -> Any:
        import chromadb

        if self._client is None:
            self._client = chromadb.HttpClient(
                host=self.engine_config.host,
                port=self.engine_config.port,
            )
        return self._client

    def _get_collection(self) -> Any:
        if self._collection is None:
            self._collection = self._get_client().get_collection(name=self.dataset_config.index_name)
        return self._collection

    def create_index(self) -> None:
        cfg = self.dataset_config
        print(f"Creating {cfg.index_name}... ", end="")
        try:
            collection = self._get_client().create_collection(
                name=cfg.index_name,
                metadata={
                    "hnsw:space": self.normalize_distance(cfg.distance),
                    "hnsw:construction_ef": cfg.hnsw_ef_construction,
                    "hnsw:search_ef": cfg.hnsw_ef,
                    "hnsw:M": cfg.hnsw_m,
                },
            )
            self._collection = collection
            print(f"[OK] -> {collection}")
        except Exception as e:
            print(f"[FAIL] {e}")

    def delete_index(self) -> None:
        cfg = self.dataset_config
        print(f"Deleting {cfg.index_name}... ", end="")
        try:
            self._get_client().delete_collection(name=cfg.index_name)
            self._collection = None
            print("[OK]")
        except Exception as e:
            print(f"[FAIL] {e}")

    def get_index_info(self) -> dict[str, Any]:
        try:
            collection = self._get_collection()
            return {"num_of_docs": collection.count()}
        except Exception as e:
            print(f"[FAIL] {e}")
        return {}

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        logger.debug(f"Preparing insert request: {len(ids)} docs")

        # Convert integer IDs to strings (Chroma requires string IDs)
        str_ids = list(map(str, ids))

        start_time = time.time()
        try:
            self._get_collection().add(
                embeddings=embeddings,
                metadatas=documents,
                ids=str_ids,
            )
            elapsed = time.time() - start_time
            logger.debug(f"Insert completed: {len(ids)} docs in {elapsed:.3f}s [OK]")
            return elapsed
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            return 0

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        query: dict[str, Any] = {
            "query_embeddings": query_vector,
            "n_results": top_k,
            "include": ["distances", "metadatas"],
        }

        if filter_query:
            query["where"] = filter_query

        start_time = time.time()
        try:
            result = self._get_collection().query(**query)
            took_ms = (time.time() - start_time) * 1000

            ids_list = result.get("ids", [[]])[0]
            distances = result.get("distances", [[]])[0]

            return SearchResult(
                query_id=0,
                took_ms=took_ms,
                hits=len(ids_list),
                ids=[int(x) for x in ids_list],
                scores=[1 - d for d in distances],  # Convert distance to similarity
            )
        except Exception as e:
            print(f"[FAIL] {e}")
            return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

    def _build_filter_query(self, section: str) -> dict[str, Any]:
        return {"section": {"$eq": section}}

    def normalize_distance(self, distance: str) -> str:
        mapping = {
            "dot_product": "ip",
            "cosine": "cosine",
        }
        return mapping.get(distance, distance)
