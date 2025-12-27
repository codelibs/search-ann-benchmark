"""Milvus vector search engine implementation."""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("engines.milvus")


@dataclass
class MilvusConfig(EngineConfig):
    """Milvus-specific configuration."""

    name: str = "milvus"
    host: str = "localhost"
    port: int = 19540
    version: str = "2.5.4"
    container_name: str = "benchmark_milvus"
    etcd_version: str = "3.5.5"
    minio_version: str = "RELEASE.2023-03-20T20-16-18Z"
    compose_yaml_path: Path = Path("milvus-compose.yaml")


class MilvusEngine(VectorSearchEngine):
    """Milvus vector database implementation."""

    engine_name = "milvus"

    def __init__(self, dataset_config: DatasetConfig, engine_config: MilvusConfig | None = None):
        engine_config = engine_config or MilvusConfig()
        super().__init__(dataset_config, engine_config)

    @property
    def milvus_config(self) -> MilvusConfig:
        return self.engine_config  # type: ignore

    def get_docker_command(self) -> list[str]:
        # Milvus uses docker-compose, return empty list
        # The actual start is handled by generate_compose_file and run_compose
        return []

    def generate_compose_file(self) -> str:
        """Generate docker-compose.yaml for Milvus."""
        cfg = self.milvus_config
        volume_dir = os.getenv("VOLUME_DIR", "./data")
        use_volume = "#" if os.getenv("VOLUME_DIR") is None else ""

        compose_yaml = f"""
services:
  etcd:
    container_name: {cfg.container_name}-etcd
    image: quay.io/coreos/etcd:v{cfg.etcd_version}
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
{use_volume}    volumes:
{use_volume}      - {volume_dir}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: {cfg.container_name}-minio
    image: minio/minio:{cfg.minio_version}
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
{use_volume}    volumes:
{use_volume}      - {volume_dir}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: {cfg.container_name}-standalone
    image: milvusdb/milvus:v{cfg.version}
    command: ["milvus", "run", "standalone"]
    security_opt:
    - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
{use_volume}    volumes:
{use_volume}      - {volume_dir}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "{cfg.port}:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: {cfg.container_name}
"""
        with open(cfg.compose_yaml_path, "w", encoding="utf-8") as f:
            f.write(compose_yaml)

        return str(cfg.compose_yaml_path)

    def wait_until_ready(self, timeout: int = 120) -> bool:
        logger.info(f"Waiting for {self.engine_config.container_name}...")
        start = time.time()
        for attempt in range(timeout):
            elapsed = time.time() - start
            try:
                logger.debug(f"Health check attempt {attempt+1}/{timeout}, elapsed={elapsed:.1f}s")
                response = requests.post(
                    f"{self.base_url}/v1/vector/collections/create",
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                    json={"collectionName": "healthcheck", "dimension": 256},
                    timeout=5,
                )
                obj = response.json()
                logger.debug(f"Health check response: status={response.status_code}, code={obj.get('code')}")
                if response.status_code == 200 and obj.get("code") == 200:
                    logger.info(f"Engine ready after {elapsed:.1f}s [OK]")
                    # Clean up healthcheck collection
                    requests.post(
                        f"{self.base_url}/v1/vector/collections/drop",
                        headers={"Accept": "application/json", "Content-Type": "application/json"},
                        json={"collectionName": "healthcheck"},
                    )
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
        print(f"Creating Collection {cfg.index_name}... ", end="")

        # Create collection via REST API v2
        schema = {
            "collectionName": cfg.index_name,
            "dimension": cfg.dimension,
            "metricType": self.normalize_distance(cfg.distance),
            "primaryField": "id",
            "vectorField": "embedding",
        }

        response = requests.post(
            f"{self.base_url}/v1/vector/collections/create",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json=schema,
        )

        if response.status_code == 200 and response.json().get("code") == 200:
            print("[OK]")
        else:
            print(f"[FAIL] {response.text}")

    def delete_index(self) -> None:
        cfg = self.dataset_config
        print(f"Deleting Collection {cfg.index_name}... ", end="")
        response = requests.post(
            f"{self.base_url}/v1/vector/collections/drop",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json={"collectionName": cfg.index_name},
        )
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def get_index_info(self) -> dict[str, Any]:
        cfg = self.dataset_config
        response = requests.post(
            f"{self.base_url}/v1/vector/query",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json={
                "collectionName": cfg.index_name,
                "outputFields": ["count(*)"],
                "filter": "id > 0",
                "limit": 0,
            },
        )
        obj = response.json()
        count = obj.get("data", [{}])[0].get("count(*)", 0) if obj.get("code") == 200 else 0
        return {"num_of_docs": count}

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        cfg = self.dataset_config
        logger.debug(f"Preparing insert request: {len(ids)} docs")

        # Build document list for batch insert
        docs = [
            {"id": doc_id, "embedding": embedding, **doc}
            for doc, embedding, doc_id in zip(documents, embeddings, ids)
        ]

        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/v1/vector/insert",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json={"collectionName": cfg.index_name, "data": docs},
        )
        elapsed = time.time() - start_time

        result = response.json()
        if result.get("code") == 200:
            insert_count = result.get("data", {}).get("insertCount", len(ids))
            logger.debug(f"Insert completed: {insert_count} docs in {elapsed:.3f}s [OK]")
            return elapsed
        else:
            logger.error(f"Insert failed: code={result.get('code')} message={result.get('message')}")
            return 0

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        cfg = self.dataset_config

        query: dict[str, Any] = {
            "collectionName": cfg.index_name,
            "annsField": "embedding",
            "data": [query_vector],
            "limit": top_k,
            "searchParams": {
                "metricType": self.normalize_distance(cfg.distance),
            },
        }

        if filter_query:
            query["filter"] = filter_query

        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/v2/vectordb/entities/search",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json=query,
        )
        took_ms = (time.time() - start_time) * 1000

        obj = response.json()
        if obj.get("code") == 0:
            results = obj.get("data", [])
            return SearchResult(
                query_id=0,
                took_ms=took_ms,
                hits=len(results),
                ids=[x.get("id") for x in results],
                scores=[x.get("distance") for x in results],
            )

        print(f"[FAIL][{response.status_code}] {response.text}")
        return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

    def _build_filter_query(self, section: str) -> dict[str, Any]:
        # Milvus uses string filter expressions
        return f'section == "{section}"'  # type: ignore

    def normalize_distance(self, distance: str) -> str:
        mapping = {
            "dot_product": "IP",
            "cosine": "COSINE",
        }
        return mapping.get(distance, distance)
