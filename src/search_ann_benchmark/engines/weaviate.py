"""Weaviate vector search engine implementation."""

import json
import time
from dataclasses import dataclass
from typing import Any

import requests

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult


@dataclass
class WeaviateConfig(EngineConfig):
    """Weaviate-specific configuration."""

    name: str = "weaviate"
    host: str = "localhost"
    port: int = 8091
    version: str = "1.28.2"
    container_name: str = "benchmark_weaviate"


class WeaviateEngine(VectorSearchEngine):
    """Weaviate vector database implementation."""

    engine_name = "weaviate"

    def __init__(self, dataset_config: DatasetConfig, engine_config: WeaviateConfig | None = None):
        engine_config = engine_config or WeaviateConfig()
        super().__init__(dataset_config, engine_config)

    def get_docker_command(self) -> list[str]:
        return [
            "docker", "run", "-d",
            "--name", self.engine_config.container_name,
            "-p", f"{self.engine_config.port}:8080",
            "-e", "ASYNC_INDEXING=true",
            f"cr.weaviate.io/semitechnologies/weaviate:{self.engine_config.version}",
        ]

    def wait_until_ready(self, timeout: int = 60) -> bool:
        print(f"Waiting for {self.engine_config.container_name}", end="")
        for _ in range(timeout):
            try:
                response = requests.get(f"{self.base_url}/v1/nodes", timeout=5)
                if response.status_code == 200:
                    obj = response.json()
                    nodes = obj.get("nodes", [])
                    if nodes and nodes[0].get("status") == "HEALTHY":
                        print(" [OK]")
                        return True
            except requests.exceptions.RequestException:
                pass
            print(".", end="", flush=True)
            time.sleep(1)
        print(" [FAIL]")
        return False

    def create_index(self) -> None:
        cfg = self.dataset_config
        print(f"Creating {cfg.index_name} with {cfg.quantization}... ", end="")

        schema: dict[str, Any] = {
            "class": cfg.index_name,
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
        cfg = self.dataset_config
        print(f"Deleting {cfg.index_name}... ", end="")
        response = requests.delete(f"{self.base_url}/v1/schema/{cfg.index_name}")
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def get_index_info(self) -> dict[str, Any]:
        cfg = self.dataset_config
        response = requests.post(
            f"{self.base_url}/v1/graphql",
            headers={"Content-Type": "application/json"},
            json={"query": f"{{ Aggregate {{ {cfg.index_name} {{ meta {{ count }} }} }} }}"},
        )
        if response.status_code == 200:
            obj = response.json()
            count = obj.get("data", {}).get("Aggregate", {}).get(cfg.index_name, [{}])[0].get("meta", {}).get("count", 0)
            return {"num_of_docs": count}
        return {}

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        cfg = self.dataset_config
        print(f"Sending {len(ids)} docs... ", end="")

        objects = []
        for doc, embedding, doc_id in zip(documents, embeddings, ids):
            objects.append({
                "class": cfg.index_name,
                "properties": {"doc_id": doc_id, **doc},
                "vector": embedding,
            })

        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/v1/batch/objects",
            headers={"Content-Type": "application/json"},
            json={"objects": objects},
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            print(f"[OK] {elapsed}")
            return elapsed
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

        where_clause = ""
        if filter_query:
            where_clause = filter_query.get("where", "")

        query = f"""{{
  Get {{
    {cfg.index_name} (
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
        response = requests.post(
            f"{self.base_url}/v1/graphql",
            json={"query": query},
        )
        took_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            obj = response.json()
            results = obj.get("data", {}).get("Get", {}).get(cfg.index_name, [])
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
        return {
            "where": f"""
      where: {{
        operator: Equal
        path: ["section"]
        valueString: "{section}"
      }}
"""
        }

    def normalize_distance(self, distance: str) -> str:
        mapping = {
            "dot_product": "dot",
            "cosine": "cosine",
        }
        return mapping.get(distance, distance)

    def wait_for_indexing_complete(self, check_interval: float = 1.0, stable_count: int = 60) -> None:
        cfg = self.dataset_config
        print(f"Waiting for {cfg.index_name}", end="")
        for _ in range(stable_count):
            try:
                response = requests.get(f"{self.base_url}/v1/schema/{cfg.index_name}/shards")
                if response.status_code == 200:
                    obj = response.json()
                    if obj and len(obj) > 0 and obj[0].get("status") == "READY":
                        print(" [OK]")
                        return
            except requests.exceptions.RequestException:
                pass
            print(".", end="", flush=True)
            time.sleep(check_interval)
        print(" [FAIL]")
