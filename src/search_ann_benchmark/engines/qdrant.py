"""Qdrant vector search engine implementation."""

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult


@dataclass
class QdrantConfig(EngineConfig):
    """Qdrant-specific configuration."""

    name: str = "qdrant"
    host: str = "localhost"
    port: int = 6344
    version: str = "1.13.6"
    container_name: str = "benchmark_qdrant"


class QdrantEngine(VectorSearchEngine):
    """Qdrant vector database implementation."""

    engine_name = "qdrant"

    def __init__(self, dataset_config: DatasetConfig, engine_config: QdrantConfig | None = None):
        engine_config = engine_config or QdrantConfig()
        super().__init__(dataset_config, engine_config)

    def get_docker_command(self) -> list[str]:
        return [
            "docker", "run", "-d",
            "--name", self.engine_config.container_name,
            "-p", f"{self.engine_config.port}:6333",
            f"qdrant/qdrant:v{self.engine_config.version}",
        ]

    def wait_until_ready(self, timeout: int = 60) -> bool:
        print(f"Waiting for {self.engine_config.container_name}", end="")
        for _ in range(timeout):
            try:
                response = requests.get(f"{self.base_url}/cluster", timeout=5)
                if response.status_code == 200:
                    print("[OK]")
                    return True
            except requests.exceptions.RequestException:
                pass
            print(".", end="", flush=True)
            time.sleep(1)
        print("[FAIL]")
        return False

    def create_index(self) -> None:
        cfg = self.dataset_config
        print(f"Creating Collection {cfg.index_name} with {cfg.quantization}... ", end="")

        schema: dict[str, Any] = {
            "vectors": {
                "size": cfg.dimension,
                "distance": self.normalize_distance(cfg.distance),
                "hnsw_config": {
                    "m": cfg.hnsw_m,
                    "ef_construct": cfg.hnsw_ef_construction,
                },
            },
            "hnsw_config": {
                "m": cfg.hnsw_m,
                "ef_construct": cfg.hnsw_ef_construction,
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

        # Create payload indices
        for field_name in ["page_id", "rev_id"]:
            print(f"Creating Payload integer:{field_name}... ", end="")
            response = requests.put(
                f"{self.base_url}/collections/{cfg.index_name}/index",
                headers={"Content-Type": "application/json"},
                json={"field_name": field_name, "field_schema": "integer"},
            )
            print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

        for field_name in ["section"]:
            print(f"Creating Payload keyword:{field_name}... ", end="")
            response = requests.put(
                f"{self.base_url}/collections/{cfg.index_name}/index",
                headers={"Content-Type": "application/json"},
                json={"field_name": field_name, "field_schema": "keyword"},
            )
            print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

        for field_name in ["title", "text"]:
            print(f"Creating Payload text:{field_name}... ", end="")
            response = requests.put(
                f"{self.base_url}/collections/{cfg.index_name}/index",
                headers={"Content-Type": "application/json"},
                json={
                    "field_name": field_name,
                    "field_schema": {
                        "type": "text",
                        "tokenizer": "word",
                        "min_token_len": 2,
                        "max_token_len": 2,
                        "lowercase": True,
                    },
                },
            )
            print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def delete_index(self) -> None:
        cfg = self.dataset_config
        print(f"Deleting Collection {cfg.index_name}... ", end="")
        response = requests.delete(f"{self.base_url}/collections/{cfg.index_name}")
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

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

        response = requests.post(
            f"{self.base_url}/collections/{cfg.index_name}/points/search",
            headers={"Content-Type": "application/json"},
            json=query,
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
