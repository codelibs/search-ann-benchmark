"""OpenSearch vector search engine implementation."""

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult


@dataclass
class OpenSearchConfig(EngineConfig):
    """OpenSearch-specific configuration."""

    name: str = "opensearch"
    host: str = "localhost"
    port: int = 9212
    version: str = "2.19.1"
    container_name: str = "benchmark_opensearch"
    heap: str = "2g"
    engine: str = "lucene"  # or "faiss"


class OpenSearchEngine(VectorSearchEngine):
    """OpenSearch vector database implementation."""

    engine_name = "opensearch"

    def __init__(self, dataset_config: DatasetConfig, engine_config: OpenSearchConfig | None = None):
        engine_config = engine_config or OpenSearchConfig()
        # Override engine from environment
        engine_config.engine = os.getenv("SETTING_ENGINE", engine_config.engine)
        super().__init__(dataset_config, engine_config)

    @property
    def os_config(self) -> OpenSearchConfig:
        return self.engine_config  # type: ignore

    def get_docker_command(self) -> list[str]:
        return [
            "docker", "run", "-d",
            "--name", self.engine_config.container_name,
            "--ulimit", "memlock=-1:-1",
            "--ulimit", "nofile=65535:65535",
            "-p", f"{self.engine_config.port}:9200",
            "-e", "discovery.type=single-node",
            "-e", "bootstrap.memory_lock=true",
            "-e", "plugins.security.disabled=true",
            "-e", f"OPENSEARCH_JAVA_OPTS=-Xms{self.os_config.heap} -Xmx{self.os_config.heap}",
            "-e", "OPENSEARCH_INITIAL_ADMIN_PASSWORD=0LX4wquYDZu6jsve",
            f"opensearchproject/opensearch:{self.engine_config.version}",
        ]

    def wait_until_ready(self, timeout: int = 60) -> bool:
        print(f"Waiting for {self.engine_config.container_name}", end="")
        for _ in range(timeout):
            try:
                response = requests.get(f"{self.base_url}/", timeout=5)
                if response.status_code == 200:
                    print("[OK]")
                    return True
            except requests.exceptions.RequestException:
                pass
            print(".", end="", flush=True)
            time.sleep(1)
        print("[FAIL]")
        return False

    def create_index(self, number_of_shards: int = 1, number_of_replicas: int = 0) -> None:
        cfg = self.dataset_config
        data_type = "byte" if cfg.quantization == "byte" else "float"
        print(f"Creating {cfg.index_name} with {data_type}... ", end="")

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
                            "type": "knn_vector",
                            "dimension": cfg.dimension,
                            "data_type": data_type,
                            "method": {
                                "name": "hnsw",
                                "space_type": self.normalize_distance(cfg.distance),
                                "engine": self.os_config.engine,
                                "parameters": {
                                    "ef_construction": cfg.hnsw_ef_construction,
                                    "m": cfg.hnsw_m,
                                },
                            },
                        },
                    },
                },
                "settings": {
                    "index": {
                        "number_of_shards": number_of_shards,
                        "number_of_replicas": number_of_replicas,
                        "knn": True,
                        "knn.algo_param.ef_search": cfg.hnsw_ef,
                    },
                },
            },
        )
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def delete_index(self) -> None:
        cfg = self.dataset_config
        print(f"Deleting {cfg.index_name}... ", end="")
        response = requests.delete(f"{self.base_url}/{cfg.index_name}")
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def get_index_info(self) -> dict[str, Any]:
        cfg = self.dataset_config
        response = requests.get(f"{self.base_url}/_cat/indices")
        for line in response.text.split("\n"):
            values = line.split()
            if len(values) >= 9 and values[2] == cfg.index_name:
                return {
                    "num_of_docs": values[6],
                    "index_size": values[8],
                }
        return {}

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        cfg = self.dataset_config
        print(f"Sending {len(ids)} docs... ", end="")

        bulk_data = []
        for doc, embedding, doc_id in zip(documents, embeddings, ids):
            bulk_data.append(json.dumps({"index": {"_index": cfg.index_name, "_id": doc_id}}))
            doc_with_embedding = {**doc, "embedding": embedding}
            bulk_data.append(json.dumps(doc_with_embedding))

        response = requests.post(
            f"{self.base_url}/_bulk",
            headers={"Content-Type": "application/x-ndjson"},
            data="\n".join(bulk_data) + "\n",
        )

        if response.status_code == 200:
            t = response.json().get("took", 0) / 1000
            print(f"[OK] {t}")
            return t
        else:
            print(f"[FAIL] {response.status_code} {response.text}")
            return 0

    def flush_index(self) -> None:
        cfg = self.dataset_config
        print(f"Flushing {cfg.index_name}... ", end="")
        response = requests.post(f"{self.base_url}/{cfg.index_name}/_flush", timeout=600)
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def refresh_index(self) -> None:
        cfg = self.dataset_config
        print(f"Refreshing {cfg.index_name}... ", end="")
        response = requests.post(f"{self.base_url}/{cfg.index_name}/_refresh", timeout=600)
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def close_index(self) -> None:
        cfg = self.dataset_config
        print(f"Closing {cfg.index_name}... ", end="")
        response = requests.post(f"{self.base_url}/{cfg.index_name}/_close", timeout=600)
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def open_index(self) -> None:
        cfg = self.dataset_config
        print(f"Opening {cfg.index_name}... ", end="")
        response = requests.post(f"{self.base_url}/{cfg.index_name}/_open", timeout=600)
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        cfg = self.dataset_config

        query: dict[str, Any] = {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k,
                }
            }
        }

        if filter_query:
            query["knn"]["embedding"]["filter"] = filter_query

        query_dsl = {
            "query": query,
            "size": top_k,
            "_source": False,
            "sort": [{"_score": "desc"}],
        }

        response = requests.post(
            f"{self.base_url}/{cfg.index_name}/_search?request_cache=false",
            json=query_dsl,
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

    def normalize_distance(self, distance: str) -> str:
        mapping = {
            "dot_product": "innerproduct",
            "cosine": "cosinesimil",
        }
        return mapping.get(distance, distance)

    def wait_for_indexing_complete(self, check_interval: float = 1.0, stable_count: int = 30) -> None:
        self.flush_index()
        self.close_index()
        time.sleep(10)
        self.open_index()
        self.refresh_index()
