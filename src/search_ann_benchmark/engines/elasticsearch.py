"""Elasticsearch vector search engine implementation."""

import json
import time
from dataclasses import dataclass
from typing import Any

import requests

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult


@dataclass
class ElasticsearchConfig(EngineConfig):
    """Elasticsearch-specific configuration."""

    name: str = "elasticsearch"
    host: str = "localhost"
    port: int = 9211
    version: str = "8.17.4"
    container_name: str = "benchmark_es"
    heap: str = "2g"


class ElasticsearchEngine(VectorSearchEngine):
    """Elasticsearch vector database implementation."""

    engine_name = "elasticsearch"

    def __init__(self, dataset_config: DatasetConfig, engine_config: ElasticsearchConfig | None = None):
        engine_config = engine_config or ElasticsearchConfig()
        super().__init__(dataset_config, engine_config)

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
        print(f"Creating {cfg.index_name} with {knn_type}... ", end="")

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
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def delete_index(self) -> None:
        cfg = self.dataset_config
        print(f"Deleting {cfg.index_name}... ", end="")
        response = requests.delete(f"{self.base_url}/{cfg.index_name}")
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

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

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        cfg = self.dataset_config
        print(f"Sending {len(ids)} docs... ", end="")

        bulk_data = []
        for i, (doc, embedding, doc_id) in enumerate(zip(documents, embeddings, ids)):
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

    def forcemerge_index(self) -> None:
        cfg = self.dataset_config
        print(f"Merging {cfg.index_name}... ", end="")
        response = requests.post(f"{self.base_url}/{cfg.index_name}/_forcemerge?max_num_segments=1", timeout=3600)
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

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

    def wait_for_indexing_complete(self, check_interval: float = 1.0, stable_count: int = 30) -> None:
        # Elasticsearch uses flush/close/open instead of waiting
        self.flush_index()
        self.close_index()
        time.sleep(10)
        self.open_index()
        self.refresh_index()
