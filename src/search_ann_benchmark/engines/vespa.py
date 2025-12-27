"""Vespa vector search engine implementation."""

import io
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import requests

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import VectorSearchEngine, SearchResult
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("engines.vespa")


@dataclass
class VespaConfig(EngineConfig):
    """Vespa-specific configuration."""

    name: str = "vespa"
    host: str = "localhost"
    port: int = 8090
    version: str = "8.499.20"
    container_name: str = "benchmark_vespa"
    management_port: int = 19081


class VespaEngine(VectorSearchEngine):
    """Vespa search platform implementation."""

    engine_name = "vespa"

    def __init__(self, dataset_config: DatasetConfig, engine_config: VespaConfig | None = None):
        engine_config = engine_config or VespaConfig()
        super().__init__(dataset_config, engine_config)

    @property
    def vespa_config(self) -> VespaConfig:
        return self.engine_config  # type: ignore

    @property
    def management_url(self) -> str:
        return f"http://{self.engine_config.host}:{self.vespa_config.management_port}"

    def get_docker_command(self) -> list[str]:
        cfg = self.vespa_config
        return [
            "docker", "run", "-d",
            "--name", cfg.container_name,
            "-p", f"{cfg.port}:8080",
            "-p", f"{cfg.management_port}:19071",
            f"vespaengine/vespa:{cfg.version}",
        ]

    def wait_until_ready(self, timeout: int = 60) -> bool:
        logger.info(f"Waiting for {self.engine_config.container_name}...")
        start = time.time()
        for attempt in range(timeout):
            elapsed = time.time() - start
            try:
                logger.debug(f"Health check attempt {attempt+1}/{timeout}, elapsed={elapsed:.1f}s")
                response = requests.get(f"{self.management_url}/state/v1/health", timeout=5)
                if response.status_code == 200:
                    obj = response.json()
                    status = obj.get("status", {}).get("code")
                    logger.debug(f"Health check response: status={status}")
                    if status == "up":
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
        vcfg = self.vespa_config

        float_type = cfg.quantization if cfg.quantization and cfg.quantization != "none" else "float"

        service_xml = f"""<?xml version='1.0' encoding='UTF-8'?>
<services version="1.0" xmlns:deploy="vespa" xmlns:preprocess="properties">
  <container id='default' version='1.0'>
    <search></search>
    <document-api></document-api>
    <nodes>
      <node hostalias='node1'></node>
    </nodes>
  </container>
  <content id='wikipedia' version='1.0'>
    <redundancy>2</redundancy>
    <documents>
      <document type="{cfg.index_name}" mode="index"/>
    </documents>
    <nodes>
      <node hostalias="node1" distribution-key="0" />
    </nodes>
    <tuning>
      <resource-limits>
        <disk>0.95</disk>
        <memory>0.95</memory>
      </resource-limits>
    </tuning>
  </content>
</services>
"""

        sd_str = f"""
schema {cfg.index_name} {{
    document {cfg.index_name} {{
        field page_id type int {{
            indexing: attribute | summary
        }}
        field rev_id type int {{
            indexing: attribute | summary
        }}
        field title type string {{
            indexing: index | summary
            index: enable-bm25
        }}
        field section type string {{
            indexing: attribute | summary
            attribute: fast-search
        }}
        field text type string {{
            indexing: index | summary
            index: enable-bm25
        }}
        field embedding type tensor<{float_type}>(x[{cfg.dimension}]) {{
            indexing: attribute | index
            attribute {{
                distance-metric: {self.normalize_distance(cfg.distance)}
            }}
            index {{
                hnsw {{
                    max-links-per-node: {cfg.hnsw_m}
                    neighbors-to-explore-at-insert: {cfg.hnsw_ef_construction}
                }}
            }}
        }}
    }}

    fieldset default {{
        fields: title,text
    }}

    rank-profile default {{
        first-phase {{
            expression: nativeRank(title, text)
        }}
    }}

    rank-profile closeness {{
        num-threads-per-search: 1
        match-features: distance(field, embedding)

        inputs {{
            query(q)  tensor<{float_type}>(x[{cfg.dimension}])
            query(qa) tensor<{float_type}>(x[{cfg.dimension}])
        }}

        first-phase {{
            expression: closeness(field, embedding)
        }}
    }}
}}
"""

        query_profile = """
<query-profile id="default">
    <field name="presentation.timing">true</field>
</query-profile>
"""

        # Create zip file in memory
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as zip_file:
            zip_file.writestr("services.xml", service_xml)
            zip_file.writestr(f"schemas/{cfg.index_name}.sd", sd_str)
            zip_file.writestr("search/query-profiles/default.xml", query_profile)
        zip_buffer.seek(0)

        print(f"Creating {cfg.index_name}... ", end="")
        response = requests.post(
            f"{self.management_url}/application/v2/tenant/default/prepareandactivate",
            headers={"Content-Type": "application/zip"},
            data=zip_buffer,
        )
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def wait_for_index_ready(self, timeout: int = 60) -> None:
        cfg = self.dataset_config
        print(f"Waiting for index {cfg.index_name}", end="")
        for _ in range(timeout):
            try:
                response = requests.get(
                    f"{self.base_url}/search/",
                    headers={"Content-Type": "application/json"},
                    params={"yql": f"select * from sources * where sddocname contains '{cfg.index_name}';"},
                    timeout=5,
                )
                if response.status_code == 200:
                    print(".")
                    return
            except requests.exceptions.RequestException:
                pass
            print(".", end="", flush=True)
            time.sleep(1)
        print(" [FAIL]")

    def delete_index(self) -> None:
        cfg = self.dataset_config
        print(f"Deleting {cfg.index_name}... ", end="")
        response = requests.delete(f"{self.management_url}/application/v2/tenant/default")
        print("[OK]" if response.status_code == 200 else f"[FAIL]\n{response.text}")

    def get_index_info(self) -> dict[str, Any]:
        cfg = self.dataset_config
        response = requests.get(
            f"{self.base_url}/search/",
            headers={"Content-Type": "application/json"},
            params={"yql": f"select * from sources * where sddocname contains '{cfg.index_name}';"},
        )
        if response.status_code == 200:
            obj = response.json()
            count = obj.get("root", {}).get("fields", {}).get("totalCount", 0)
            return {"num_of_docs": count}
        return {}

    def _check_error_counts(self, output: str) -> bool:
        """Check if vespa feed output contains any errors."""
        regex = r'"http\.response\.error\.count": (\d+)'
        matches = re.findall(regex, output)
        for count in [int(match) for match in matches]:
            if count > 0:
                return False
        return True

    def _has_vespa_cli(self) -> bool:
        """Check if vespa CLI is available."""
        return shutil.which("vespa") is not None

    def _insert_with_vespa_cli(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        """Insert documents using vespa feed CLI command."""
        cfg = self.dataset_config

        # Create JSONL file for vespa feed command
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)
            for doc, embedding, doc_id in zip(documents, embeddings, ids):
                feed_doc = {
                    "put": f"id:{cfg.index_name}:{cfg.index_name}::{doc_id}",
                    "fields": {
                        **doc,
                        "embedding": embedding,
                    },
                }
                f.write(json.dumps(feed_doc))
                f.write("\n")

        try:
            vespa_cmd = [
                "vespa", "feed", str(temp_path),
                "--target", self.base_url,
            ]
            start_time = time.time()
            result = subprocess.run(vespa_cmd, capture_output=True, text=True)
            elapsed = time.time() - start_time

            if result.returncode == 0 and self._check_error_counts(result.stdout):
                print(f"[OK] {elapsed:.2f}s")
            else:
                print(f"[FAIL] returncode={result.returncode}")
                logger.debug(f"STDOUT: {result.stdout}")
                logger.debug(f"STDERR: {result.stderr[:1000] if result.stderr else ''}")
        finally:
            temp_path.unlink(missing_ok=True)

        return elapsed

    def _insert_single_doc(
        self,
        session: requests.Session,
        doc_url: str,
        payload: dict[str, Any],
    ) -> bool:
        """Insert a single document via HTTP API."""
        try:
            response = session.post(
                doc_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _insert_with_http_parallel(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
        max_workers: int = 32,
    ) -> float:
        """Insert documents using parallel HTTP requests.

        Note: Vespa REST API only supports single-document operations.
        For best performance, install vespa CLI: brew install vespa-cli
        """
        cfg = self.dataset_config

        # Prepare requests
        requests_data = []
        for doc, embedding, doc_id in zip(documents, embeddings, ids):
            doc_url = f"{self.base_url}/document/v1/{cfg.index_name}/{cfg.index_name}/docid/{doc_id}"
            payload = {"fields": {**doc, "embedding": {"values": embedding}}}
            requests_data.append((doc_url, payload))

        start_time = time.time()
        success_count = 0
        fail_count = 0

        # Use session for connection reuse
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._insert_single_doc, session, url, payload): (url, payload)
                    for url, payload in requests_data
                }
                for future in as_completed(futures):
                    if future.result():
                        success_count += 1
                    else:
                        fail_count += 1

        elapsed = time.time() - start_time

        if fail_count == 0:
            print(f"[OK] {elapsed:.2f}s")
        else:
            logger.warning(f"Partial bulk failure: {fail_count}/{len(ids)} docs failed")
            print(f"[PARTIAL] {success_count}/{len(ids)} succeeded in {elapsed:.2f}s")

        return elapsed

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        print(f"Sending {len(ids)} docs... ", end="")

        # Prefer vespa CLI if available (much faster)
        if self._has_vespa_cli():
            return self._insert_with_vespa_cli(documents, embeddings, ids)
        else:
            # Fallback to parallel HTTP requests
            logger.warning(
                "vespa CLI not found. Using parallel HTTP requests which is slower. "
                "Install vespa-cli for better performance: brew install vespa-cli"
            )
            return self._insert_with_http_parallel(documents, embeddings, ids)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        cfg = self.dataset_config
        target_hits = max(cfg.hnsw_ef, top_k)
        approximate = "true" if not cfg.exact else "false"

        yql = f"select documentid from {cfg.index_name} where {{approximate:{approximate},targetHits:{target_hits}}}nearestNeighbor(embedding,q)"
        if filter_query:
            yql = f"{yql} and {filter_query}"

        query = {
            "hits": top_k,
            "yql": yql.strip(),
            "ranking": "closeness",
            "input.query(q)": query_vector,
        }

        response = requests.post(
            f"{self.base_url}/search/",
            headers={"Content-Type": "application/json"},
            data=json.dumps(query),
        )

        if response.status_code == 200:
            obj = response.json()
            took_ms = obj.get("timing", {}).get("searchtime", 0) * 1000
            hits = obj.get("root", {}).get("children", []) or []
            return SearchResult(
                query_id=0,
                took_ms=took_ms,
                hits=len(hits),
                total_hits=obj.get("root", {}).get("coverage", {}).get("documents", 0),
                ids=[int(x.get("id").split(":")[-1]) for x in hits],
                scores=[x.get("relevance") for x in hits],
            )

        print(f"[FAIL][{response.status_code}] {response.text}")
        return SearchResult(query_id=0, took_ms=-1, hits=-1, ids=[], scores=[])

    def _build_filter_query(self, section: str) -> dict[str, Any]:
        return f'section contains "{section}"'  # type: ignore

    def normalize_distance(self, distance: str) -> str:
        mapping = {
            "dot_product": "dotproduct",
            "cosine": "angular",
        }
        return mapping.get(distance, distance)
