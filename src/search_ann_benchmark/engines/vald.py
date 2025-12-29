"""Vald vector search engine implementation."""

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from search_ann_benchmark.config import DatasetConfig, EngineConfig
from search_ann_benchmark.core.base import SearchResult, VectorSearchEngine
from search_ann_benchmark.core.logging import get_logger

logger = get_logger("engines.vald")


@dataclass
class ValdConfig(EngineConfig):
    """Vald-specific configuration."""

    name: str = "vald"
    host: str = "localhost"
    port: int = 8081
    version: str = "1.7.13"
    container_name: str = "benchmark_vald"
    # NGT-specific parameters
    creation_edge_size: int = 20
    search_edge_size: int = 10
    # Auto-indexing settings
    auto_index_duration_limit: str = "1m"
    auto_index_check_duration: str = "30s"
    auto_index_length: int = 100


class ValdEngine(VectorSearchEngine):
    """Vald vector database implementation using NGT."""

    engine_name = "vald"

    def __init__(
        self, dataset_config: DatasetConfig, engine_config: ValdConfig | None = None
    ):
        engine_config = engine_config or ValdConfig()
        super().__init__(dataset_config, engine_config)
        self._channel = None
        self._insert_stub = None
        self._search_stub = None
        self._remove_stub = None
        self._index_stub = None
        self._config_dir: Path | None = None

    @property
    def vald_config(self) -> ValdConfig:
        """Get Vald-specific config."""
        return self.engine_config  # type: ignore

    def _get_config_yaml(self) -> str:
        """Generate Vald agent configuration YAML."""
        cfg = self.dataset_config
        vald_cfg = self.vald_config

        # Map distance metric to NGT distance_type
        distance_type = self.normalize_distance(cfg.distance)

        return f"""---
version: v0.0.0
time_zone: UTC
logging:
  level: info
server_config:
  servers:
    - name: grpc
      host: 0.0.0.0
      port: 8081
      mode: GRPC
      probe_wait_time: 3s
      grpc:
        max_recv_msg_size: 0
        max_send_msg_size: 0
  health_check_servers:
    - name: liveness
      host: 0.0.0.0
      port: 3000
      mode: REST
      probe_wait_time: 3s
      http:
        shutdown_duration: 5s
        handler_timeout: ""
        idle_timeout: ""
        read_header_timeout: ""
        read_timeout: ""
        write_timeout: ""
  startup_strategy:
    - liveness
    - grpc
  shutdown_strategy:
    - grpc
    - liveness
  full_shutdown_duration: 600s
ngt:
  index_path: /var/data
  dimension: {cfg.dimension}
  distance_type: {distance_type}
  object_type: float
  creation_edge_size: {vald_cfg.creation_edge_size}
  search_edge_size: {vald_cfg.search_edge_size}
  bulk_insert_chunk_size: 100
  auto_index_duration_limit: {vald_cfg.auto_index_duration_limit}
  auto_index_check_duration: {vald_cfg.auto_index_check_duration}
  auto_index_length: {vald_cfg.auto_index_length}
  enable_in_memory_mode: true
  default_pool_size: 10000
"""

    def _setup_grpc(self) -> None:
        """Set up gRPC channel and stubs."""
        import grpc
        from vald.v1.agent.core import agent_pb2_grpc
        from vald.v1.vald import insert_pb2_grpc, remove_pb2_grpc, search_pb2_grpc

        if self._channel is None:
            self._channel = grpc.insecure_channel(
                f"{self.vald_config.host}:{self.vald_config.port}"
            )
            self._insert_stub = insert_pb2_grpc.InsertStub(self._channel)
            self._search_stub = search_pb2_grpc.SearchStub(self._channel)
            self._remove_stub = remove_pb2_grpc.RemoveStub(self._channel)
            self._index_stub = agent_pb2_grpc.AgentStub(self._channel)

    def _close_grpc(self) -> None:
        """Close gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._insert_stub = None
            self._search_stub = None
            self._remove_stub = None
            self._index_stub = None

    def get_docker_command(self) -> list[str]:
        # Create temp directory for config
        self._config_dir = Path(tempfile.mkdtemp(prefix="vald_config_"))
        config_path = self._config_dir / "config.yaml"
        config_path.write_text(self._get_config_yaml())
        logger.debug(f"Created Vald config at {config_path}")

        return [
            "docker",
            "run",
            "-d",
            "--name",
            self.vald_config.container_name,
            "-v",
            f"{self._config_dir}:/etc/server",
            "-p",
            f"{self.vald_config.port}:8081",
            "-p",
            "3000:3000",
            f"vdaas/vald-agent-ngt:{self.vald_config.version}",
        ]

    def wait_until_ready(self, timeout: int = 60) -> bool:
        import requests

        logger.info(f"Waiting for {self.vald_config.container_name}...")
        start = time.time()
        for attempt in range(timeout):
            elapsed = time.time() - start
            try:
                logger.debug(
                    f"Health check attempt {attempt+1}/{timeout}, elapsed={elapsed:.1f}s"
                )
                # Check liveness endpoint
                response = requests.get("http://localhost:3000/liveness", timeout=5)
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
        """Initialize gRPC connection. Vald creates index automatically."""
        cfg = self.dataset_config
        print(f"Initializing Vald index for dimension {cfg.dimension}... ", end="")
        self._setup_grpc()
        print("[OK]")

    def delete_index(self) -> None:
        """Close gRPC connection and cleanup."""
        print("Cleaning up Vald... ", end="")
        self._close_grpc()

        # Clean up config directory
        if self._config_dir and self._config_dir.exists():
            import shutil

            shutil.rmtree(self._config_dir)
            self._config_dir = None

        print("[OK]")

    def get_index_info(self) -> dict[str, Any]:
        """Get index information from Vald agent."""
        try:
            self._setup_grpc()
            from vald.v1.payload import payload_pb2

            response = self._index_stub.IndexInfo(payload_pb2.Empty())
            return {
                "num_of_docs": response.stored,
                "uncommitted": response.uncommitted,
            }
        except Exception as e:
            logger.warning(f"Failed to get index info: {e}")
            return {"num_of_docs": 0}

    def insert_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[int],
    ) -> float:
        print(f"Sending {len(ids)} docs... ", end="")

        from vald.v1.payload import payload_pb2

        self._setup_grpc()

        # Build multi-insert request
        requests_list = []
        for doc_id, embedding in zip(ids, embeddings):
            vector = payload_pb2.Object.Vector(
                id=str(doc_id),
                vector=embedding,
            )
            config = payload_pb2.Insert.Config(
                skip_strict_exist_check=True,
            )
            req = payload_pb2.Insert.Request(
                vector=vector,
                config=config,
            )
            requests_list.append(req)

        multi_request = payload_pb2.Insert.MultiRequest(requests=requests_list)

        start_time = time.time()
        try:
            self._insert_stub.MultiInsert(multi_request)
            elapsed = time.time() - start_time
            print(f"[OK] {elapsed:.3f}s")
            return elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[FAIL] {e}")
            logger.error(f"Insert failed: {e}")
            return elapsed

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_query: dict[str, Any] | None = None,
    ) -> SearchResult:
        from vald.v1.payload import payload_pb2

        self._setup_grpc()

        # Build search request
        config = payload_pb2.Search.Config(
            num=top_k,
            radius=-1.0,  # -1 means unlimited
            epsilon=0.1,
            timeout=5000000000,  # 5 seconds in nanoseconds
        )
        request = payload_pb2.Search.Request(
            vector=query_vector,
            config=config,
        )

        start_time = time.time()
        try:
            response = self._search_stub.Search(request)
            elapsed = (time.time() - start_time) * 1000  # Convert to ms

            results = response.results
            return SearchResult(
                query_id=0,
                took_ms=elapsed,
                hits=len(results),
                ids=[int(r.id) for r in results],
                scores=[r.distance for r in results],
            )
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"Search failed: {e}")
            return SearchResult(
                query_id=0, took_ms=elapsed, hits=0, ids=[], scores=[]
            )

    def wait_for_indexing_complete(
        self, check_interval: float = 1.0, stable_count: int = 30
    ) -> None:
        """Wait for Vald to complete indexing by triggering CreateIndex."""
        from vald.v1.payload import payload_pb2

        self._setup_grpc()

        print("Triggering index creation... ", end="")
        try:
            # Trigger index creation
            control_request = payload_pb2.Control.CreateIndexRequest(
                pool_size=10000,
            )
            self._index_stub.CreateIndex(control_request)
            print("[OK]")
        except Exception as e:
            print(f"[WARN] {e}")
            logger.warning(f"CreateIndex call failed (may be expected): {e}")

        # Wait for index to stabilize
        logger.debug(
            f"Waiting for indexing to complete (stable_count={stable_count}, interval={check_interval}s)"
        )
        start = time.time()
        count = 0
        total_checks = 0
        last_stored = 0

        while count < stable_count:
            total_checks += 1
            try:
                response = self._index_stub.IndexInfo(payload_pb2.Empty())
                stored = response.stored
                uncommitted = response.uncommitted

                # Consider complete when stored count is stable and uncommitted is 0
                is_complete = stored > 0 and uncommitted == 0 and stored == last_stored

                if is_complete:
                    count += 1
                else:
                    count = 0
                    last_stored = stored

                elapsed = time.time() - start
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Indexing check {total_checks}: stored={stored}, uncommitted={uncommitted}, stable_count={count}/{stable_count}, elapsed={elapsed:.1f}s"
                    )
                else:
                    print(".", end="", flush=True)
            except Exception as e:
                logger.debug(f"Index check failed: {e}")
                count = 0

            time.sleep(check_interval)

        elapsed = time.time() - start
        if not logger.isEnabledFor(logging.DEBUG):
            print(".")
        logger.debug(f"Indexing complete after {total_checks} checks in {elapsed:.1f}s")

    def _is_indexing_complete(self) -> bool:
        """Check if indexing is complete by checking uncommitted count."""
        try:
            from vald.v1.payload import payload_pb2

            self._setup_grpc()
            response = self._index_stub.IndexInfo(payload_pb2.Empty())
            return response.uncommitted == 0
        except Exception:
            return False

    def _build_filter_query(self, section: str) -> dict[str, Any]:
        """Build filter query for section.

        Note: Vald agent standalone does not support filtering.
        This returns an empty dict and filtering is handled at query time.
        """
        return {"section": section}

    def normalize_distance(self, distance: str) -> str:
        """Convert distance metric name to NGT distance_type."""
        mapping = {
            "dot_product": "innerproduct",
            "cosine": "cosine",
            "l2": "l2",
            "euclidean": "l2",
        }
        return mapping.get(distance, distance)
