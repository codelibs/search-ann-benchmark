"""Docker container management utilities."""

import subprocess
from typing import Any


class DockerManager:
    """Manages Docker containers for search engines."""

    def __init__(self, container_name: str):
        """Initialize Docker manager.

        Args:
            container_name: Name of the container to manage
        """
        self.container_name = container_name

    def run(self, docker_cmd: list[str]) -> bool:
        """Run a Docker container.

        Args:
            docker_cmd: Full docker run command arguments

        Returns:
            True if successful
        """
        print(f"Starting {self.container_name}... ", end="")
        result = subprocess.run(docker_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK]")
            return True
        else:
            print("[FAIL]")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

    def stop(self) -> bool:
        """Stop the container.

        Returns:
            True if successful
        """
        print(f"Stopping {self.container_name}... ", end="")
        result = subprocess.run(
            ["docker", "stop", self.container_name],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("[OK]")
            return True
        else:
            print("[FAIL]")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

    @staticmethod
    def prune() -> bool:
        """Clean up Docker resources.

        Returns:
            True if successful
        """
        print("Cleaning up Docker... ", end="")
        result = subprocess.run(
            ["docker", "system", "prune", "-f"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("[OK]")
            return True
        else:
            print("[FAIL]")
            print(f"STDERR: {result.stderr}")
            return False

    @staticmethod
    def get_system_df() -> str:
        """Get Docker disk usage information.

        Returns:
            Docker system df output
        """
        result = subprocess.run(
            ["docker", "system", "df"],
            capture_output=True,
            text=True,
        )
        return result.stdout if result.returncode == 0 else result.stderr

    @staticmethod
    def get_container_stats() -> dict[str, dict[str, Any]]:
        """Get container resource statistics.

        Returns:
            Dictionary mapping container name to stats
        """
        result = subprocess.run(
            ["docker", "container", "stats", "--no-stream"],
            capture_output=True,
            text=True,
        )
        containers: dict[str, dict[str, Any]] = {}
        if result.returncode == 0:
            print(result.stdout)
            for line in result.stdout.split("\n"):
                if line.startswith("CONTAINER") or len(line) == 0:
                    continue
                values = line.split()
                if len(values) >= 14:
                    containers[values[1]] = {
                        "container_id": values[0],
                        "cpu": values[2],
                        "mem": values[6],
                        "mem_usage": values[3],
                        "mem_limit": values[5],
                        "net_in": values[7],
                        "net_out": values[9],
                        "block_in": values[10],
                        "block_out": values[12],
                        "pids": values[13],
                    }
        else:
            print(result.stderr)
        return containers

    def run_compose(self, compose_file: str, project_name: str | None = None) -> bool:
        """Start services with docker-compose.

        Args:
            compose_file: Path to docker-compose.yml
            project_name: Optional project name

        Returns:
            True if successful
        """
        cmd = ["docker", "compose", "-f", compose_file]
        if project_name:
            cmd.extend(["-p", project_name])
        cmd.extend(["up", "-d"])

        print(f"Starting services from {compose_file}... ", end="")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK]")
            return True
        else:
            print("[FAIL]")
            print(f"STDERR: {result.stderr}")
            return False

    def stop_compose(self, compose_file: str, project_name: str | None = None) -> bool:
        """Stop services with docker-compose.

        Args:
            compose_file: Path to docker-compose.yml
            project_name: Optional project name

        Returns:
            True if successful
        """
        cmd = ["docker", "compose", "-f", compose_file]
        if project_name:
            cmd.extend(["-p", project_name])
        cmd.extend(["down", "-v"])

        print(f"Stopping services from {compose_file}... ", end="")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK]")
            return True
        else:
            print("[FAIL]")
            print(f"STDERR: {result.stderr}")
            return False
