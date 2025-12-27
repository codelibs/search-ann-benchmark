"""Embedding loading and processing utilities."""

from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

from search_ann_benchmark.config import DatasetConfig


class EmbeddingLoader:
    """Loads embeddings and content data for benchmarking."""

    def __init__(self, config: DatasetConfig):
        """Initialize embedding loader.

        Args:
            config: Dataset configuration
        """
        self.config = config
        self._embedding_cache: dict[int, np.ndarray] = {}

    def get_embedding(self, doc_id: int) -> np.ndarray:
        """Get embedding for a document ID.

        Args:
            doc_id: Document ID

        Returns:
            Embedding vector, normalized if using dot product distance
        """
        emb_index = (doc_id // 100000) * 100000

        if emb_index not in self._embedding_cache:
            npz_path = self.config.embedding_path / f"{emb_index}.npz"
            with np.load(npz_path) as data:
                self._embedding_cache[emb_index] = data["embs"]

        embedding = self._embedding_cache[emb_index][doc_id - emb_index].astype(np.float32)

        # Normalize for dot product similarity
        if self.config.distance in ("dot_product", "Dot"):
            embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def iter_embeddings(self, start_offset: int = 0, max_size: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over embeddings starting from offset.

        Args:
            start_offset: Starting embedding index
            max_size: Maximum number of embeddings to yield

        Yields:
            Tuples of (doc_id, embedding)
        """
        pos = start_offset
        count = 0
        max_count = max_size or float("inf")

        while count < max_count:
            npz_path = self.config.embedding_path / f"{pos}.npz"
            if not npz_path.exists():
                pos = 0
                continue

            with np.load(npz_path) as data:
                embeddings = data["embs"]

            for i, embedding in enumerate(embeddings):
                if count >= max_count:
                    return

                doc_id = pos + i + 1
                embedding = embedding.astype(np.float32)

                if self.config.distance in ("dot_product", "Dot"):
                    embedding = embedding / np.linalg.norm(embedding)

                yield doc_id, embedding
                count += 1

            pos += 100000
            if pos > self.config.num_of_docs:
                pos = 0

    def clear_cache(self) -> None:
        """Clear embedding cache to free memory."""
        self._embedding_cache.clear()


class ContentLoader:
    """Loads content data from parquet files."""

    def __init__(self, config: DatasetConfig):
        """Initialize content loader.

        Args:
            config: Dataset configuration
        """
        self.config = config

    def iter_documents(
        self,
        max_size: int | None = None,
        collect_section_values: bool = False,
        min_section_count: int = 10000,
    ) -> Iterator[tuple[pd.Series, list[str]]]:
        """Iterate over documents from parquet files.

        Args:
            max_size: Maximum number of documents to yield
            collect_section_values: Whether to collect section values for filtering
            min_section_count: Minimum count for section values to be collected

        Yields:
            Tuples of (row, section_values) where section_values is updated per file
        """
        count = 0
        max_count = max_size or float("inf")
        section_values: list[str] = []

        for content_file in sorted(self.config.content_path.glob("*.parquet")):
            if count >= max_count:
                break

            df = pd.read_parquet(content_file)

            if collect_section_values:
                file_sections = self._get_section_values(df, min_section_count)
                section_values.extend(file_sections)

            for _, row in df.iterrows():
                if count >= max_count:
                    break
                yield row, section_values
                count += 1

    def _get_section_values(self, df: pd.DataFrame, min_count: int) -> list[str]:
        """Get section values that appear at least min_count times.

        Args:
            df: DataFrame with section column
            min_count: Minimum occurrence count

        Returns:
            List of section values
        """
        section_counts = df[["id", "section"]].groupby("section").count().reset_index()
        section_counts = section_counts[section_counts["id"] >= min_count]
        return section_counts["section"].values.tolist()

    def get_content_files(self) -> list[Path]:
        """Get list of content parquet files.

        Returns:
            Sorted list of parquet file paths
        """
        return sorted(self.config.content_path.glob("*.parquet"))
