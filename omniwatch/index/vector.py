"""FAISS Vector Index Storage and Filtered Retrieval.

Implements exact Inner Product (cosine similarity over L2-normalized float32 vectors)
via FAISS IndexFlatIP.
Stores raw embeddings as .npy shards alongside index for complete rebuildability.
Supports SQL pre-filtering by allowed row IDs with exact subset scoring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger("omniwatch.index.vector")


class VectorIndex:
    """FAISS Flat IP vector index with SQLite ID mapping and candidate pre-filtering."""

    def __init__(self, dimension: int = 512, index_name: str = "trk"):
        self.dimension = dimension
        self.index_name = index_name
        self.index = faiss.IndexFlatIP(dimension)
        self.id_map: list[int] = []  # Maps row index -> entity ID (obs_id or tracklet_id)
        self.raw_embeddings: list[np.ndarray] = []

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, ids: list[int]) -> list[int]:
        """Adds L2-normalized embeddings and associates them with database entity IDs.

        Args:
            embeddings: np.ndarray of shape [N, D] (float32).
            ids: List of database entity IDs of length N.

        Returns:
            list[int]: Assigned 0-indexed row numbers in the vector index.
        """
        if embeddings.size == 0 or not ids:
            return []

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )

        embs_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        start_row = self.ntotal

        self.index.add(embs_f32)
        self.id_map.extend(ids)
        self.raw_embeddings.append(embs_f32)

        return list(range(start_row, start_row + len(ids)))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Searches index for top_k closest vectors.

        Supports SQL pre-filtering via allowed_ids; if specified, calculates
        exact cosine similarity exclusively over that candidate subset.

        Returns:
            list[tuple[int, float]]: List of (entity_id, cosine_similarity_score).
        """
        if self.ntotal == 0 or not self.id_map:
            return []

        q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)

        # 1. Unfiltered search via native FAISS
        if allowed_ids is None:
            k = min(top_k, self.ntotal)
            scores, indices = self.index.search(q_vec, k)
            results: list[tuple[int, float]] = []
            for score, row_idx in zip(scores[0], indices[0], strict=False):
                if row_idx >= 0 and row_idx < len(self.id_map):
                    results.append((self.id_map[row_idx], float(score)))
            return results

        # 2. Candidate pre-filtering: exact scoring over allowed_ids subset
        candidate_rows = [
            row_idx
            for row_idx, entity_id in enumerate(self.id_map)
            if entity_id in allowed_ids
        ]

        if not candidate_rows:
            return []

        # Get stacked embeddings
        all_embs = self.get_all_embeddings()
        sub_embs = all_embs[candidate_rows]  # shape [M, D]

        # Exact dot product
        scores_arr = np.dot(sub_embs, q_vec.flatten())  # shape [M,]
        sorted_sub_indices = np.argsort(-scores_arr)[:top_k]

        return [
            (self.id_map[candidate_rows[i]], float(scores_arr[i]))
            for i in sorted_sub_indices
        ]

    def get_all_embeddings(self) -> np.ndarray:
        """Returns single contiguous numpy array of all indexed vectors."""
        if not self.raw_embeddings:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack(self.raw_embeddings)

    def save(self, dir_path: str | Path) -> None:
        """Persists FAISS index, ID mapping, and raw .npy embedding shards."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)

        index_file = path / f"{self.index_name}.index"
        id_map_file = path / f"{self.index_name}_id_map.json"
        shard_file = path / f"{self.index_name}_embs.npy"

        faiss.write_index(self.index, str(index_file))
        with open(id_map_file, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f)

        all_embs = self.get_all_embeddings()
        np.save(str(shard_file), all_embs)
        logger.info("Saved vector index '%s' (%d vectors) to %s", self.index_name, self.ntotal, path)

    def load(self, dir_path: str | Path) -> None:
        """Loads index and metadata from disk."""
        path = Path(dir_path)
        index_file = path / f"{self.index_name}.index"
        id_map_file = path / f"{self.index_name}_id_map.json"
        shard_file = path / f"{self.index_name}_embs.npy"

        if not index_file.exists() or not id_map_file.exists():
            raise FileNotFoundError(f"Vector index files not found in {path}")

        self.index = faiss.read_index(str(index_file))
        with open(id_map_file, encoding="utf-8") as f:
            self.id_map = json.load(f)

        if shard_file.exists():
            embs = np.load(str(shard_file))
            self.raw_embeddings = [embs]
        else:
            self.raw_embeddings = []

        logger.info("Loaded vector index '%s' (%d vectors) from %s", self.index_name, self.ntotal, path)

    def rebuild_from_shards(self, dir_path: str | Path) -> None:
        """Rebuilds FAISS index from stored .npy shards and id_map."""
        path = Path(dir_path)
        id_map_file = path / f"{self.index_name}_id_map.json"
        shard_file = path / f"{self.index_name}_embs.npy"

        if not shard_file.exists() or not id_map_file.exists():
            raise FileNotFoundError(f"Cannot rebuild: missing shards in {path}")

        with open(id_map_file, encoding="utf-8") as f:
            self.id_map = json.load(f)

        embs = np.load(str(shard_file)).astype(np.float32)
        self.raw_embeddings = [embs]

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embs)
        self.save(dir_path)
        logger.info("Rebuilt vector index '%s' from shards (%d vectors)", self.index_name, self.ntotal)


def load_vector_index(
    dir_path: str | Path,
    index_name: str = "trk",
    dimension: int = 512,
) -> VectorIndex:
    """Convenience factory function to load a VectorIndex from disk."""
    vi = VectorIndex(dimension=dimension, index_name=index_name)
    vi.load(dir_path)
    return vi

