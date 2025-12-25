from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SearchResult:
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class SimpleVectorStore:
    """
    Vector store lokal berbasis NumPy:
    - add(): simpan embeddings + text + metadata
    - search(): cosine similarity top-k
    - save()/load(): persist ke folder (npz + json)
    """

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._emb: Optional[np.ndarray] = None         # (N, D)
        self._emb_norm: Optional[np.ndarray] = None    # normalized (N, D)

    @staticmethod
    def _to_2d_float_array(vectors: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Embeddings harus 2D (N, D).")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("Embeddings kosong / dimensi tidak valid.")
        return arr

    @staticmethod
    def _normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return mat / norms

    @staticmethod
    def _ensure_query_shape(q: Sequence[float], dim: int) -> np.ndarray:
        qv = np.array(q, dtype=np.float32).reshape(1, -1)
        if qv.shape[1] != dim:
            raise ValueError(f"Dimensi query ({qv.shape[1]}) != dim store ({dim}).")
        return qv

    def add(
        self,
        *,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        if not (len(ids) == len(texts) == len(embeddings)):
            raise ValueError("Panjang ids, texts, embeddings harus sama.")

        if metadatas is None:
            metadatas = [{} for _ in range(len(ids))]
        if len(metadatas) != len(ids):
            raise ValueError("Panjang metadatas harus sama dengan ids.")

        new_emb = self._to_2d_float_array(embeddings)

        if self._emb is None:
            self._emb = new_emb
        else:
            if new_emb.shape[1] != self._emb.shape[1]:
                raise ValueError(
                    f"Dimensi embedding baru ({new_emb.shape[1]}) != store ({self._emb.shape[1]})."
                )
            self._emb = np.vstack([self._emb, new_emb])

        self._ids.extend([str(x) for x in ids])
        self._texts.extend([str(x) for x in texts])
        self._metas.extend([dict(m) for m in metadatas])

        self._emb_norm = self._normalize_rows(self._emb)

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        if self._emb is None or self._emb_norm is None or len(self._ids) == 0:
            return []

        dim = self._emb_norm.shape[1]
        qv = self._ensure_query_shape(query_embedding, dim)
        qv_norm = self._normalize_rows(qv)

        scores = (self._emb_norm @ qv_norm.T).reshape(-1)  # cosine sim

        top_k = max(1, int(top_k))
        k = min(top_k, scores.shape[0])

        idxs = np.argpartition(-scores, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]

        results: List[SearchResult] = []
        for i in idxs:
            s = float(scores[i])
            if score_threshold is not None and s < float(score_threshold):
                continue
            results.append(
                SearchResult(
                    doc_id=self._ids[i],
                    score=s,
                    text=self._texts[i],
                    metadata=self._metas[i],
                )
            )
        return results

    def save(self, folder: str | Path) -> None:
        if self._emb is None:
            raise ValueError("Store kosong, tidak ada yang disimpan.")

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(folder / "embeddings.npz", emb=self._emb)

        docs = {"ids": self._ids, "texts": self._texts, "metadatas": self._metas}
        with open(folder / "docs.json", "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, folder: str | Path) -> "SimpleVectorStore":
        folder = Path(folder)
        emb_path = folder / "embeddings.npz"
        docs_path = folder / "docs.json"

        if not emb_path.exists() or not docs_path.exists():
            raise FileNotFoundError("embeddings.npz atau docs.json tidak ditemukan.")

        obj = cls()

        data = np.load(emb_path)
        emb = data["emb"].astype(np.float32)
        if emb.ndim != 2:
            raise ValueError("embeddings.npz tidak valid (harus 2D).")
        obj._emb = emb
        obj._emb_norm = obj._normalize_rows(obj._emb)

        with open(docs_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        obj._ids = [str(x) for x in docs.get("ids", [])]
        obj._texts = [str(x) for x in docs.get("texts", [])]
        obj._metas = [dict(x) for x in docs.get("metadatas", [])]

        if not (len(obj._ids) == len(obj._texts) == len(obj._metas) == obj._emb.shape[0]):
            raise ValueError("docs.json tidak konsisten dengan embeddings.")

        return obj
