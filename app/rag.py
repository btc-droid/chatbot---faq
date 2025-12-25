from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.preprocessing import TextPreprocessor
from app.vector_store import SimpleVectorStore, SearchResult


@dataclass
class RAGAnswer:
    answer: str
    confidence: float
    contexts: List[SearchResult]


class TfidfRAGRetriever:
    """
    RAG retriever berbasis TF-IDF embeddings.
    - Load vector store + tfidf.pkl
    - Query -> embedding -> similarity search -> return top chunks
    """

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.store = SimpleVectorStore.load(self.index_dir)
        self.prep = TextPreprocessor()

        tfidf_path = self.index_dir / "tfidf.pkl"
        if not tfidf_path.exists():
            raise FileNotFoundError(f"tfidf.pkl tidak ditemukan di {self.index_dir}")

        with open(tfidf_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def embed_query(self, query: str) -> np.ndarray:
        q = self.prep.clean_text(query)
        vec = self.vectorizer.transform([q]).toarray().astype(np.float32)[0]
        return vec

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.20) -> List[SearchResult]:
        q_emb = self.embed_query(query)
        return self.store.search(query_embedding=q_emb, top_k=top_k, score_threshold=score_threshold)

    def answer(self, query: str, top_k: int = 3) -> RAGAnswer:
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return RAGAnswer(
                answer="Maaf, saya belum menemukan jawaban di handbook.",
                confidence=0.0,
                contexts=[],
            )

        # Jawaban sederhana: gabungkan 1-3 chunk teratas sebagai konteks jawaban
        # (Nanti kalau pakai LLM, konteks ini jadi prompt untuk generate jawaban)
        best = hits[0]
        combined = "\n\n".join([h.text for h in hits])

        # Batasi panjang supaya tidak “kepanjangan”
        combined = combined[:1200].strip()

        return RAGAnswer(
            answer=combined,
            confidence=float(best.score),
            contexts=hits,
        )
