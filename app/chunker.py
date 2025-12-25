from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


def chunk_text(
    *,
    text: str,
    base_id: str,
    metadata: Dict[str, Any],
    max_chars: int = 800,
    overlap: int = 120,
) -> List[Chunk]:
    """
    Chunking sederhana berbasis karakter dengan overlap.
    Cocok untuk RAG awal.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Normalisasi whitespace sederhana
    text = " ".join(text.split())

    chunks: List[Chunk] = []
    start = 0
    idx = 1

    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk_str = text[start:end].strip()

        if chunk_str:
            chunks.append(
                Chunk(
                    chunk_id=f"{base_id}_c{idx}",
                    text=chunk_str,
                    metadata=dict(metadata),
                )
            )

        if end == len(text):
            break

        # maju dengan overlap
        start = max(0, end - overlap)
        idx += 1

    return chunks
