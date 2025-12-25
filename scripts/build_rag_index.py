from __future__ import annotations

import sys
from pathlib import Path
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pdf_loader import load_pdf_pages
from app.chunker import chunk_text
from app.preprocessing import TextPreprocessor
from app.vector_store import SimpleVectorStore


def main():
    base_dir = ROOT
    pdf_path = base_dir / "data" / "documents" / "handbook.pdf"
    index_dir = base_dir / "models" / "vector_store"
    index_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF tidak ditemukan: {pdf_path}")

    pages = load_pdf_pages(pdf_path)
    prep = TextPreprocessor()

    all_ids, all_texts, all_metas = [], [], []

    for p in pages:
        if not p.text.strip():
            continue

        page_chunks = chunk_text(
            text=p.text,
            base_id=f"handbook_p{p.page_number}",
            metadata={"source": "handbook.pdf", "page": p.page_number},
            max_chars=300,
            overlap=60,
        )

        for ch in page_chunks:
            all_ids.append(ch.chunk_id)
            all_texts.append(ch.text)
            all_metas.append(ch.metadata)

    if not all_texts:
        raise RuntimeError(
            "Tidak ada teks yang berhasil diekstrak dari PDF. "
            "Kemungkinan PDF hasil scan (gambar) dan butuh OCR."
        )

    clean_texts = prep.preprocess_list(all_texts)

    vectorizer = TfidfVectorizer(lowercase=False, ngram_range=(1, 2))
    X = vectorizer.fit_transform(clean_texts).toarray().astype(np.float32)

    store = SimpleVectorStore()
    store.add(ids=all_ids, texts=all_texts, embeddings=X, metadatas=all_metas)
    store.save(index_dir)

    with open(index_dir / "tfidf.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"✅ RAG index berhasil dibuat di: {index_dir}")
    print(f"   Total chunks: {len(all_texts)}")
    print("   File yang dibuat: embeddings.npz, docs.json, tfidf.pkl")


if __name__ == "__main__":
    main()
