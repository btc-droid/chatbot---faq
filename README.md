# FAQ Chatbot (NLP) + API (FastAPI)

Proyek ini adalah **Chatbot FAQ** berbasis **TF-IDF + Cosine Similarity** yang dapat menjawab pertanyaan dari data FAQ (CSV/JSON).
Struktur proyek juga sudah disiapkan untuk pengembangan lanjut ke **RAG (PDF Handbook + Vector Store)**.

## Fitur
- ✅ Chatbot FAQ berbasis NLP (TF-IDF + Cosine Similarity)
- ✅ REST API menggunakan FastAPI (`/chat`)
- ✅ Struktur data rapi: `faq.csv` dan `faq.json`
- ✅ Preprocessing teks sederhana
- ✅ Fondasi Vector Store lokal (untuk RAG tahap lanjut)

---

## Struktur Folder
chatbot-faq/
│
├── data/
│ ├── faq.json
│ ├── faq.csv
│ └── documents/
│ └── handbook.pdf
│
├── app/
│ ├── main.py
│ ├── chatbot.py
│ ├── faq.py
│ ├── preprocessing.py
│ ├── config.py
│ └── vector_store.py
│
├── models/
│ └── vector_store/ # output index vector store (otomatis dibuat jika dipakai)
│
├── requirements.txt
└── README.md

yaml
Copy code

---

## Instalasi
Pastikan Python >= 3.10.

1) Buat virtual environment (opsional tapi disarankan):
```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

## Install dependencies:

bash
Copy code
pip install -r requirements.txt
Contoh isi requirements.txt minimal:

txt
Copy code
fastapi
uvicorn
scikit-learn

## Menyiapkan Data

Letakkan file data di folder berikut:

data/faq.json (dipakai chatbot saat ini)

data/faq.csv (opsional untuk editing dengan Excel/Sheets)

data/documents/handbook.pdf (opsional untuk tahap RAG)

## Format faq.json

json
Copy code
[
  {
    "id": 1,
    "category": "pendaftaran",
    "question": "Bagaimana cara pendaftaran?",
    "answer": "Pendaftaran dilakukan secara online melalui website resmi kampus.",
    "tags": ["daftar", "registrasi"]
  }
]

## Menjalankan Chatbot (CLI)

Jalankan langsung mode terminal:

bash
Copy code
python app/chatbot.py
Ketik exit untuk keluar.

## Menjalankan API (FastAPI)

Jalankan server:

bash
Copy code
uvicorn app.main:app --reload
Buka:

Swagger UI: http://127.0.0.1:8000/docs

Root check: http://127.0.0.1:8000/

## Request ke /chat
## POST /chat

Body:

json
Copy code
{
  "message": "cara pendaftaran"
}
Response:

json
Copy code
{
  "answer": "Pendaftaran dilakukan secara online melalui website resmi kampus.",
  "confidence": 0.73
}

## Konfigurasi

Konfigurasi ada di app/config.py:

FAQ_PATH → lokasi faq.json

SIMILARITY_THRESHOLD → ambang confidence (default 0.25)

TFIDF_CONFIG → setting TF-IDF

(opsional) OPENAI_API_KEY → untuk tahap RAG/LLM nanti

Jika suatu saat pakai environment variable:

bash
Copy code
export OPENAI_API_KEY="..."

## Vector Store (Tahap Lanjut / RAG)

Folder models/vector_store/ tidak diisi manual.
Folder ini akan berisi hasil indexing seperti:

embeddings.npz (vektor embedding)

docs.json (teks chunk + metadata)

Vector store engine lokal ada di: app/vector_store.py (SimpleVectorStore).

Catatan: untuk RAG full, kita butuh pipeline tambahan:

PDF loader (extract teks)

chunker (memecah teks)

embedding generator

build index → simpan ke models/vector_store/

## Tips Pengembangan

Tambah/ubah FAQ paling enak lewat faq.csv, lalu generate ulang faq.json.

Untuk produksi, sebaiknya buat admin CRUD (opsional).

## Roadmap

 Dukungan load langsung dari CSV (tanpa konversi manual)

 PDF loader + chunker

 RAG: handbook.pdf → vector store → jawaban lebih kaya konteks

 Deploy (Railway/Render/AWS)
