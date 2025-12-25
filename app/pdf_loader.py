from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from pypdf import PdfReader
except ImportError as e:
    raise ImportError(
        "Package 'pypdf' belum terpasang. Jalankan: pip install pypdf"
    ) from e


@dataclass
class PDFPage:
    page_number: int  # 1-indexed
    text: str


def load_pdf_pages(pdf_path: str | Path) -> List[PDFPage]:
    """
    Extract teks per halaman dari PDF.
    Catatan: jika PDF hasil scan (gambar), teks bisa kosong (butuh OCR).
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    pages: List[PDFPage] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Rapikan sedikit
        text = text.replace("\u00a0", " ").strip()
        pages.append(PDFPage(page_number=i + 1, text=text))
    return pages
