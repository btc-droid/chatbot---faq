FROM python:3.13-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Build RAG index at build-time (butuh handbook.pdf ada di repo)
RUN python scripts/build_rag_index.py

# Railway will provide PORT env var
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
