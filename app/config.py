import os
from pathlib import Path

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

# =========================
# DATA PATH
# =========================
DATA_DIR = BASE_DIR / "data"
FAQ_PATH = DATA_DIR / "faq.json"

# =========================
# CHATBOT CONFIG
# =========================
TFIDF_CONFIG = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "stop_words": None
}

SIMILARITY_THRESHOLD = 0.25

# =========================
# API CONFIG
# =========================
API_CONFIG = {
    "title": "FAQ Chatbot API",
    "version": "1.0.0",
    "description": "Chatbot FAQ berbasis NLP"
}

# =========================
# ENVIRONMENT
# =========================
ENV = os.getenv("ENV", "development")
DEBUG = ENV == "development"

# =========================
# FUTURE LLM CONFIG (optional)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
