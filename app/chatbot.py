import json
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessing import TextPreprocessor
from app.rag import TfidfRAGRetriever


class FAQChatbot:
    def __init__(
        self,
        faq_path: str,
        *,
        faq_threshold: float = 0.25,
        enable_rag: bool = True,
        rag_index_dir: str = "models/vector_store",
        rag_top_k: int = 3,
        rag_score_threshold: float = 0.20,
        rag_max_answer_chars: int = 600,
    ):
        self.faq_path = faq_path
        self.faq_threshold = float(faq_threshold)
        self.rag_top_k = int(rag_top_k)
        self.rag_score_threshold = float(rag_score_threshold)
        self.rag_max_answer_chars = int(rag_max_answer_chars)

        self.prep = TextPreprocessor()

        # ---- FAQ setup (TF-IDF) ----
        self.faq_data = self._load_faq()
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=(1, 2))
        self._prepare_faq_vectors()

        # ---- RAG setup (optional) ----
        self.rag: Optional[TfidfRAGRetriever] = None
        if enable_rag:
            try:
                index_dir = Path(rag_index_dir)
                if (index_dir / "embeddings.npz").exists() and (index_dir / "tfidf.pkl").exists():
                    self.rag = TfidfRAGRetriever(index_dir)
            except Exception:
                self.rag = None

    def _load_faq(self):
        with open(self.faq_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _prepare_faq_vectors(self):
        self.questions = []
        self.answers = []

        for item in self.faq_data:
            q = item["question"]
            tags = item.get("tags", [])
            if tags:
                q += " " + " ".join(tags)

            self.questions.append(self.prep.clean_text(q))
            self.answers.append(item["answer"])

        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def _format_rag_answer(self, text: str) -> str:
        """
        Rapikan jawaban dari chunk handbook agar lebih enak dibaca.
        """
        if not text:
            return ""

        # rapikan whitespace
        text = " ".join(text.split()).strip()

        # bikin BAB terlihat lebih jelas kalau ada
        text = text.replace("BAB", "\nBAB").strip()

        # batasi panjang jawaban
        if len(text) > self.rag_max_answer_chars:
            text = text[: self.rag_max_answer_chars].rstrip() + "..."

        return text

    def get_answer(self, user_input: str):
        user_input_clean = self.prep.clean_text(user_input)

        # 1) Coba jawab dari FAQ
        user_vec = self.vectorizer.transform([user_input_clean])
        sim = cosine_similarity(user_vec, self.tfidf_matrix)[0]

        best_idx = int(sim.argmax())
        best_score = float(sim[best_idx])

        if best_score >= self.faq_threshold:
            return {
                "answer": self.answers[best_idx],
                "confidence": best_score,
                "source": "faq",
                "contexts": [],
            }

        # 2) Fallback ke RAG (handbook) jika tersedia
        if self.rag is not None:
            hits = self.rag.retrieve(
                user_input_clean,
                top_k=self.rag_top_k,
                score_threshold=self.rag_score_threshold,
            )
            if hits:
                best_text = hits[0].text  # ambil chunk terbaik saja
                answer = self._format_rag_answer(best_text)

                return {
                    "answer": answer,
                    "confidence": float(hits[0].score),
                    "source": "handbook",
                    "contexts": [
                        {"id": h.doc_id, "score": float(h.score), "metadata": h.metadata} for h in hits
                    ],
                }

        return {
            "answer": "Maaf, saya belum menemukan jawaban yang sesuai.",
            "confidence": best_score,
            "source": "none",
            "contexts": [],
        }


if __name__ == "__main__":
    bot = FAQChatbot("data/faq.json", enable_rag=True)
    print("Chatbot FAQ + RAG siap! (ketik 'exit' untuk keluar)\n")

    while True:
        msg = input("Anda: ")
        if msg.lower() == "exit":
            break
        res = bot.get_answer(msg)
        print(f"Bot ({res.get('source')}): {res['answer']}\n(confidence: {res['confidence']:.2f})\n")
