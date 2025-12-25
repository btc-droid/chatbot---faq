import re
import string


class TextPreprocessor:
    """
    Utility class untuk preprocessing teks
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def clean_text(self, text: str) -> str:
        """
        Pipeline utama preprocessing
        """
        text = text.strip()
        text = self._lowercase(text)
        text = self._remove_punctuation(text)
        text = self._remove_extra_whitespace(text)
        return text

    def _lowercase(self, text: str) -> str:
        if self.lowercase:
            return text.lower()
        return text

    def _remove_punctuation(self, text: str) -> str:
        """
        Hapus tanda baca
        """
        return text.translate(str.maketrans("", "", string.punctuation))

    def _remove_extra_whitespace(self, text: str) -> str:
        """
        Hapus spasi berlebih
        """
        return re.sub(r"\s+", " ", text).strip()

    def preprocess_list(self, texts: list[str]) -> list[str]:
        """
        Preprocessing banyak teks sekaligus
        """
        return [self.clean_text(text) for text in texts]


# =========================
# TEST MANUAL
# =========================
if __name__ == "__main__":
    processor = TextPreprocessor()

    sample = "   Bagaimana   cara   Pendaftaran???  "
    print("Original :", sample)
    print("Processed:", processor.clean_text(sample))
