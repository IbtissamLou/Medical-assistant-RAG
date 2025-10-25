import re

class TextPreprocessor:

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)  # normalize whitespace
        text = text.strip()
        return text

    @staticmethod
    def validate(text: str):
        if not text or len(text) < 50:
            raise ValueError("Document too short or empty — rejected as low-quality data")
        return True
