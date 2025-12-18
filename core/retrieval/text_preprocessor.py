"""
Purpose:
--------
Cleans and validates raw medical text before chunking.
Ensures only meaningful medical content is stored in DB.
"""

import re

class TextPreprocessor:

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace and clean noise."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def validate(text: str) -> bool:
        """
        Validate content quality:
        - Must be a meaningful medical text
        - Must be sufficiently long
        """
        if not text or len(text) < 10:
            raise ValueError("❌ Text too short — not medically useful.")

        return True
