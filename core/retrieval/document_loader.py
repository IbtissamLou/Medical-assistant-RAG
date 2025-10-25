import os
import requests
import pdfplumber
from typing import Optional

class DocumentLoader:
    """Loader capable of:
       - fetching via PubMed/NCBI API
       - fetching WHO guideline content
       - loading local PDF/text files uploaded by user
    """

    @staticmethod
    def load_pdf(file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text

    @staticmethod
    def load_text(file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def fetch_pubmed_abstract(pmid: str, api_key: Optional[str]=None) -> str:
        """Fetch abstract (and maybe full-text if OA) for a given PubMed ID (PMID)."""
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        if api_key:
            params["api_key"] = api_key

        resp = requests.get(base, params=params)
        resp.raise_for_status()
        xml = resp.text
        # For production: parse xml properly, here quick extraction
        # you can use xml.etree.ElementTree
        return xml

    @staticmethod
    def fetch_who_guideline(download_url: str) -> str:
        """Fetch WHO guideline document via direct download link (PDF/HTML)."""
        resp = requests.get(download_url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type:
            # save locally as temp file and load via pdf loader
            temp_path = "/tmp/who_guideline.pdf"
            with open(temp_path, "wb") as f:
                f.write(resp.content)
            return DocumentLoader.load_pdf(temp_path)
        else:
            # assume HTML or text
            return resp.text
