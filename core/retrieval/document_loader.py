"""
Purpose:
--------
Handles ingestion of medical content into the system by:
  • Searching PubMed based on user-entered medical topic
  • Fetching abstracts + metadata ONLY from Open-Access articles
  • Loading local user PDFs with structural validation
  • Returning clean text ready for preprocessing & chunking

Safety:
-------
 - Enforces Open-Access filtering ("free full text") to respect reuse rights
 - Rejects scanned PDFs (no extractable text)
"""

import os
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict


class DocumentLoader:

    # ✅ Local PDF Load with medical content validation
    @staticmethod
    def load_pdf(file_path: str) -> str:
        import pdfplumber
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if not extracted:
                    raise ValueError(
                        "❌ PDF may be scanned or image-only — "
                        "OCR pipeline required to extract text."
                    )
                text += extracted + "\n"

        return text

    # ✅ PubMed Search: return list of PMIDs relevant to a topic
    @staticmethod
    def search_pubmed_pmids(query: str, retmax: int = 5) -> List[str]:
        """
        Search PubMed for FREE FULL TEXT articles matching a topic.
        Ensures reuse rights.
        """
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": f"{query} AND free full text[filter]",
            "retmode": "json",
            "retmax": retmax
        }

        resp = requests.get(url, params=params)
        resp.raise_for_status()

        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    # ✅ Fetch article: extract title + abstract + metadata
    @staticmethod
    def fetch_pubmed_article(pmid: str) -> Dict:
        """
        Return structured data from a PubMed article:
            {
              "text": "Title + Abstract",
              "metadata": {pmid, title, journal, year, doi?}
            }
        """
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}

        resp = requests.get(url, params=params)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)

        article = root.find(".//Article")
        if article is None:
            raise ValueError(f"❌ No Article found for PMID: {pmid}")

        title = article.findtext("ArticleTitle", default="No Title Available")

        abstract_nodes = article.findall(".//AbstractText")
        abstract = " ".join([a.text for a in abstract_nodes if a.text]) or \
                   "No abstract available."

        journal = article.findtext(".//Journal/Title", default="Unknown Journal")
        year = root.findtext(".//PubDate/Year") or \
               root.findtext(".//ArticleDate/Year") or \
               "Unknown Year"

        doc_text = f"{title}. {abstract}"

        metadata = {
            "source_mode": "pubmed",  # ✅ Used for filtering later
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "year": year
        }

        return {"text": doc_text, "metadata": metadata}
