"""
Builds a medical-safe prompt for RAG answers.
The LLM is asked to only use the provided CONTEXT and to cite sources
as [1], [2], ... which we later map to the metadata (PMID, journal, file name).
"""

from typing import List, Dict


SYSTEM_INSTRUCTIONS = """You are a medical assistant. Answer using only the supplied CONTEXT.
- If insufficient CONTEXT: say “Not enough evidence in provided sources.”
- Be concise, factual, and avoid speculation.
- Use clear headings and bullet points when helpful.
- Every medical statement MUST include inline citation like [1], [2].
- Never include information not present in CONTEXT — no hallucination."""

def format_context(chunks: List[Dict]) -> str:
    """Create a numbered context block for the LLM."""
    lines = []
    for i, item in enumerate(chunks, start=1):
        meta = item["metadata"]
        header = []
        if meta.get("source_mode") == "pubmed":
            header.append(f"PMID: {meta.get('pmid', 'N/A')}")
            if meta.get("journal"): header.append(meta["journal"])
            if meta.get("year"): header.append(str(meta["year"]))
        else:
            header.append(meta.get("file_name", "user_pdf"))
        lines.append(f"[{i}] {' — '.join(header)}\n{item['text']}\n")
    return "\n".join(lines)


def build_prompt(question: str, retrieved: List[Dict]) -> Dict[str, str]:
    """
    Returns a dict with system and user messages for a chat-style LLM.
    """
    context_block = format_context(retrieved)
    user_message = f"""QUESTION:
{question}

CONTEXT:
{context_block}

TASK:
1) Answer the question using only the CONTEXT above.
2) Add inline citations like [1], [2] where each citation points to a numbered source in CONTEXT.
3) If the CONTEXT is insufficient, say "I don't have enough evidence in the provided sources."""

    return {"system": SYSTEM_INSTRUCTIONS, "user": user_message}
