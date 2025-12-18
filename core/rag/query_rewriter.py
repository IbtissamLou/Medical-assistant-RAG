from core.rag.llm_provider import LLMProvider

class QueryRewriter:
    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm or LLMProvider(model="phi3:latest", temperature=0.0)

    def expand(self, user_query: str) -> str:
        prompt = (
            "Rewrite the following medical search into a PubMed-friendly query. "
            "Include synonyms and possible MeSH terms, add tiab field where helpful. "
            "Keep it concise, no explanation.\n\n"
            f"Query: {user_query}"
        )
        return self.llm.llm(prompt).strip()
