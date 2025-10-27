
"""
Local LLM provider using LangChain + Ollama (Llama 3).
"""

from langchain.llms import Ollama


class LLMProvider:
    def __init__(self, model: str = "llama3", temperature: float = 0.2):
        # Local Llama-3 model via Ollama
        self.llm = Ollama(model=model, temperature=temperature)

    def generate(self, system: str, user: str) -> str:
        # Format prompt according to Ollama / LangChain expectations
        prompt = f"<SYSTEM>\n{system}\n</SYSTEM>\n<USER>\n{user}\n</USER>"
        response = self.llm(prompt)
        return response.strip()
