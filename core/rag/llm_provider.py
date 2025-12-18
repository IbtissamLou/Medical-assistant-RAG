
"""
Local LLM provider using LangChain + Ollama (Llama 3).
"""

from langchain.llms import Ollama


class LLMProvider:
    def __init__(self, model: str = "phi3:latest", temperature: float = 0.2):
        self.llm = Ollama(model=model, temperature=temperature)

    def generate(self, system: str, user: str) -> str:
        # Use chat-instruction format
        prompt = (
            f"[SYSTEM]\n{system}\n[/SYSTEM]\n"
            f"[INSTRUCTION]\n{user}\n[/INSTRUCTION]"
        )
        response = self.llm(prompt)
        return response.strip()

