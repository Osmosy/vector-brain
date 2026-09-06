"""Эмбеддер через локальный Ollama. Пакетный режим."""
import requests
from config import EMBED_MODEL, OLLAMA_URL


class OllamaEmbedder:
    def __init__(self, model: str = EMBED_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url
        self.session = requests.Session()

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            r = self.session.post(
                f"{self.url}/api/embed",
                json={"model": self.model, "input": t},
                timeout=60,
            )
            r.raise_for_status()
            out.append(r.json()["embeddings"][0])
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]