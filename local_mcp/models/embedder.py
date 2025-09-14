from __future__ import annotations
import os
from typing import List, Optional

class OpenAIEmbedder:
    """
    OpenAI embeddings (text-embedding-3-small by default).
    Requires: pip install openai  AND  OPENAI_API_KEY env var.
    """
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use OpenAI embeddings.")
        try:
            from openai import OpenAI  # lazy import
        except ModuleNotFoundError as e:
            raise RuntimeError("The 'openai' package is not installed. Run: pip install openai") from e
        self._OpenAI = OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
