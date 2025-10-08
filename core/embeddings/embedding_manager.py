import numpy as np
from typing import List
from .local_embeddings import LocalEmbeddings
from .openai_embeddings import OpenAIEmbeddings


class EmbeddingManager:

    def __init__(self, provider = "local", model_name = None):
        provider = provider.lower().strip()

        if provider == "openai":
            model_name = model_name or "text-embedding-3-small"
            self.embedder = OpenAIEmbeddings(model_name)
        elif provider == "local":
            model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self.embedder = LocalEmbeddings(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.provider = provider

    def embed_texts(self, texts):
        return self.embedder.embed_texts(texts)

    def embed_query(self, query):
        return self.embedder.embed_query(query)
