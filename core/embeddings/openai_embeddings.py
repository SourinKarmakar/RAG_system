import numpy as np
from openai import OpenAI


class OpenAIEmbeddings:

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model_name = model_name


    def embed_texts(self, texts):
        """Generate embeddings for multiple texts using OpenAI."""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(model=self.model_name, input=text)
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings, dtype="float32")


    def embed_query(self, query):
        """Generate embedding for a single query string."""
        response = self.client.embeddings.create(model=self.model_name, input=query)
        return np.array([response.data[0].embedding], dtype="float32")
