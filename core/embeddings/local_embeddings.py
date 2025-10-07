from sentence_transformers import SentenceTransformer


class LocalEmbeddings:

    def __init__(self, model_name = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)


    def embed_texts(self, texts, show_progress=False):
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings


    def embed_query(self, query):
        """Generate embedding for a single query."""
        return self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
