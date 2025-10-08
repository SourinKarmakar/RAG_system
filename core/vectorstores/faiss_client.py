# vectorstores/faiss_manager.py
import os
import faiss
import numpy as np
from sklearn.preprocessing import normalize

class FAISSManager:
    def __init__(self):
        self.index = None
        self.metadata = []

    def _make_index(self, dim: int):
        # Using IP index for cosine similarity; ensure we store normalized embeddings
        self.index = faiss.IndexFlatIP(dim)


    def add(self, embeddings, metadata):
        """
        embeddings: (N, D) float32
        metadata: list of dict with same length
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        # normalize for cosine similarity
        embeddings = normalize(embeddings, axis=1).astype("float32")

        if self.index is None:
            self._make_index(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata.extend(metadata)


    def search(self, query_vec: np.ndarray, top_k: int = 5):
        q = query_vec.astype("float32")
        # normalize for IP to cosine similarity
        q = normalize(q, axis=1)
        if self.index is None:
            return []
        D, I = self.index.search(q, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.metadata):
                m = self.metadata[idx]
                results.append({
                    "heading": m.get("heading"),
                    "content": m.get("content"),
                    "type": m.get("type"),
                    "score": float(score)
                })
        return results


    def save(self, dir_path = "vectorstore"):
        os.makedirs(dir_path, exist_ok=True)
        if self.index is None:
            raise RuntimeError("No index to save")
        faiss.write_index(self.index, os.path.join(dir_path, "faiss.index"))
        np.save(os.path.join(dir_path, "metadata.npy"), np.array(self.metadata, dtype=object))
        return dir_path


    def load(self, dir_path = "vector_store"):
        idx_path = os.path.join(dir_path, "faiss.index")
        meta_path = os.path.join(dir_path, "metadata.npy")
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Index or metadata not found")
        self.index = faiss.read_index(idx_path)
        self.metadata = np.load(meta_path, allow_pickle=True).tolist()
