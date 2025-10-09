from rank_bm25 import BM25Okapi
import numpy as np
import pickle
import os

class BM25Manager:

    def __init__(self):
        self.corpus = []     # texts
        self.meta = []      # metadata parallel to corpus
        self.bm25 = None


    def build(self, metadata):
        self.meta = metadata.copy()
        self.corpus = [m["content"] for m in self.meta]
        tokenized = [c.split() for c in self.corpus]
        self.bm25 = BM25Okapi(tokenized)


    def query(self, q, top_k = 10):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(q.split())
        idxs = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in idxs:
            results.append({
                "heading": self.meta[i].get("heading"),
                "content": self.meta[i].get("content"),
                "type": self.meta[i].get("type"),
                "score": float(scores[i])
            })
        return results


    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"meta": self.meta, "corpus": self.corpus}, f)


    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("bm25 not found")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.meta = data["meta"]
        self.corpus = data["corpus"]
        tokenized = [c.split() for c in self.corpus]
        self.bm25 = BM25Okapi(tokenized)
