# pipeline/chunker_pipeline.py
from typing import List, Dict
from core.embeddings.embedding_manager import EmbeddingManager
from core.vectorstores.faiss_client import FAISSManager
from core.retriever.bm_25_client import BM25Manager
from core.chunking.base_processor import BaseProcessor
import os

class ChunkerPipeline:
    def __init__(self, embed_provider = "local", model_name = None):
        self.processor = BaseProcessor()
        self.embed_mgr = EmbeddingManager(provider=embed_provider, model_name=model_name)
        self.faiss = FAISSManager()
        self.bm25 = BM25Manager()


    def ingest_file(self, file_path):
        chunks = self.processor.process_file(file_path)
        texts = [c["content"] for c in chunks]
        embeddings = self.embed_mgr.embed_texts(texts)
        self.faiss.add(embeddings, chunks)
        # rebuild BM25 from all metadata (simple approach)
        self.bm25.build(self.faiss.metadata)
        return {"ingested": len(chunks)}


    def query_faiss(self, q, top_k = 5):
        qvec = self.embed_mgr.embed_query(q)
        return self.faiss.search(qvec, top_k=top_k)


    def query_bm25(self, q, top_k = 5):
        return self.bm25.query(q, top_k=top_k)


    def query_deep(self, q, faiss_k = 500, rerank_k = 10, alpha=0.6):
        """
        Deep flow: FAISS retrieve faiss_k -> BM25 rerank on those candidates -> return top rerank_k
        alpha: weight for FAISS score (0..1), (1-alpha) for BM25
        """
        # Stage1: dense candidates
        qvec = self.embed_mgr.embed_query(q)
        dense = self.faiss.index.search(qvec.astype("float32"), faiss_k)
        D, I = dense  # squared similarities (IP since normalized)
        candidate_idxs = [int(i) for i in I[0] if i != -1]
        # Build local subset for BM25 scoring
        subset_meta = [self.faiss.metadata[i] for i in candidate_idxs]
        # BM25 on subset
        tmp_bm25 = BM25Manager()
        tmp_bm25.build(subset_meta)
        bm25_results = tmp_bm25.query(q, top_k=len(subset_meta))
        # Merge scores: find bm25 score map (by content)
        bm25_score_map = { (r["content"]): r["score"] for r in bm25_results }
        merged = []
        for idx_pos, (idx, sim) in enumerate(zip(candidate_idxs, D[0])):
            meta = self.faiss.metadata[idx]
            bm25_score = bm25_score_map.get(meta["content"], 0.0)
            combined = alpha * float(sim) + (1 - alpha) * float(bm25_score)
            merged.append({"heading": meta["heading"], "content": meta["content"], "type": meta["type"], "score": combined})
        merged_sorted = sorted(merged, key=lambda x: x["score"], reverse=True)[:rerank_k]
        return merged_sorted


    def save(self, dir_path="vectorstore"):
        self.faiss.save(dir_path)
        self.bm25.save(os.path.join(dir_path, "bm25.pkl"))


    def load(self, dir_path="vectorstore"):
        self.faiss.load(dir_path)
        self.bm25.load(os.path.join(dir_path, "bm25.pkl"))
