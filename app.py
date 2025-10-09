# main.py
import os
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import redis
import asyncio

from core.pipeline import ChunkerPipeline
from config.settings import (
    REDIS_URL,
    EMBED_PROVIDER,
    MODEL_NAME,
    FILE_STORAGE_PATH,
    INDEX_PERSISTENCE_STORAGE_PATH
)


r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

app = FastAPI()

pipeline = ChunkerPipeline(embed_provider=EMBED_PROVIDER, model_name=MODEL_NAME)


# Simple in-memory lock for request coalescing (per query) using Redis SETNX
async def get_cached_or_compute(key, ttl, compute_coro):
    # check cache
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    lock_key = f"lock:{key}"
    got_lock = r.set(lock_key, "1", nx=True, ex=30)
    if not got_lock:
        # someone else is computing; poll for result
        for _ in range(30):
            await asyncio.sleep(0.2)
            cached = r.get(key)
            if cached:
                return json.loads(cached)
        raise HTTPException(status_code=504, detail="Timeout waiting for cached result")
    try:
        result = await compute_coro()
        r.set(key, json.dumps(result), ex=ttl)
        return result
    finally:
        r.delete(lock_key)


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    results = []
    os.makedirs(FILE_STORAGE_PATH, exist_ok=True)
    for file in files:
        temp_path = os.path.join(FILE_STORAGE_PATH, file.filename)
        with open(temp_path, "wb") as f_out:
            f_out.write(await file.read())
        result = pipeline.ingest_file(temp_path)
        results.append({"file": file.filename, "chunks": result["ingested"]})
    return JSONResponse({"status": "ok", "results": results})


@app.get("/search/quick")
async def search_quick(q: str, top_k: int = 5):
    # check redis cache first
    key = f"quick:{q}:{top_k}"
    async def do_search():
        # bm25 -> quick lexical search; we return both bm25 and faiss top 1 as quick hybrid
        bm = pipeline.query_bm25(q, top_k=top_k)
        if not bm:
            # fallback to faiss
            fa = pipeline.query_faiss(q, top_k=top_k)
            return {"source":"faiss", "results": fa}
        return {"source":"bm25", "results": bm}
    result = await get_cached_or_compute(key, ttl=60, compute_coro=do_search)
    return result


@app.get("/search/deep")
async def search_deep(q: str, faiss_k: int = 500, rerank_k: int = 10):
    key = f"deep:{q}:{faiss_k}:{rerank_k}"
    async def do_deep():
        res = pipeline.query_deep(q, faiss_k=faiss_k, rerank_k=rerank_k)
        return {"source":"hybrid", "results": res}
    result = await get_cached_or_compute(key, ttl=30, compute_coro=do_deep)
    return result


@app.post("/save")
def save_index():
    pipeline.faiss.save(INDEX_PERSISTENCE_STORAGE_PATH)
    bm_25_filepath = os.path.join(INDEX_PERSISTENCE_STORAGE_PATH, "bm25.pkl")
    pipeline.bm25.save(bm_25_filepath)
    return {"saved": True}


@app.post("/load")
def load_index():
    pipeline.faiss.load(INDEX_PERSISTENCE_STORAGE_PATH)
    bm_25_filepath = os.path.join(INDEX_PERSISTENCE_STORAGE_PATH, "bm25.pkl")
    pipeline.bm25.load(bm_25_filepath)
    return {"loaded": True}


@app.get("/status")
def status():
    return {
        "faiss_vectors": pipeline.faiss.index.ntotal if pipeline.faiss.index is not None else 0,
        "bm25_corpus": len(pipeline.bm25.meta) if pipeline.bm25.bm25 is not None else 0,
        "embed_provider": pipeline.embed_mgr.provider
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
