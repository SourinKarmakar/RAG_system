# Performace Analysis

## Latency Breakdown

| Component | Avg Time (ms) | Description |
|------------|----------------|-------------|
| API Layer | 2-5 ms | Request handling |
| Redis cache lookup | 5-10 ms | Checks if query result already available |
| Redis cache writing | 2-3 ms | Prevents duplicate computations |
| Chunk Extraction | 100 - 2000 ms | PDF / docx parsing |
| Embedding | 200–600 ms | Local MiniLM / OpenAI API |
| FAISS Search | 20–50 ms | Dense vector retrieval |
| BM25 Search | 10–30 ms | Lexical token scoring |
| Cache Miss | >500ms | One-time compute |
| API Response (total) | ~700ms | Average for deep search |
| API Response (total) | ~200ms | Average for quick search |
| File upload and chunking | ~1000 - 3000 ms | Creating chunks, embedding for large document(s) |

---

## **Bottlenecks**

- **Embedding latency**: OpenAI API-bound (most of the time utilized in API calling), Local LLMs using GPUs might be extremely fast wrt OpenAI.
- **Chunking large PDFs**: I/O bound, context aware structural retrieval with heuristics.
- **Reranking in Python loop**: CPU heavy for large K
- **Documents contents**: Some PDF might contain images but currently only supports headers, content and tables

---

## **Optimization Opportunities**

| Layer | Improvement |
|--------|-------------|
| Embedding | Batch encode + caching |
| Chunking | Async I/O for multi-page PDFs |
| Search | Use FAISS IVF-HNSW for faster top-K |
| Reranking | Parallelize using multiprocessing |
| Local LLM on GPUs | Faster embeddings |
