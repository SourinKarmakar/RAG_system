# Runbook

## **Common Failure Scenarios**

| Scenario | Cause | Mitigation |
|-----------|--------|-------------|
| Redis not running | Connection refused | Restart Redis and check URL env |
| FAISS index file missing | Manual delete or wrong path | Rebuild index |
| Memory error | Too many documents | Use incremental indexing or batch mode |
| API timeout | Long embedding compute | Use Redis coalescing or async tasks |

---

## **Rollback Procedure**

1. Stop the API server  
2. Restore previous `faiss.index` and `bm25.pkl` from backup  
3. Restart FastAPI  
4. Verify index counts via `/status`
5. Check if all services are up and running (OpenAI, OpenAI API Keys, GPU system where LLM is hosted locally)

---

## **Recovery Notes**

- Failed embedding batches are skipped and logged
- Keep weekly snapshot of `vector_store/`
- Keep a track of all the logs generated due to error
- Check for metrics if tracked using monitoring app
