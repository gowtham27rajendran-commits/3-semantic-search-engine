from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.core.search_engine import SemanticSearchEngine, Document

app = FastAPI(title="Semantic Search Engine", version="1.0.0")
engine = SemanticSearchEngine()


class IndexRequest(BaseModel):
    documents: List[Dict]   # [{id, text, metadata}]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


@app.post("/index")
def index_documents(req: IndexRequest):
    docs = [Document(id=d["id"], text=d["text"], metadata=d.get("metadata", {}))
            for d in req.documents]
    count = engine.add_documents(docs)
    return {"indexed": count, "total_in_index": engine.index.size()}


@app.post("/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = engine.search(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "results": [
            {"id": r.document.id, "text": r.document.text[:200],
             "score": round(r.score, 4), "rank": r.rank, "metadata": r.document.metadata}
            for r in results
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok", "indexed_documents": engine.index.size()}
