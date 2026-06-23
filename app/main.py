from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from uuid import uuid4
import logging
import time


import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from app.core.search_engine import SemanticSearchEngine, Document



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Semantic Search Engine...")
    yield
    logger.info("Shutting down Semantic Search Engine...")


app = FastAPI(
    title="Semantic Search Engine API",
    version="2.0.0",
    description="Production-grade semantic vector search API",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


engine = SemanticSearchEngine()



class DocumentPayload(BaseModel):
    id: Optional[str] = None
    text: str = Field(..., min_length=1)
    metadata: Optional[Dict] = Field(default_factory=dict)

    @validator("text")
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("Document text cannot be empty")
        return value.strip()


class IndexRequest(BaseModel):
    documents: List[DocumentPayload]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)

    @validator("query")
    def validate_query(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be empty")
        return value.strip()


class SearchResponse(BaseModel):
    id: str
    text: str
    score: float
    rank: int
    metadata: Dict



@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()

    response = await call_next(request)

    duration = round(time.time() - start, 4)

    logger.info(
        f"{request.method} {request.url.path} "
        f"Status={response.status_code} "
        f"Duration={duration}s"
    )

    return response



@app.get("/")
async def root():
    return {
        "service": "Semantic Search Engine",
        "version": "2.0.0",
        "status": "running"
    }



@app.post("/index")
async def index_documents(req: IndexRequest):

    try:
        docs = []

        for d in req.documents:

            doc_id = d.id or str(uuid4())

            docs.append(
                Document(
                    id=doc_id,
                    text=d.text,
                    metadata=d.metadata or {}
                )
            )

        indexed_count = engine.add_documents(docs)

        logger.info(f"Indexed {indexed_count} documents")

        return {
            "success": True,
            "indexed_documents": indexed_count,
            "total_documents": engine.index.size()
        }

    except Exception as e:
        logger.exception("Indexing failed")

        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(e)}"
        )



@app.post("/search")
async def search_documents(req: SearchRequest):

    try:
        results = engine.search(
            query=req.query,
            top_k=req.top_k
        )

        formatted_results = []

        for r in results:

            formatted_results.append(
                SearchResponse(
                    id=r.document.id,
                    text=r.document.text[:300],
                    score=round(float(r.score), 4),
                    rank=r.rank,
                    metadata=r.document.metadata
                )
            )

        return {
            "success": True,
            "query": req.query,
            "count": len(formatted_results),
            "results": formatted_results
        }

    except Exception as e:
        logger.exception("Search failed")

        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )



@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):

    try:
        deleted = engine.delete_document(document_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )

        return {
            "success": True,
            "deleted_document": document_id
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Deletion failed")

        raise HTTPException(
            status_code=500,
            detail=f"Deletion failed: {str(e)}"
        )



@app.get("/documents/{document_id}")
async def get_document(document_id: str):

    try:
        doc = engine.get_document(document_id)

        if not doc:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )

        return {
            "id": doc.id,
            "text": doc.text,
            "metadata": doc.metadata
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Failed to fetch document")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch document: {str(e)}"
        )



@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "indexed_documents": engine.index.size(),
        "numpy_version": np.__version__
    }


@app.get("/stats")
async def stats():

    return {
        "documents_indexed": engine.index.size(),
        "engine": "semantic-vector-search",
        "version": "2.0.0"
    }



"""
Run locally:

uvicorn app.main:app --reload

Production:

gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app
"""
