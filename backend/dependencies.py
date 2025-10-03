"""Application dependencies."""

from typing import Optional

from fastapi import HTTPException

from rag.rag_engine_simple import QueenRAGEngine

# Initialize RAG engine as global
rag_engine: Optional[QueenRAGEngine] = None


def get_rag_engine() -> QueenRAGEngine:
    """Get RAG engine instance."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return rag_engine


def set_rag_engine(engine: QueenRAGEngine) -> None:
    """Set the RAG engine instance."""
    global rag_engine
    rag_engine = engine
