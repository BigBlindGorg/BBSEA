import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.chat import router as chat_router
from api.documents import router as document_router
from api.health import router as health_router
from dependencies import rag_engine as global_rag_engine
from dependencies import set_rag_engine

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Manage application lifecycle - initialize and cleanup resources.
    """
    # Startup
    logger.info("Starting Queen-RAG application...")

    # Create necessary directories
    Path("./storage/documents").mkdir(parents=True, exist_ok=True)
    Path("./storage/chroma_db").mkdir(parents=True, exist_ok=True)
    Path("./storage/logs").mkdir(parents=True, exist_ok=True)

    # Initialize RAG engine
    try:
        from rag.config import settings
        from rag.rag_engine_simple import QueenRAGEngine
        engine = QueenRAGEngine()
        await engine.initialize()
        set_rag_engine(engine)
        logger.info("RAG engine initialized successfully")
        logger.info(f"RAG top-K results: {settings.rag_top_k_results}, similarity threshold: {settings.rag_similarity_threshold} (unused for filtering)")
        logger.info(f"RAG chunk size: {settings.rag_chunk_size}, overlap: {settings.rag_chunk_overlap}")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Queen-RAG application...")
    if global_rag_engine:
        await global_rag_engine.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Queen-RAG API",
    description="A production-ready RAG system with Assistant UI",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
allowed_origins = [
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3002",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(document_router, prefix="/api/documents", tags=["documents"])
app.include_router(health_router, prefix="/api/health", tags=["health"])

# Root endpoint
@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "Queen-RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "documents": "/api/documents",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT", "development") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
