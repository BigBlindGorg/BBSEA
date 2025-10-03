from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class RAGSettings(BaseSettings):
    """
    RAGLight and OpenAI configuration settings.
    """

    # OpenAI Settings
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", validation_alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL")

    # Chroma Settings
    chroma_persist_directory: str = Field(default="./storage/chroma_db", validation_alias="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="queen_rag_collection", validation_alias="CHROMA_COLLECTION_NAME")

    # RAG Settings
    rag_chunk_size: int = Field(default=1000, validation_alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=200, validation_alias="RAG_CHUNK_OVERLAP")
    rag_top_k_results: int = Field(default=5, validation_alias="RAG_TOP_K_RESULTS")
    rag_similarity_threshold: float = Field(default=0.7, validation_alias="RAG_SIMILARITY_THRESHOLD")

    # Document Settings
    upload_directory: str = Field(default="./storage/documents", validation_alias="UPLOAD_DIRECTORY")
    max_file_size_mb: int = Field(default=50, validation_alias="MAX_FILE_SIZE_MB")

    # System Settings
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "allow"
    }


# Create singleton instance
settings = RAGSettings()


def get_raglight_config() -> dict[str, Any]:
    """
    Get RAGLight configuration for pipeline initialization.
    """
    # RAGLight uses a single config object with all settings
    config = {
        "llm_provider": "openai",
        "llm_model": settings.openai_model,
        "embedding_provider": "openai",
        "embedding_model": settings.openai_embedding_model,
        "chunk_size": settings.rag_chunk_size,
        "chunk_overlap": settings.rag_chunk_overlap,
        "top_k": settings.rag_top_k_results,
        "temperature": 0.7,
        "max_tokens": 2000,
        "vector_store_type": "chroma",
        "persist_directory": settings.chroma_persist_directory,
        "collection_name": settings.chroma_collection_name
    }

    return config
