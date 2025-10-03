from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dependencies import get_rag_engine
from rag.config import settings
from rag.rag_engine_simple import QueenRAGEngine

logger = logging.getLogger(__name__)
router = APIRouter()


class DocumentInfo(BaseModel):
    """Document information model."""
    filename: str
    size: int
    type: str
    modified: float
    metadata: dict[str, Any] = {}


class DocumentListResponse(BaseModel):
    """Document list response model."""
    documents: list[DocumentInfo]
    count: int


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    status: str
    filename: str
    message: str
    size: int | None = None


class DocumentDeleteRequest(BaseModel):
    """Document deletion request model."""
    filename: str = Field(..., description="Name of the file to delete")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    description: str | None = Form(None),
    tags: str | None = Form(None),
    rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> DocumentUploadResponse:
    """
    Upload a document to the knowledge base.
    Supports PDF, DOCX, TXT, MD, and other text-based files.
    """
    try:
        # Validate filename
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required"
            )

        # Check file size
        file_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        # Create temporary file to check size
        temp_path = Path(settings.upload_directory) / f"temp_{file.filename}"

        async with aiofiles.open(temp_path, 'wb') as f:
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                if file_size > settings.max_file_size_mb * 1024 * 1024:
                    await f.close()
                    temp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size exceeds maximum of {settings.max_file_size_mb}MB"
                    )
                await f.write(chunk)

        # Move to final location
        final_path = Path(settings.upload_directory) / file.filename

        # Check if file already exists
        if final_path.exists():
            temp_path.unlink()
            return DocumentUploadResponse(
                status="exists",
                filename=file.filename,
                message="Document already exists in knowledge base",
                size=file_size
            )

        # Move temp file to final location
        shutil.move(str(temp_path), str(final_path))

        # Prepare metadata
        metadata: dict[str, Any] = {
            "original_filename": file.filename,
            "content_type": file.content_type,
            "size": file_size
        }

        if description:
            metadata["description"] = description

        if tags:
            metadata["tags"] = [tag.strip() for tag in tags.split(",")]

        # Add to RAG engine
        result = await rag_engine.add_document(
            file_path=str(final_path),
            metadata=metadata
        )

        return DocumentUploadResponse(
            status=result["status"],
            filename=file.filename,
            message=result["message"],
            size=file_size
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error for {file.filename}: {e}")
        # Clean up on error
        if 'temp_path' in locals():
            Path(temp_path).unlink(missing_ok=True)
        if 'final_path' in locals():
            Path(final_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}") from e


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> DocumentListResponse:
    """
    List all documents in the knowledge base.
    """
    try:
        documents = await rag_engine.get_document_list()

        # Convert to DocumentInfo models
        doc_infos = [
            DocumentInfo(
                filename=doc["filename"],
                size=doc["size"],
                type=doc["type"],
                modified=doc["modified"],
                metadata=doc.get("metadata", {})
            )
            for doc in documents
        ]

        return DocumentListResponse(
            documents=doc_infos,
            count=len(doc_infos)
        )

    except Exception as e:
        logger.error(f"List error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}") from e


@router.delete("/delete", response_model=DocumentUploadResponse)
async def delete_document(
    request: DocumentDeleteRequest,
    rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> DocumentUploadResponse:
    """
    Delete a document from the knowledge base.
    """
    try:
        result = await rag_engine.remove_document(request.filename)

        return DocumentUploadResponse(
            status=result["status"],
            filename=request.filename,
            message=result["message"]
        )

    except Exception as e:
        logger.error(f"Delete error for {request.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}") from e


@router.get("/download/{filename}")
async def download_document(
    filename: str,
    _rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> FileResponse:
    """
    Download a document from the knowledge base.
    """
    try:
        file_path = Path(settings.upload_directory) / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}") from e


@router.post("/bulk-upload", response_model=list[DocumentUploadResponse])
async def bulk_upload_documents(
    files: list[UploadFile] = File(...),
    rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> list[DocumentUploadResponse]:
    """
    Upload multiple documents at once.
    """
    results = []

    for file in files:
        try:
            # Validate filename
            if not file.filename:
                results.append(DocumentUploadResponse(
                    status="error",
                    filename="unknown",
                    message="Filename is required",
                    size=0
                ))
                continue

            # Process each file
            file_size = 0
            temp_path = Path(settings.upload_directory) / f"temp_{file.filename}"

            async with aiofiles.open(temp_path, 'wb') as f:
                content = await file.read()
                file_size = len(content)

                # Check size
                if file_size > settings.max_file_size_mb * 1024 * 1024:
                    temp_path.unlink(missing_ok=True)
                    results.append(DocumentUploadResponse(
                        status="error",
                        filename=file.filename,
                        message=f"File size exceeds maximum of {settings.max_file_size_mb}MB",
                        size=file_size
                    ))
                    continue

                await f.write(content)

            # Move to final location
            final_path = Path(settings.upload_directory) / file.filename

            if final_path.exists():
                temp_path.unlink()
                results.append(DocumentUploadResponse(
                    status="exists",
                    filename=file.filename,
                    message="Document already exists",
                    size=file_size
                ))
                continue

            shutil.move(str(temp_path), str(final_path))

            # Add to RAG
            result = await rag_engine.add_document(
                file_path=str(final_path),
                metadata={
                    "original_filename": file.filename,
                    "content_type": file.content_type,
                    "size": file_size,
                    "bulk_upload": True
                }
            )

            results.append(DocumentUploadResponse(
                status=result["status"],
                filename=file.filename,
                message=result["message"],
                size=file_size
            ))

        except Exception as e:
            logger.error(f"Bulk upload error for {file.filename}: {e}")
            results.append(DocumentUploadResponse(
                status="error",
                filename=file.filename or "unknown",
                message=f"Upload failed: {str(e)}"
            ))

    return results


@router.get("/stats")
async def document_stats(
    rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> dict[str, Any]:
    """
    Get statistics about the document knowledge base.
    """
    try:
        documents = await rag_engine.get_document_list()

        # Calculate stats
        total_size = sum(doc["size"] for doc in documents)
        file_types: dict[str, int] = {}

        for doc in documents:
            file_type = doc["type"]
            if file_type in file_types:
                file_types[file_type] += 1
            else:
                file_types[file_type] = 1

        return {
            "total_documents": len(documents),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": file_types,
            "max_file_size_mb": settings.max_file_size_mb,
            "storage_path": settings.upload_directory
        }

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}") from e


@router.get("/health")
async def documents_health() -> dict[str, str]:
    """
    Check documents API health.
    """
    return {"status": "healthy", "service": "documents"}
