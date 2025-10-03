import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dependencies import get_rag_engine
from rag.rag_engine_simple import QueenRAGEngine

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    history: list[ChatMessage] = Field(default=[], description="Chat history")
    use_rag: bool = Field(default=True, description="Whether to use RAG context")
    stream: bool = Field(default=True, description="Whether to stream the response")
    top_k: Optional[int] = Field(default=None, description="Number of RAG results to use")
    attachments: Optional[list[dict[str, Any]]] = Field(default=None, description="File attachments")


class ChatResponse(BaseModel):
    """Chat response model for non-streaming responses."""
    response: str = Field(..., description="AI response")
    context_used: bool = Field(..., description="Whether RAG context was used")
    sources: list[str] = Field(default=[], description="Source documents used")


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(default=5, description="Number of results to return")


class SearchResponse(BaseModel):
    """Search response model."""
    results: list[dict[str, Any]] = Field(..., description="Search results")
    count: int = Field(..., description="Number of results")


@router.post("/message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> Union[ChatResponse, StreamingResponse]:
    """
    Send a chat message and get a response.
    Supports both streaming and non-streaming responses.
    """
    try:
        # Convert history to dict format
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

        if request.stream:
            # Return streaming response
            async def generate() -> AsyncIterator[str]:
                try:
                    async for chunk in rag_engine.chat(
                        message=request.message,
                        history=history,
                        use_rag=request.use_rag,
                        stream=True
                    ):
                        # Format as Server-Sent Events
                        yield f"data: {json.dumps({'content': chunk})}\n\n"

                    # Send done signal
                    yield f"data: {json.dumps({'done': True})}\n\n"

                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"  # Disable Nginx buffering
                }
            )
        else:
            # Return complete response
            response_text = ""
            async for chunk in rag_engine.chat(
                message=request.message,
                history=history,
                use_rag=request.use_rag,
                stream=False
            ):
                response_text += chunk

            # Get sources if RAG was used
            sources = []
            if request.use_rag:
                # Extract source documents from the last search
                # This is a simplified approach - you might need to enhance the RAG engine
                # to properly track and return sources
                sources = ["README.md", "daddy_project.md"]  # Placeholder for now

            return ChatResponse(
                response=response_text,
                context_used=request.use_rag,
                sources=sources
            )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}") from e


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> SearchResponse:
    """
    Search the knowledge base for relevant documents.
    """
    try:
        results = await rag_engine.search(
            query=request.query,
            top_k=request.top_k
        )

        return SearchResponse(
            results=results,
            count=len(results)
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    rag_engine: QueenRAGEngine = Depends(get_rag_engine)
) -> StreamingResponse:
    """
    Stream chat response using Server-Sent Events.
    This is a dedicated streaming endpoint for Assistant UI compatibility.
    """
    try:
        # Convert history to dict format
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

        # Handle attachments if present
        attachment_context = ""
        image_attachments = []

        logger.info(f"Received request with attachments: {bool(request.attachments)}")
        if request.attachments:
            logger.info(f"Number of attachments: {len(request.attachments)}")
            logger.info(f"Attachment types: {[a.get('type') for a in request.attachments]}")
            for attachment in request.attachments:
                attachment_type = attachment.get('type', 'document')
                filename = attachment.get('name', 'attachment')
                file_content = attachment.get('content', '')

                # Handle images for vision
                if attachment_type == 'image':
                    # Store image data for vision model
                    image_attachments.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': file_content  # Base64 data URL
                        }
                    })
                    logger.info(f"Added image attachment: {filename}")
                # Handle documents
                elif attachment_type == 'document' or attachment_type == 'file':
                    # Add to temporary context for this conversation
                    attachment_context += f"\n\n[Attached File: {filename}]\n{file_content[:2000]}"  # Limit preview
                    logger.info(f"Added document attachment: {filename}")

        # Build enhanced message with attachment context
        enhanced_message = request.message
        if attachment_context:
            enhanced_message = f"{request.message}\n\nAttached Files Context:{attachment_context}"

        async def generate() -> AsyncIterator[str]:
            try:
                # Send initial connection message
                yield f"data: {json.dumps({'type': 'start'})}\n\n"

                # Stream the response (with images if present)
                if image_attachments:
                    logger.info(f"Sending {len(image_attachments)} image(s) to chat engine")
                    logger.debug(f"Image attachment format: {image_attachments[0].keys() if image_attachments else 'None'}")

                async for chunk in rag_engine.chat(
                    message=enhanced_message,
                    history=history,
                    use_rag=request.use_rag,
                    stream=True,
                    images=image_attachments if image_attachments else None
                ):
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

                # Send completion message
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}") from e


@router.get("/health")
async def chat_health() -> dict[str, str]:
    """
    Check chat API health.
    """
    return {"status": "healthy", "service": "chat"}
