from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from openai import AsyncOpenAI

from .config import settings
from .document_parser import DocumentParser

logger = logging.getLogger(__name__)


class QueenRAGEngine:
    """
    Simplified RAG engine using ChromaDB directly with OpenAI embeddings.
    """

    def __init__(self) -> None:
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        # Initialize ChromaDB with OpenAI embeddings
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory
        )

        # Create embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(  # type: ignore[attr-defined]
            api_key=settings.openai_api_key,
            model_name=settings.openai_embedding_model
        )

        # Collection will be initialized in async initialize()
        self.collection: chromadb.Collection | None = None

        # Track loaded documents
        self.loaded_documents: set[str] = set()

        logger.info("QueenRAGEngine initialized")

    async def initialize(self) -> None:
        """
        Async initialization of the RAG engine.
        """
        try:
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.chroma_collection_name,
                embedding_function=self.embedding_function
            )

            # Load metadata of existing documents
            await self._load_document_metadata()

            logger.info(f"RAG engine initialized with {len(self.loaded_documents)} documents")

        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise

    async def _load_document_metadata(self) -> None:
        """
        Load metadata about existing documents in the knowledge base.
        """
        doc_path = Path(settings.upload_directory)
        if doc_path.exists():
            for file_path in doc_path.iterdir():
                if file_path.is_file() and not file_path.name.endswith('.meta.json'):
                    self.loaded_documents.add(file_path.name)

    async def add_document(self, file_path: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Add a new document to the RAG knowledge base.
        """
        try:
            file_path_obj = Path(file_path)
            file_name = file_path_obj.name

            # Check if document already exists
            if file_name in self.loaded_documents:
                logger.warning(f"Document {file_name} already exists in knowledge base")
                return {
                    "status": "exists",
                    "filename": file_name,
                    "message": "Document already in knowledge base"
                }

            # Parse document content based on file type
            content = DocumentParser.parse_document(file_path)

            # Split content into chunks
            chunks = self._split_text(content)

            # Add to ChromaDB
            ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "filename": file_name,
                    "chunk": i,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                }
                for i in range(len(chunks))
            ]

            if self.collection is None:
                raise RuntimeError("Collection not initialized")
            self.collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas  # type: ignore[arg-type]
            )

            # Add to tracked documents
            self.loaded_documents.add(file_name)

            # Store metadata if provided
            if metadata:
                metadata_path = file_path_obj.with_suffix('.meta.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)

            logger.info(f"Successfully added document: {file_name}")

            return {
                "status": "success",
                "filename": file_name,
                "chunks": len(chunks),
                "message": "Document added to knowledge base"
            }

        except Exception as e:
            logger.error(f"Failed to add document {file_path}: {e}")
            raise

    def _split_text(self, text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
        """
        Split text into chunks.
        """
        chunk_size = chunk_size or settings.rag_chunk_size
        chunk_overlap = chunk_overlap or settings.rag_chunk_overlap

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap

        return chunks

    async def remove_document(self, filename: str) -> dict[str, Any]:
        """
        Remove a document from the knowledge base.
        """
        try:
            file_path = Path(settings.upload_directory) / filename

            if filename not in self.loaded_documents:
                return {
                    "status": "not_found",
                    "filename": filename,
                    "message": "Document not found in knowledge base"
                }

            # Remove from ChromaDB
            if self.collection is None:
                raise RuntimeError("Collection not initialized")
            results = self.collection.get(
                where={"filename": filename}
            )

            if results['ids']:
                self.collection.delete(ids=results['ids'])

            # Remove the file
            if file_path.exists():
                file_path.unlink()

            # Remove metadata if exists
            metadata_path = file_path.with_suffix('.meta.json')
            if metadata_path.exists():
                metadata_path.unlink()

            # Remove from tracked documents
            self.loaded_documents.discard(filename)

            logger.info(f"Successfully removed document: {filename}")

            return {
                "status": "success",
                "filename": filename,
                "message": "Document removed from knowledge base"
            }

        except Exception as e:
            logger.error(f"Failed to remove document {filename}: {e}")
            raise

    async def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Search for relevant documents using semantic search.
        """
        try:
            top_k = top_k or settings.rag_top_k_results

            # Query ChromaDB - this creates embeddings and performs vector search
            if self.collection is None:
                raise RuntimeError("Collection not initialized")
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            # Format results
            formatted_results = []
            if (results['documents'] and results['documents'][0] and
                results['metadatas'] and results['metadatas'][0] and
                results['distances'] and results['distances'][0]):
                # Python 3.9 doesn't support strict parameter in zip()
                for idx, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity_score = 1.0 - (distance / 2.0)  # Convert distance to similarity score
                    formatted_results.append({
                        "index": idx,
                        "content": doc,
                        "metadata": metadata,
                        "score": similarity_score
                    })

            # Log search results with top-K info
            logger.debug(f"Search query: '{query}' returned top {len(formatted_results)} most similar results")

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            raise

    async def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        use_rag: bool = True,
        stream: bool = True,
        images: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[str]:
        """
        Chat with the AI using RAG-enhanced context.
        """
        try:
            history = history or []

            # Build messages list
            messages = []

            # Add system message
            system_prompt = (
                "You are Queen-RAG, an intelligent AI assistant with access to a specialized knowledge base. "
                "Your responses should be accurate, helpful, and grounded in the provided context.\n\n"
                "Key behaviors:\n"
                "1. When answering from the knowledge base, always cite your sources using [Source: filename] format\n"
                "2. Distinguish clearly between information from the knowledge base and your general knowledge\n"
                "3. If the knowledge base doesn't contain relevant information, explicitly state this before providing general knowledge\n"
                "4. Structure complex answers with clear sections and bullet points for readability\n"
                "5. Be direct and concise - lead with the answer, then provide supporting details\n"
                "6. When multiple sources conflict, acknowledge the discrepancy and present all viewpoints\n"
                "7. For ambiguous queries, clarify what you're interpreting before answering\n"
                "8. Use confidence indicators in your language ('clearly states', 'suggests', 'appears to')\n\n"
                "Remember: Your primary value is providing accurate, well-sourced information from the knowledge base "
                "while being transparent about the limitations and gaps in available information."
            )
            messages.append({"role": "system", "content": system_prompt})

            # Add RAG context if enabled
            if use_rag and self.collection:
                # Perform vector search to find relevant context
                context_results = await self.search(message, top_k=settings.rag_top_k_results)

                if context_results:
                    # Build context from retrieved chunks with improved formatting
                    context_parts = []
                    for idx, r in enumerate(context_results, 1):
                        filename = r['metadata'].get('filename', 'Unknown')
                        chunk_num = r['metadata'].get('chunk', 0) + 1
                        total_chunks = r['metadata'].get('total_chunks', 'Unknown')
                        score = r.get('score', 0)

                        # Include all top-K results with similarity scores for LLM to evaluate
                        context_parts.append(
                            f"[Document {idx}: {filename} | Section {chunk_num}/{total_chunks} | Relevance: {score:.2f}]\n"
                            f"{r['content']}"
                        )

                    if context_parts:
                        context_text = "\n\n---\n\n".join(context_parts)
                        context_message = (
                            f"# Knowledge Base Context\n\n"
                            f"Here are the top {len(context_parts)} most similar sections from your documents (ranked by relevance score):\n\n"
                            f"{context_text}\n\n"
                            f"---\n\n"
                            f"Use the above context to answer the user's question. Evaluate the relevance scores and focus on the most relevant sections. "
                            f"If none of the sections are sufficiently relevant to the question, say so and provide a general response. "
                            f"Always cite specific documents when referencing information from the context."
                        )
                        messages.append({"role": "system", "content": context_message})

            # Add chat history
            messages.extend(history)

            # Add current message (with images if present)
            if images:
                # For vision models, create message with text and images
                user_content = [{"type": "text", "text": message}]
                user_content.extend(images)
                messages.append({"role": "user", "content": user_content})  # type: ignore[dict-item]
                logger.info(f"Processing message with {len(images)} image(s)")
                logger.debug(f"Message structure for vision: {len(user_content)} content parts")
            else:
                messages.append({"role": "user", "content": message})

            # Get response from OpenAI
            # gpt-4o has built-in vision capabilities
            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,  # type: ignore
                stream=stream,
                temperature=0.7,
                max_tokens=2000
            )

            if stream:
                # Stream response chunks
                async for chunk in response:  # type: ignore
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                # Return complete response
                yield response.choices[0].message.content  # type: ignore

        except Exception as e:
            logger.error(f"Chat failed for message '{message}': {e}")
            raise

    async def get_document_list(self) -> list[dict[str, Any]]:
        """
        Get list of all documents in the knowledge base.
        """
        documents = []
        doc_path = Path(settings.upload_directory)

        if doc_path.exists():
            for file_path in doc_path.iterdir():
                if file_path.is_file() and not file_path.name.endswith('.meta.json'):
                    # Get file stats
                    stats = file_path.stat()

                    # Load metadata if exists
                    metadata = {}
                    metadata_path = file_path.with_suffix('.meta.json')
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)

                    documents.append({
                        "filename": file_path.name,
                        "size": stats.st_size,
                        "modified": stats.st_mtime,
                        "type": file_path.suffix[1:] if file_path.suffix else "unknown",
                        "metadata": metadata
                    })

        return documents

    async def cleanup(self) -> None:
        """
        Cleanup resources when shutting down.
        """
        try:
            # ChromaDB handles its own cleanup
            logger.info("QueenRAGEngine cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def health_check(self) -> dict[str, Any]:
        """
        Check health status of the RAG engine.
        """
        return {
            "status": "healthy" if self.collection else "unhealthy",
            "documents_count": len(self.loaded_documents),
            "vector_store": "chroma",
            "model": settings.openai_model,
            "embedding_model": settings.openai_embedding_model,
            "initialized": self.collection is not None
        }
