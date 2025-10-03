# RAG Project Guidelines

Mandatory rules for Claude and all agents when working on this RAG application.

## Core Development Principles

**KISS (Keep It Simple, Stupid)**
- Use Assistant UI's built-in chat - don't build custom
- Use Tailwind utilities - don't write custom CSS
- Use existing libraries - don't reinvent the wheel
- One function, one purpose

**YAGNI (You Aren't Gonna Need It)**
- Don't add features "just in case"
- Don't optimize prematurely
- Don't abstract until you have 3+ use cases
- Build only what's requested

**DRY (Don't Repeat Yourself)**
- Extract common patterns after 3rd occurrence
- Use environment variables for all config
- Centralize error handling
- Share types between frontend/backend

**SOLID for RAG**
- Single responsibility: One agent, one expertise
- Open/closed: Extend via config, don't modify core
- Interface segregation: Minimal API surfaces
- Dependency inversion: Depend on abstractions (types)

## Agent Assignment Rules

**ALWAYS use the correct specialist agent:**
- `rag_backend` → API endpoints, document processing, vector search
- `rag_frontend` → Chat UI, SSE handlers, upload components
- `vector_specialist` → Embeddings, chunking, similarity search
- `cheeky_backend` → Python validation (MyPy, Ruff)
- `cheeky_frontend` → Frontend validation (ESLint, CSS rules)
- `debuggy` → Fix errors in streaming, embeddings, uploads

**NEVER use generic agents when specialists exist.**

## Non-Negotiable Technical Rules

### Backend MUST:
- Use async/await for ALL I/O operations
- Include type hints on EVERY function
- Pass MyPy strict mode with ZERO errors
- Pass Ruff with ZERO warnings
- Validate embedding dimensions match model output
- Handle OpenAI rate limits with exponential backoff
- Store document metadata (source, chunk_index, timestamp)

### Frontend MUST:
- Use Assistant UI components ONLY (no custom chat UI)
- NEVER use !important in CSS
- Use Tailwind utilities ONLY (no custom CSS)
- Pass TypeScript strict mode
- Accumulate SSE text properly
- Handle connection drops with retry

### Vector Operations MUST:
- Chunk at 1000 tokens with 200 overlap
- Use OpenAI text-embedding-3-small (default) or text-embedding-3-large
- Apply cosine similarity with 0.7 threshold
- Validate all embeddings before storage
- Include metadata for traceability

## Forbidden Practices

**NEVER:**
- Store API keys in code (use .env only)
- Send entire documents to LLM (respect 8192 token limit)
- Mix embedding models in same collection
- Create custom chat components (Assistant UI has them)
- Use hardcoded colors (use theme variables)
- Skip error handling in async operations
- Allow unlimited file uploads without validation

## Quality Gates

**Before ANY commit:**
1. `cd backend && mypy . --strict` → 0 errors
2. `cd backend && ruff check .` → 0 warnings
3. `cd frontend && npm run type-check` → passes
4. `cd frontend && npm run lint` → 0 warnings
5. Test: upload document → ask question → get response with sources

**Performance requirements:**
- Embedding generation < 100ms
- Vector search < 200ms
- First streaming token < 1s
- Document indexing < 10s/MB

---

**CRITICAL:** These are requirements, not suggestions. Non-compliance = task failure.