---
name: check
description: Verify RAG application quality - frontend, backend, and vector operations
---

**THIS IS NOT A REPORTING TASK - THIS IS A FIXING TASK!**

# ✅ /check - RAG System Quality Verification

**🚨 CRITICAL PARALLEL EXECUTION REQUIREMENT:**
This command MUST spawn ALL agents IN PARALLEL using single messages with multiple Task tool invocations.
NEVER spawn agents sequentially - this defeats the purpose of comprehensive quality verification.

## RAG-Specific Quality Checks

### Frontend Checks (Next.js + Assistant UI)
```bash
cd frontend
npm run lint        # ESLint with zero warnings
npm run type-check  # TypeScript strict mode
npm run build       # Production build succeeds
```

### Backend Checks (FastAPI + ChromaDB)
```bash
cd backend
source venv/bin/activate
mypy . --strict               # Type checking
ruff check .                  # Python linting
pytest tests/ -v              # All tests pass
python -m pytest tests/test_rag.py  # RAG pipeline tests
```

### Vector Database Checks
- Verify embedding dimensions (1536 for OpenAI)
- Test similarity search accuracy
- Check duplicate detection
- Validate metadata preservation

## Quality Verification & Fixing

**CRITICAL: Use PARALLEL spawning - ALL agents in ONE message:**
```
"I found issues across the RAG system. Spawning multiple agents IN PARALLEL to fix all issues:
- rag_backend: Fix FastAPI/Python issues in backend API
- rag_frontend: Fix React/Assistant UI issues in frontend
- vector_specialist: Fix ChromaDB and embedding issues
- debuggy: Fix SSE streaming and connection issues

Spawning all agents simultaneously now..."
[Use single message with multiple Task tool invocations]
```

## RAG-Specific Validation

**CRITICAL: Spawn ALL validation agents IN PARALLEL:**
```
"Spawning agents IN PARALLEL to validate RAG functionality:
- rag_backend: Validate API endpoints and RAG pipeline
- rag_frontend: Validate chat interface and SSE streaming
- vector_specialist: Validate embeddings and search quality
- cheeky_backend: Validate Python code standards
- cheeky_frontend: Validate CSS and Assistant UI usage

Spawning all validation agents simultaneously now..."
[MUST use single message with multiple Task tool invocations]
```

## Success Criteria

**YOU ARE NOT DONE UNTIL:**
- ✅ Frontend: Zero ESLint warnings, TypeScript passes
- ✅ Backend: Zero MyPy errors, Ruff passes
- ✅ Chat interface sends and receives messages
- ✅ Document upload works (PDF, TXT, MD, DOCX)
- ✅ Vector search returns relevant results
- ✅ SSE streaming works without disconnections
- ✅ No !important in CSS, uses Tailwind utilities
- ✅ Assistant UI components used (not custom)
- ✅ API keys properly configured in .env

## RAG-Specific Issues to Fix

**Common Frontend Issues:**
- SSE connection drops → Fix in assistant-provider.tsx
- Message accumulation bugs → Fix streaming handler
- Attachment upload failures → Check base64 encoding
- Assistant UI misuse → Use built-in components

**Common Backend Issues:**
- Embedding dimension errors → Must be 1536
- ChromaDB connection failures → Check initialization
- OpenAI rate limiting → Add retry logic
- Document chunking failures → Fix text splitter

**Common Vector Issues:**
- Poor retrieval accuracy → Tune similarity threshold
- Duplicate documents → Implement deduplication
- Missing metadata → Preserve source info
- Slow searches → Optimize collection size

## Failure Response Protocol

**When issues are found:**
1. **IMMEDIATELY SPAWN AGENTS IN PARALLEL** to fix issues simultaneously
2. **FIX EVERYTHING** - address EVERY issue, no matter how "minor"
3. **VERIFY** - re-run all checks after fixes
4. **TEST END-TO-END** - Upload doc → Ask question → Get answer
5. **NO STOPPING** - keep working until ALL checks show ✅ GREEN

## Final RAG System Test

1. **Upload Test**: Upload a sample document
2. **Embedding Test**: Verify document is chunked and embedded
3. **Search Test**: Ask a question about the document
4. **Response Test**: Verify answer includes source citations
5. **Stream Test**: Confirm response streams character-by-character

---

**Core principle: Fix everything until the complete RAG pipeline works flawlessly.**
