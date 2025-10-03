---
name: start-rag
description: Install dependencies and start both backend and frontend servers
---

# 🚀 /start-rag - Start Your RAG Application

Install all dependencies and start both servers. Simple and fast.

## Setup Process

### 1. Check Environment

**Create .env files if missing:**
```bash
# Copy .env.example to .env in both directories
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

**Add your OpenAI API key to backend/.env:**
```
OPENAI_API_KEY=sk-your-key-here
```
Get your key from: https://platform.openai.com/api-keys

### 2. Install Backend Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 4. Start Both Servers

**Start backend (Terminal 1):**
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python main.py
```

**Start frontend (Terminal 2):**
```bash
cd frontend
npm run dev
```

## Success

✅ **Backend**: http://localhost:8000/docs
✅ **Frontend**: http://localhost:3000
✅ **Test**: Upload a document and ask questions about it

## Troubleshooting

**Dependencies fail?**
- Backend: `pip install -r requirements.txt --force-reinstall`
- Frontend: `rm -rf node_modules package-lock.json && npm install`

**Ports busy?** Kill existing processes or change ports in code.

**API key missing?** Add OPENAI_API_KEY to backend/.env

---

**Install dependencies for both backend and frontend, then start both servers in parallel terminals.**