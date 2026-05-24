# HR Policy RAG Chatbot 🤝

An AI-powered HR Policy Assistant that answers employee questions grounded in your company's official HR documents. Built with a full RAG (Retrieval-Augmented Generation) pipeline, FastAPI backend, Streamlit UI, and Docker deployment.

---

## ✨ Features

- **Accurate answers** — grounded in your HR policy documents, not hallucinated
- **Source citations** — every answer shows which document and page it came from
- **Multi-turn conversations** — remembers context across follow-up questions
- **REST API** — FastAPI backend for integration with other systems
- **Beautiful UI** — dark glassmorphism Streamlit interface
- **Fast** — Llama 3.1 via Groq API (free, very low latency)
- **Fully local embeddings** — sentence-transformers running on CPU (no GPU needed)
- **Docker ready** — single command to run everything

---

## 🏗️ Architecture

```
Employee Question
      │
      ▼
  Streamlit UI (app.py)
      │
      ▼
  FastAPI (api.py)
      │
      ▼
  Agent (agents.py) ──── Intent Detection
      │
      ▼
  Tools (tools.py)
      │
      ▼
  RAG Pipeline (rag.py)
      │
   ┌──┴──┐
   │     │
   ▼     ▼
ChromaDB  sentence-transformers
(vectors) (CPU embeddings)
              │
              ▼
         Groq API → Llama 3.1 8B
              │
              ▼
         Answer + Sources
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.1 8B via [Groq API](https://console.groq.com) (free) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (CPU, local) |
| Vector Store | ChromaDB (persistent, local) |
| RAG Framework | Custom pipeline (no LangChain dependency) |
| API | FastAPI + Uvicorn |
| UI | Streamlit (dark glassmorphism theme) |
| Containerization | Docker + docker-compose |
| Cloud Deploy | AWS ECS Fargate + ECR + EFS |
| Document Loaders | pypdf, python-docx |

---

## 📁 Project Structure

```
hr-rag-chatbot/
│
├── app.py              # Streamlit UI entry point
├── api.py              # FastAPI REST API
├── agents.py           # LLM orchestration + intent detection
├── tools.py            # Search + utility tools
├── rag.py              # Embeddings + ChromaDB vector store
├── ingest.py           # Document ingestion pipeline
├── prompts.py          # LLM prompt templates
├── models.py           # Internal Pydantic schemas
├── schemas.py          # API request/response schemas
├── config.py           # Centralized configuration
│
├── Dockerfile          # Multi-stage Docker build
├── docker-compose.yml  # Local multi-service setup
├── .dockerignore       # Docker build exclusions
├── requirements.txt    # Python dependencies
│
├── data/
│   └── hr_policies/    # ← Put your HR PDFs/DOCX here
│
├── vectorstore/        # ChromaDB persisted data (auto-created)
│
└── infra/
    ├── ecs-task-api.json        # AWS ECS task definition
    ├── ecs-task-streamlit.json  # AWS ECS Streamlit task
    ├── buildspec.yml            # AWS CodeBuild CI/CD
    └── deploy.sh                # Step-by-step AWS deploy script
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Groq API key](https://console.groq.com/keys) (free)
- HR policy documents (PDF, DOCX, TXT, or MD)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/hr-rag-chatbot.git
cd hr-rag-chatbot
```

### 2. Set up environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Required in `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Add HR documents
```bash
mkdir -p data/hr_policies
cp /path/to/your/*.pdf data/hr_policies/
```

### 5. Ingest documents
```bash
python ingest.py

# Expected output:
# ✅ leave_policy.pdf          48 chunks
# ✅ code_of_conduct.pdf       61 chunks
# ✅ travel_reimbursement.docx 33 chunks
```

### 6. Run the app

**Streamlit UI:**
```bash
streamlit run app.py
# Open http://localhost:8501
```

**FastAPI:**
```bash
uvicorn api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

---

## 🐳 Docker

### Run with Docker Compose
```bash
# Start all services
docker-compose up --build

# Services:
# Streamlit UI  → http://localhost:8501
# FastAPI docs  → http://localhost:8000/docs
# ChromaDB      → http://localhost:8002
```

### Run specific services only
```bash
# API + ChromaDB only (saves resources)
docker-compose up --build api chromadb

# Re-ingest documents
docker-compose run --rm ingest
```

### Useful Docker commands
```bash
# View logs
docker-compose logs -f api

# Stop everything
docker-compose down

# Stop and wipe all data
docker-compose down -v

# Check image size
docker images hr-rag-chatbot
```

---

## 🌐 API Reference

### `POST /chat`
Ask an HR policy question.

**Request:**
```json
{
  "question": "How many casual leaves am I entitled to per year?",
  "session_id": "optional-for-multi-turn"
}
```

**Response:**
```json
{
  "answer": "You are entitled to 12 casual leaves per year...",
  "session_id": "abc-123",
  "message_type": "answer",
  "sources": [
    {
      "source_file": "leave_policy.pdf",
      "page_number": 3,
      "similarity_score": 0.92,
      "excerpt": "Employees are entitled to 12 casual leaves..."
    }
  ],
  "source_files": ["leave_policy.pdf"],
  "latency_ms": 820.5
}
```

### `POST /ingest`
Trigger document ingestion.

```json
{ "reset": false, "filename": "leave_policy.pdf" }
```

### `GET /health`
Health check (used by AWS ALB).

### `GET /stats`
Vector store statistics.

---

## ☁️ AWS Deployment

Deploy to AWS ECS Fargate (serverless containers):

```bash
# Set your AWS details
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=ap-south-1

# Follow step-by-step guide
cat infra/deploy.sh
```

**AWS services used:**

| Service | Purpose | Est. Cost |
|---|---|---|
| ECS Fargate | Run containers | ~$5-8/mo |
| ECR | Store Docker images | ~$0.10/GB/mo |
| EFS | Persistent vector store | ~$0.30/GB/mo |
| Secrets Manager | Store API keys | ~$0.40/secret/mo |
| CloudWatch | Logs | ~$0.50/GB |
| **Total** | | **~$8-12/mo** |

---

## 🔧 Configuration

All settings in `.env`:

```env
# LLM
GROQ_API_KEY=your_key_here
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1024

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store
VECTOR_STORE_TYPE=chroma
VECTOR_STORE_PATH=./vectorstore

# Documents
HR_DOCS_PATH=./data/hr_policies
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Retrieval
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.75

# App
APP_ENV=development
LOG_LEVEL=INFO
```

**Alternative LLM models on Groq (update `LLM_MODEL`):**

| Model | Speed | Quality |
|---|---|---|
| `llama-3.1-8b-instant` | ⚡ Fastest | Good (default) |
| `llama-3.3-70b-versatile` | Medium | Best |
| `mixtral-8x7b-32768` | Fast | Good, 32k context |
| `gemma2-9b-it` | Fast | Good |

---

## 📄 Supported Document Formats

| Format | Extension | Notes |
|---|---|---|
| PDF | `.pdf` | Page numbers preserved in citations |
| Word | `.docx` | Paragraphs extracted |
| Text | `.txt` | Plain text |
| Markdown | `.md` | Treated as plain text |

---

## 🔄 Re-ingesting Documents

When you update or add HR policy documents:

```bash
# Add new documents and re-ingest all
python ingest.py

# Wipe everything and start fresh
python ingest.py --reset

# Ingest a single file
python ingest.py --file leave_policy_2024.pdf
```

---

## 🤔 How RAG Works

```
1. INGEST (one-time):
   PDF → extract text → split into chunks
       → embed with sentence-transformers
       → store vectors in ChromaDB

2. QUERY (every question):
   Question → embed → similarity search in ChromaDB
            → retrieve top-5 relevant chunks
            → inject chunks into LLM prompt
            → Llama 3.1 generates grounded answer
            → return answer + source citations
```

---

## 🛡️ Security Notes

- Never commit `.env` to git (it's in `.gitignore`)
- Use AWS Secrets Manager for production API keys
- The `/ingest` endpoint should be protected in production
- HR documents may be sensitive — use private repos and private S3/EFS

---

## 🐛 Troubleshooting

**`GROQ_API_KEY is not set`**
```bash
# Check .env file exists and has the key
cat .env | grep GROQ
```

**`Vector store is empty`**
```bash
# Run ingestion first
python ingest.py
```

**`No relevant HR policy found`**
```bash
# Check documents were ingested
python -c "from rag import get_collection_stats; print(get_collection_stats())"
# Lower similarity threshold in .env
SIMILARITY_THRESHOLD=0.5
```

**Docker build slow / out of space**
```bash
# Clean Docker cache
docker builder prune -af
docker system prune -af
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) — ultra-fast LLM inference
- [Meta](https://ai.meta.com) — Llama 3.1 open source model
- [ChromaDB](https://www.trychroma.com) — vector database
- [sentence-transformers](https://www.sbert.net) — embeddings
- [Streamlit](https://streamlit.io) — UI framework
- [FastAPI](https://fastapi.tiangolo.com) — REST API framework
