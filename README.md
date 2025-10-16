# RAG Chat Application

A production-ready Retrieval-Augmented Generation (RAG) application that ingests PDF documents, stores them in a PostgreSQL vector database (pgvector), and answers questions using OpenAI's GPT models with retrieved context.

## 📚 Documentation

- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Comprehensive architecture documentation with ASCII and Mermaid diagrams
- **[README.md](./README.md)** - This file (setup and usage guide)

## 🏗️ Architecture Overview

```
Browser UI → FastAPI → Ingestion Module → OpenAI API
                ↓                ↓
         PostgreSQL + pgvector
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed diagrams and component descriptions.

## 🚀 Quick Start (Docker - Recommended)

The fastest way to run the application:

```bash
# 1. Ensure Docker Desktop is running

# 2. Create .env file with your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 3. Start the stack (database + web server)
docker compose up --build -d

# 4. Open browser
open http://localhost:8000
```

That's it! The application is now running with:
- PostgreSQL + pgvector on port 5432
- FastAPI web server on port 8000
- Automatic database initialization

Notes on model selection in Docker
- By default the compose file mounts `./models` into the container at `/models` and also bind-mounts the repo into `/app`. This means local models are used from your host filesystem (no large models are baked into the image).
- To run the container with the OpenAI-backed LLM, set `LLM_PROVIDER=OPENAI` in your `.env` (and ensure `OPENAI_API_KEY` is set).
- To run the container with a local HF model, set `LLM_PROVIDER=LOCAL` and `LOCAL_LLM_MODEL=/models/<model-dir>` in `.env` (for example `/models/flan-t5-large`).

### Ingest a PDF

**Option 1: Via UI**
1. Open http://localhost:8000
2. Click "Choose File" and select a PDF
3. Click "Upload and ingest PDF"
4. Wait for confirmation message

**Option 2: Via API**
```bash
# Use default PDF from .env (PDF_PATH)
curl -X POST http://localhost:8000/ingest

# Or upload a specific file
curl -X POST http://localhost:8000/ingest \
  -F "pdf=@/path/to/your/document.pdf"
```

### Ask Questions

**Via UI**: Type your question in the chat box and press Enter

**Via API**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What does this document discuss?"}'
```

Testing expectations
- If `LLM_PROVIDER=LOCAL` you will see the local model produce answers (quality depends on model size). The container reads models from `/models` at runtime.
- If `LLM_PROVIDER=OPENAI` the container will call OpenAI (requires `OPENAI_API_KEY`).

## 📋 Requirements
## 📋 Requirements

- Docker & Docker Compose (recommended), or
- Python 3.11+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- PostgreSQL 15+ with pgvector extension (provided by Docker Compose)

## ⚙️ Configuration

All configuration is done via environment variables in `.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | Your OpenAI API key |
| `OPENAI_CHAT_MODEL` | No | `gpt-4o` | Model for answers (gpt-4o, gpt-4o-mini, etc.) |
| `EMBEDDING_PROVIDER` | No | `OPENAI` | `OPENAI` or `LOCAL` |
| `LLM_PROVIDER` | No | `OPENAI` | `OPENAI`, `LOCAL`, or `LLAMA` |
| `DATABASE_URL` | Yes | - | PostgreSQL connection string |
| `PGVECTOR_TABLE` | No | `documents_vectors` | Table name for vectors |
| `PDF_PATH` | No | - | Default PDF to ingest |

*Not required if using `LLM_PROVIDER=LOCAL`

### Example .env

```bash
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_CHAT_MODEL=gpt-4o
DATABASE_URL=postgresql://postgres:pass@db:5432/demo
EMBEDDING_PROVIDER=OPENAI
LLM_PROVIDER=OPENAI
PDF_PATH=./citus-doc-readthedocs-io-en-latest.pdf
PGVECTOR_TABLE=documents_vectors
```

## 🔧 Local Development Setup (Without Docker)

If you prefer running without Docker:

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL with pgvector
docker run --rm -d \
  -e POSTGRES_PASSWORD=pass \
  -e POSTGRES_DB=demo \
  -p 5432:5432 \
  ankane/pgvector

# 4. Configure .env
cp .env.example .env
# Edit .env and set:
# DATABASE_URL=postgresql://postgres:pass@localhost:5432/demo

# 5. Run the server
uvicorn backend.main:app --reload --port 8000
```

Using a local HF model (offline) in local dev

1. Download a model to `./models` (example uses `google/flan-t5-large`):

```bash
python scripts/download_model.py --model google/flan-t5-large --dest models/flan-t5-large
```

2. Set `.env` to use the local model:

```bash
EMBEDDING_PROVIDER=LOCAL
LLM_PROVIDER=LOCAL
LOCAL_LLM_MODEL=/models/flan-t5-large
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

3. Re-run the server and ask questions (same `/chat` endpoint).

Tip: If you change `EMBEDDING_PROVIDER` you'll need to re-ingest your PDFs so vector dimensions match the chosen embedder.

## 🎛️ Advanced Configuration

### Using Local Models (No OpenAI Required)
### Using Local Models (No OpenAI Required)

For completely offline operation:

```bash
# In .env
EMBEDDING_PROVIDER=LOCAL
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_PROVIDER=LOCAL
LOCAL_LLM_MODEL=google/flan-t5-small
```

**Notes**:
- Local embeddings produce 384-dimensional vectors (vs OpenAI's 1536)
- Must re-ingest PDFs when switching embedding providers
- Requires `transformers`, `torch`, `sentence-transformers` (included in requirements)

### Using Llama Models

```bash
# In .env
LLM_PROVIDER=LLAMA
LLAMA_MODEL_PATH=/path/to/model.gguf
```

Requires: `llama-cpp-python` (included in requirements-docker.txt)

## 🐛 Debugging

### View Logs

```bash
# All services
docker compose logs -f

# Just web server
docker compose logs web -f

# Just database
docker compose logs db -f
```

### Common Issues

**1. "tiktoken not found"**
- Fixed in latest version (included in requirements)

**2. "Vector dimension mismatch"**
- You switched embedding providers
- Solution: Clear DB and re-ingest
```bash
docker compose down -v  # Delete volumes
docker compose up --build -d
curl -X POST http://localhost:8000/ingest
```

**3. "OpenAI API key not set"**
- Verify `.env` has `OPENAI_API_KEY=sk-...`
- Restart containers: `docker compose restart web`

**4. "Database connection failed"**
- Ensure database is running: `docker compose ps`
- Check DATABASE_URL in `.env`

### Test Connection

```bash
# Test ingestion
curl -X POST http://localhost:8000/ingest

# Test chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Test question"}'
```

## 🏗️ Architecture

This application uses:

- **FastAPI**: Web framework and API
- **LangChain**: RAG orchestration
- **OpenAI**: Embeddings (text-embedding-ada-002) + LLM (GPT-4o)
- **PostgreSQL + pgvector**: Vector similarity search
- **PyPDF**: PDF parsing
- **Docker**: Containerization

**Data Flow**:
1. PDF → PyPDFLoader → Text chunks (1000 chars, 200 overlap)
2. Chunks → OpenAI Embeddings → 1536-dim vectors
3. Vectors → PostgreSQL pgvector table
4. Question → Embed → Vector search (k=4) → Top chunks
5. Chunks + Question → GPT-4o prompt → Answer

See **[ARCHITECTURE.md](./ARCHITECTURE.md)** for detailed diagrams.

## 📊 Performance

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| PDF Ingestion (100 pages) | 30-60s | Limited by OpenAI API rate |
| Vector Search | <50ms | PostgreSQL performance |
| Answer Generation | 2-5s | GPT-4o response time |
| Total Query Time | 3-6s | End-to-end |

## 🔒 Security

- ✅ `.env` is gitignored (never commit secrets)
- ✅ Database only accessible via Docker network
- ✅ Input validation via Pydantic models
- ✅ No user data stored beyond documents

## 🚀 Production Considerations

Before deploying to production:

1. **Use managed services**:
   - Managed PostgreSQL (AWS RDS, Google Cloud SQL)
   - Consider dedicated vector DB (Pinecone, Weaviate)

2. **Add authentication**:
   - API keys for endpoints
   - User management
   - Rate limiting

3. **Monitoring**:
   - Application logs → CloudWatch/Datadog
   - Metrics → Prometheus + Grafana
   - Error tracking → Sentry

4. **Optimize costs**:
   - Cache embeddings (Redis)
   - Batch API calls
   - Use gpt-4o-mini for lower cost

5. **Scale horizontally**:
   - Multiple web containers
   - Load balancer
   - Read replicas for DB

See [ARCHITECTURE.md](./ARCHITECTURE.md) "Scalability Considerations" section.

## 📚 Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5 + Vanilla JavaScript |
| Backend | Python 3.11 + FastAPI + Uvicorn |
| AI/ML | LangChain + OpenAI (GPT-4o) + tiktoken |
| Embeddings | OpenAI text-embedding-ada-002 (1536d) |
| Database | PostgreSQL 15+ + pgvector |
| Infrastructure | Docker + Docker Compose |

## 🛠️ Development Commands

```bash
# Start in development mode (with logs)
docker compose up --build

# Start in background
docker compose up --build -d

# Rebuild only web service
docker compose up --build web -d

# Stop all services
docker compose down

# Stop and delete volumes (fresh start)
docker compose down -v

# View logs in real-time
docker compose logs -f

# Execute command in web container
docker compose exec web python -c "import openai; print(openai.__version__)"

# Access PostgreSQL shell
docker compose exec db psql -U postgres -d demo
```

## 📄 API Reference

### POST /ingest
Upload and ingest a PDF document.

**Request**:
```http
POST /ingest
Content-Type: multipart/form-data

pdf: <file> (optional, uses PDF_PATH if omitted)
```

**Response**:
```json
{
  "status": "ingested",
  "pdf_path": "./document.pdf"
}
```

### POST /chat
Ask a question and receive an AI answer with sources.

**Request**:
```http
POST /chat
Content-Type: application/json

{
  "question": "What does this document discuss?"
}
```

**Response**:
```json
{
  "answer": "The document discusses...",
  "sources": [
    {
      "source": "./document.pdf",
      "text": "Relevant excerpt from the document..."
    }
  ]
}
```

### GET /
Serves the web UI (index.html).

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add authentication & user management
- [ ] Implement conversation history
- [ ] Add support for multiple document formats (Word, TXT, etc.)
- [ ] Hybrid search (keyword + vector)
- [ ] Response streaming for real-time answers
- [ ] Export answers to PDF/Markdown
- [ ] Multi-language support

## 📝 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- **LangChain**: RAG framework
- **OpenAI**: GPT models and embeddings
- **pgvector**: PostgreSQL vector extension
- **FastAPI**: Modern Python web framework

---

For detailed architecture, diagrams, and technical decisions, see **[ARCHITECTURE.md](./ARCHITECTURE.md)**.

