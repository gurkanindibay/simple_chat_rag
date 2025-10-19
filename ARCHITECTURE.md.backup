# Architecture Documentation

## System Overview

This is a RAG (Retrieval-Augmented Generation) application that ingests PDF documents, stores them in a vector database,
 and answers questions using AI models (OpenAI, Azure OpenAI, or local models) with context from the retrieved documents.

## High-Level Architecture (ASCII)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface (Browser)                     │
│                        index.html + app.js                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ HTTP Requests
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (main.py)                       │
│  ┌──────────────────┐           ┌──────────────────────────────┐   │
│  │  /ingest         │           │  /chat                       │   │
│  │  Upload PDF      │           │  Question Answering          │   │
│  └────────┬─────────┘           └──────────┬───────────────────┘   │
│           │                                 │                       │
│           │                                 │                       │
│           └────────────────┬────────────────┘                       │
│                            │                                        │
│                            ▼                                        │
│               ┌────────────────────────────┐                        │
│               │   Ingestion Module         │                        │
│               │   (ingestion.py)           │                        │
│               │                            │                        │
│               │  • PDF Loading             │                        │
│               │  • Text Splitting          │                        │
│               │  • Embedding Generation    │                        │
│               │  • Vector Storage          │                        │
│               │  • Retrieval               │                        │
│               │  • LLM Integration         │                        │
│               └────────┬───────────────────┘                        │
└────────────────────────┼────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┬──────────────┐
          │              │              │              │
          ▼              ▼              ▼              ▼
┌─────────────────┐ ┌──────────┐ ┌────────────────┐ ┌─────────────────┐
│ OpenAI API      │ │PostgreSQL│ │ Azure OpenAI   │ │ Local Models   │
│                 │ │+ pgvector│ │                │ │ (Optional)     │
│ • Embeddings    │ │          │ │ • Embeddings   │ │                │
│ • GPT-4o/mini   │ │ Vector   │ │ • GPT-4o/mini  │ │ • Flan-T5      │
│ • tiktoken      │ │ Search   │ │ • Entra ID     │ │ • Llama.cpp    │
└─────────────────┘ └──────────┘ └────────────────┘ └────────────────┘
```

## Component Architecture (Mermaid)

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Web UI<br/>index.html + app.js]
    end

    subgraph "API Layer"
        FastAPI[FastAPI Server<br/>main.py]
        IngestAPI[POST /ingest<br/>PDF Upload]
        ChatAPI[POST /chat<br/>Question Answering]
    end

    subgraph "Business Logic Layer"
        Ingestion[Ingestion Module<br/>ingestion.py]
        PDFLoader[PyPDFLoader<br/>PDF Processing]
        TextSplitter[RecursiveCharacterTextSplitter<br/>Chunking]
        Embedder[Embedding Generator]
        VectorStore[Vector Store Manager]
        Retriever[Document Retriever]
        LLMChain[LLM Chain<br/>RetrievalQA]
    end

    subgraph "External Services"
        OpenAI[OpenAI API<br/>Embeddings + GPT]
        AzureOpenAI[Azure OpenAI<br/>Embeddings + GPT]
        tiktoken[tiktoken<br/>Token Counter]
    end

    subgraph "Data Layer"
        Postgres[(PostgreSQL<br/>+ pgvector)]
    end

    subgraph "Optional Local Models"
        LocalEmbed[HuggingFace<br/>Embeddings]
        LocalLLM[Flan-T5 /<br/>Llama.cpp]
    end

    UI -->|Upload PDF| IngestAPI
    UI -->|Ask Question| ChatAPI
    IngestAPI --> Ingestion
    ChatAPI --> Ingestion

    Ingestion --> PDFLoader
    PDFLoader --> TextSplitter
    TextSplitter --> Embedder
    Embedder --> VectorStore
    VectorStore --> Postgres

    ChatAPI --> Retriever
    Retriever --> Postgres
    Retriever --> LLMChain
    LLMChain --> OpenAI

    Embedder -.->|if OPENAI| OpenAI
    Embedder -.->|if AZURE_OPENAI| AzureOpenAI
    Embedder -.->|if LOCAL| LocalEmbed
    LLMChain -.->|if OPENAI| OpenAI
    LLMChain -.->|if AZURE_OPENAI| AzureOpenAI
    LLMChain -.->|if LOCAL| LocalLLM

    OpenAI --> tiktoken
    AzureOpenAI --> tiktoken

    style UI fill:#e1f5ff
    style FastAPI fill:#fff4e6
    style Ingestion fill:#f3e5f5
    style OpenAI fill:#e8f5e9
    style AzureOpenAI fill:#e3f2fd
    style Postgres fill:#fce4ec
    style LocalEmbed fill:#f1f8e9
    style LocalLLM fill:#f1f8e9
```

## Data Flow Diagrams

### Ingestion Flow

```mermaid
sequenceDiagram
    participant User
    participant UI
    participant FastAPI
    participant Ingestion
    participant OpenAI
    participant Postgres

    User->>UI: Upload PDF file
    UI->>FastAPI: POST /ingest (multipart/form-data)
    FastAPI->>Ingestion: ingest_pdf(pdf_path)
    
    Ingestion->>Ingestion: Load PDF with PyPDFLoader
    Ingestion->>Ingestion: Split into chunks (1000 chars, 200 overlap)
    
    loop For each chunk
        Ingestion->>OpenAI: Generate embedding (1536 dims)
        OpenAI-->>Ingestion: Embedding vector
    end
    
    Ingestion->>Postgres: Store vectors in documents_vectors table
    Postgres-->>Ingestion: Success
    
    Ingestion-->>FastAPI: Ingestion complete
    FastAPI-->>UI: {"status":"ingested","pdf_path":"..."}
    UI-->>User: Display success message
```

### Question Answering Flow

```mermaid
sequenceDiagram
    participant User
    participant UI
    participant FastAPI
    participant Ingestion
    participant OpenAI
    participant Postgres

    User->>UI: Type question
    UI->>FastAPI: POST /chat {"question":"..."}
    FastAPI->>Ingestion: load_retriever()
    
    Ingestion->>OpenAI: Embed question
    OpenAI-->>Ingestion: Query embedding
    
    Ingestion->>Postgres: Vector similarity search (k=4)
    Postgres-->>Ingestion: Top 4 matching documents
    
    FastAPI->>FastAPI: Dedupe sources by content
    FastAPI->>Ingestion: chat_with_retriever(question, retriever)
    
    Ingestion->>Ingestion: Build context from retrieved docs
    Ingestion->>OpenAI: GPT-4o prompt with context + question
    OpenAI-->>Ingestion: Generated answer
    
    Ingestion-->>FastAPI: {"answer":"..."}
    FastAPI->>FastAPI: Attach sources metadata
    FastAPI-->>UI: {"answer":"...","sources":[...]}
    UI-->>User: Display answer + sources
```

## Technology Stack

### Frontend
```
┌─────────────────────────────────────┐
│ HTML5 + Vanilla JavaScript          │
│ • index.html (UI structure)         │
│ • app.js (API calls & rendering)    │
│ • Fetch API for HTTP requests       │
└─────────────────────────────────────┘
```

### Backend
```
┌─────────────────────────────────────┐
│ Python 3.11                         │
│ ├─ FastAPI (web framework)          │
│ ├─ Uvicorn (ASGI server)            │
│ ├─ LangChain (RAG orchestration)    │
│ ├─ PyPDF (PDF parsing)              │
│ └─ python-dotenv (config)           │
└─────────────────────────────────────┘
```

### AI/ML Stack
```
┌─────────────────────────────────────┐
│ OpenAI Integration (Primary)        │
│ ├─ text-embedding-ada-002           │
│ │  (1536 dimensions)                │
│ ├─ GPT-4o / GPT-4o-mini             │
│ └─ tiktoken (tokenization)          │
│                                     │
│ Azure OpenAI Integration            │
│ ├─ text-embedding-ada-002           │
│ │  (1536 dimensions)                │
│ ├─ GPT-4o / GPT-4o-mini / o4-mini   │
│ ├─ Microsoft Entra ID auth          │
│ └─ Azure AI Foundry                 │
│                                     │
│ Optional Local Models               │
│ ├─ HuggingFace Transformers         │
│ │  • Flan-T5 (seq2seq)              │
│ │  • sentence-transformers          │
│ └─ llama-cpp-python (Llama)         │
└─────────────────────────────────────┘
```

### Database
```
┌─────────────────────────────────────┐
│ PostgreSQL 15+                      │
│ └─ pgvector extension               │
│    • Vector similarity search       │
│    • Cosine distance operator       │
│    • HNSW indexing (optional)       │
└─────────────────────────────────────┘
```

### Infrastructure
```
┌─────────────────────────────────────┐
│ Docker + Docker Compose             │
│ ├─ db service (ankane/pgvector)     │
│ └─ web service (Python 3.11-slim)   │
│                                     │
│ Volumes                             │
│ └─ pgdata (persist database)        │
└─────────────────────────────────────┘
```

## Configuration Architecture

```mermaid
graph LR
    subgraph "Environment Variables"
        API[OPENAI_API_KEY]
        MODEL[OPENAI_CHAT_MODEL]
        DB[DATABASE_URL]
        EMBED[EMBEDDING_PROVIDER]
        LLM[LLM_PROVIDER]
        PDF[PDF_PATH]
        AZURE_EP[AZURE_OPENAI_ENDPOINT]
        AZURE_KEY[AZURE_OPENAI_API_KEY]
        AZURE_DEP[AZURE_OPENAI_CHAT_DEPLOYMENT]
    end

    subgraph "Application Behavior"
        Embeddings[Embedding Generation]
        ChatModel[Chat Model Selection]
        VectorDB[Vector Storage]
        Ingest[Default PDF]
    end

    API --> ChatModel
    MODEL --> ChatModel
    DB --> VectorDB
    EMBED --> Embeddings
    LLM --> ChatModel
    PDF --> Ingest
    AZURE_EP --> ChatModel
    AZURE_KEY --> ChatModel
    AZURE_DEP --> ChatModel

    style API fill:#ffebee
    style MODEL fill:#e3f2fd
    style DB fill:#f3e5f5
    style AZURE_EP fill:#e1f5ff
    style AZURE_KEY fill:#e1f5ff
    style AZURE_DEP fill:#e1f5ff
```

## Directory Structure (ASCII Tree)

```
ai_tryouts/
│
├── backend/                    # Python application code
│   ├── __pycache__/           # Python bytecode cache
│   ├── main.py                # FastAPI app & endpoints
│   └── ingestion.py           # RAG logic (ingest, retrieve, chat)
│
├── static/                    # Frontend assets
│   ├── index.html             # Web UI structure
│   └── app.js                 # Client-side JavaScript
│
├── db/                        # Local database files (dev)
│
├── docker/                    # Docker helper scripts
│   └── wait-for-db.sh         # Database readiness check
│
├── models/                    # Local model storage (optional)
│   └── (ggml-model.bin)       # Llama models (if using)
│
├── .env                       # Environment configuration (secrets)
├── .env.example               # Template for .env
├── .gitignore                 # Git exclusions
│
├── Dockerfile                 # Container build instructions
├── docker-compose.yml         # Multi-container orchestration
│
├── requirements.txt           # Python dependencies (local dev)
├── requirements-docker.txt    # Python dependencies (container)
│
├── ARCHITECTURE.md            # This file
└── README.md                  # Project documentation
```

## Deployment Topology

```
┌───────────────────────────────────────────────────────────────────┐
│                         Docker Host                                │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  ai_tryouts_default Network                                 │  │
│  │                                                              │  │
│  │  ┌──────────────────────┐      ┌──────────────────────┐    │  │
│  │  │  ai_tryouts-web-1    │      │  ai_tryouts-db-1     │    │  │
│  │  │                      │      │                      │    │  │
│  │  │  Python 3.11         │      │  PostgreSQL + pgvect │    │  │
│  │  │  FastAPI + Uvicorn   │─────▶│                      │    │  │
│  │  │  Port: 8000          │      │  Port: 5432          │    │  │
│  │  │                      │      │                      │    │  │
│  │  │  Volume: ./:/app     │      │  Volume: pgdata      │    │  │
│  │  └──────────┬───────────┘      └──────────────────────┘    │  │
│  │             │                                               │  │
│  └─────────────┼───────────────────────────────────────────────┘  │
│                │                                                   │
└────────────────┼───────────────────────────────────────────────────┘
                 │
                 │ Port Mapping
                 ▼
      ┌──────────────────────┐
      │  Host Machine        │
      │  localhost:8000      │
      └──────────────────────┘
```

## LLM Provider Selection Flow

```mermaid
flowchart TD
    Start([Question Received]) --> CheckProvider{LLM_PROVIDER?}
    
    CheckProvider -->|LOCAL| CheckLocal[Check Local Model]
    CheckLocal --> LocalAvail{Model Available?}
    LocalAvail -->|Yes| LoadLocal[Load Flan-T5/Transformers]
    LoadLocal --> GenLocal[Generate with Local Model]
    GenLocal --> Return([Return Answer])
    
    LocalAvail -->|No| FallbackLlama{Try Llama?}
    
    CheckProvider -->|LLAMA| CheckLlama[Check Llama Model Path]
    FallbackLlama --> CheckLlama
    CheckLlama --> LlamaAvail{llama.cpp Available?}
    LlamaAvail -->|Yes| LoadLlama[Load Llama Model]
    LoadLlama --> GenLlama[Generate with Llama]
    GenLlama --> Return
    
    LlamaAvail -->|No| CheckOpenAI
    CheckProvider -->|OPENAI| CheckOpenAI{API Key Set?}
    CheckOpenAI -->|Yes| SelectModel{OPENAI_CHAT_MODEL?}
    SelectModel -->|Set| UseCustom[Use Custom Model]
    SelectModel -->|Not Set| UseDefault[Use gpt-4o-mini]
    UseCustom --> CallOpenAI[Call OpenAI API]
    UseDefault --> CallOpenAI
    CallOpenAI --> Return
    
    CheckProvider -->|AZURE_OPENAI| CheckAzure{Azure Config Set?}
    CheckAzure -->|Yes| UseAzure[Use Azure OpenAI<br/>with Deployment Name]
    UseAzure --> CallAzure[Call Azure OpenAI API]
    CallAzure --> Return
    CheckAzure -->|No| AzureError[Raise Configuration Error]
    AzureError --> End([Fail])
    
    CheckOpenAI -->|No| Extractive[Extractive Fallback<br/>Top 3 Chunks]
    Extractive --> Return

    style CheckProvider fill:#e1f5ff
    style CheckOpenAI fill:#fff4e6
    style CallOpenAI fill:#e8f5e9
    style CheckAzure fill:#e3f2fd
    style CallAzure fill:#bbdefb
    style Extractive fill:#ffebee
```

## Embedding Provider Selection

```mermaid
flowchart TD
    Start([Need Embeddings]) --> CheckEmbed{EMBEDDING_PROVIDER?}
    
    CheckEmbed -->|OPENAI| CheckKey{OPENAI_API_KEY?}
    CheckKey -->|Set| UseOpenAI[OpenAI Embeddings<br/>text-embedding-ada-002<br/>1536 dims]
    CheckKey -->|Not Set| Error1[Raise EnvironmentError]
    
    CheckEmbed -->|AZURE_OPENAI| CheckAzureKey{Azure Config?}
    CheckAzureKey -->|Set| UseAzureOpenAI[Azure OpenAI Embeddings<br/>text-embedding-ada-002<br/>1536 dims]
    CheckAzureKey -->|Not Set| Error3[Raise EnvironmentError]
    
    CheckEmbed -->|LOCAL| CheckHF{HuggingFace Available?}
    CheckHF -->|Yes| UseLocal[HuggingFace Embeddings<br/>LOCAL_EMBEDDING_MODEL<br/>384 dims default]
    CheckHF -->|No| Error2[Raise EnvironmentError]
    
    UseOpenAI --> Return([Return Embeddings])
    UseAzureOpenAI --> Return
    UseLocal --> Return
    Error1 --> End([Fail])
    Error2 --> End
    Error3 --> End

    style UseOpenAI fill:#e8f5e9
    style UseAzureOpenAI fill:#bbdefb
    style UseLocal fill:#f1f8e9
    style Error1 fill:#ffebee
    style Error2 fill:#ffebee
    style Error3 fill:#ffebee
```

## Key Design Decisions

### 1. Embedding Dimension Consistency
- **Issue**: OpenAI/Azure OpenAI embeddings (1536d) vs local models (384d) incompatibility
- **Solution**: Environment-based provider selection, must re-ingest when switching
- **Trade-off**: Storage size vs accuracy (OpenAI/Azure OpenAI are larger but more accurate)

### 2. LLM Fallback Chain
1. **Primary**: OpenAI or Azure OpenAI (gpt-4o/gpt-4o-mini/o4-mini) - best quality
2. **Secondary**: Local Transformers (Flan-T5) - offline capability
3. **Tertiary**: Llama.cpp - GGML models for resource efficiency
4. **Fallback**: Extractive (no LLM) - always works

### 3. Vector Store Strategy
- **Choice**: PostgreSQL + pgvector
- **Rationale**: 
  - Single database (no separate vector DB needed)
  - SQL familiarity for queries
  - ACID transactions
  - Built-in persistence
- **Alternative considered**: FAISS (faster but in-memory only)

### 4. Chunking Strategy
- **Size**: 1000 characters
- **Overlap**: 200 characters
- **Rationale**: Balance between context preservation and retrieval precision

### 5. Retrieval Count (k=4)
- **Default**: Top 4 most similar chunks
- **Rationale**: Fits in most LLM context windows while providing sufficient context
- **Configurable**: Can be adjusted in `load_retriever()`

## Azure OpenAI Integration

### Architecture Overview

Azure OpenAI provides enterprise-grade AI capabilities with enhanced security, compliance, and data residency requirements.

```mermaid
graph TB
    subgraph "Application Layer"
        App[RAG Application]
        Config[Configuration]
    end
    
    subgraph "Azure OpenAI Service"
        Endpoint[Azure OpenAI Endpoint<br/>*.cognitiveservices.azure.com]
        ChatDeploy[Chat Deployment<br/>e.g., o4-mini]
        EmbedDeploy[Embedding Deployment<br/>text-embedding-ada-002]
    end
    
    subgraph "Authentication"
        APIKey[API Key Auth]
        EntraID[Microsoft Entra ID<br/>RBAC]
    end
    
    subgraph "Azure AI Foundry"
        Resource[Azure OpenAI Resource]
        Monitor[Monitoring & Logs]
        Quotas[Rate Limits & Quotas]
    end
    
    App --> Config
    Config --> Endpoint
    Config --> APIKey
    Config -.->|Optional| EntraID
    
    Endpoint --> ChatDeploy
    Endpoint --> EmbedDeploy
    
    ChatDeploy --> Resource
    EmbedDeploy --> Resource
    Resource --> Monitor
    Resource --> Quotas
    
    style App fill:#fff4e6
    style Endpoint fill:#bbdefb
    style ChatDeploy fill:#e3f2fd
    style EmbedDeploy fill:#e3f2fd
    style APIKey fill:#ffebee
    style EntraID fill:#e8f5e9
    style Resource fill:#f3e5f5
```

### Configuration Requirements

**Required Environment Variables:**
- `LLM_PROVIDER=AZURE_OPENAI`
- `EMBEDDING_PROVIDER=AZURE_OPENAI`
- `AZURE_OPENAI_ENDPOINT` - Base URL (e.g., `https://your-resource.cognitiveservices.azure.com/`)
- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key
- `AZURE_OPENAI_API_VERSION` - API version (e.g., `2025-01-01-preview`)
- `AZURE_OPENAI_CHAT_DEPLOYMENT` - Deployment name for chat model
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` - Deployment name for embeddings

### Key Benefits

1. **Enterprise Security**
   - Private endpoints and VNet integration
   - Microsoft Entra ID (Azure AD) authentication
   - Role-Based Access Control (RBAC)
   - Data stays in your Azure region

2. **Compliance**
   - SOC 2, ISO 27001, HIPAA, GDPR compliant
   - Data residency guarantees
   - No data used for model training

3. **Cost Management**
   - Predictable pricing with reserved capacity
   - Quota management and rate limiting
   - Usage monitoring and budgets

4. **Reliability**
   - 99.9% SLA
   - Regional failover support
   - Azure Monitor integration

### Authentication Flow

```mermaid
sequenceDiagram
    participant App as RAG Application
    participant Config as Configuration
    participant Azure as Azure OpenAI
    participant Entra as Microsoft Entra ID

    Note over App,Config: Option 1: API Key (Simple)
    App->>Config: Load AZURE_OPENAI_API_KEY
    App->>Azure: Request with API key in header
    Azure-->>App: Response

    Note over App,Entra: Option 2: Entra ID (Production)
    App->>Entra: Request access token
    Entra-->>App: JWT token
    App->>Azure: Request with Bearer token
    Azure->>Entra: Validate token
    Entra-->>Azure: Token valid
    Azure-->>App: Response
```

### Deployment Models

The application supports multiple Azure OpenAI deployment configurations:

| Deployment | Use Case | Model Examples |
|------------|----------|----------------|
| Chat | Question answering, conversation | gpt-4o, gpt-4o-mini, o4-mini |
| Embeddings | Document vectorization | text-embedding-ada-002 |

### Migration from OpenAI to Azure OpenAI

**Steps:**
1. Set `LLM_PROVIDER=AZURE_OPENAI` in `.env`
2. Set `EMBEDDING_PROVIDER=AZURE_OPENAI` in `.env`
3. Add Azure-specific configuration variables
4. **Important**: Re-ingest documents if switching embedding providers (dimension compatibility)
5. Restart the backend server

**No Code Changes Required** - The application automatically detects and uses Azure OpenAI when configured.

### Monitoring Integration

Azure OpenAI integrates with Azure Monitor for:
- **Request Logs**: Track all API calls, latency, and errors
- **Token Usage**: Monitor token consumption by deployment
- **Error Rates**: Alert on failed requests
- **Cost Analysis**: Track spending by resource and deployment

## Security Considerations

```
┌─────────────────────────────────────────────────────────────┐
│ Security Layer                                              │
│                                                             │
│  1. Environment Variables (.env)                            │
│     • OPENAI_API_KEY (never commit)                         │
│     • AZURE_OPENAI_API_KEY (never commit)                   │
│     • DATABASE_URL (credentials)                            │
│     • .gitignore protects secrets                           │
│                                                             │
│  2. Authentication Options                                  │
│     • OpenAI: API key authentication                        │
│     • Azure OpenAI: API key OR Microsoft Entra ID           │
│     • Application: Microsoft Entra ID with JWT tokens       │
│                                                             │
│  3. Network Isolation (Docker)                              │
│     • db service only accessible via internal network       │
│     • web service exposes only port 8000                    │
│     • Azure: Private Link support available                 │
│                                                             │
│  4. Input Validation                                        │
│     • FastAPI Pydantic models (ChatRequest)                 │
│     • File type checking (PDF only)                         │
│     • Configuration validation (provider selection)         │
│                                                             │
│  5. Dependency Management                                   │
│     • Pinned versions in requirements.txt                   │
│     • Docker BuildKit cache for reproducibility             │
│                                                             │
│  6. Azure-Specific Security                                 │
│     • RBAC roles (Cognitive Services OpenAI User)           │
│     • Data residency in chosen region                       │
│     • No training data sharing                              │
│     • Compliance certifications (SOC2, ISO, HIPAA)          │
└─────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

| Operation | Typical Latency | Bottleneck |
|-----------|----------------|------------|
| PDF Ingestion (100 pages) | 30-60s | OpenAI/Azure API rate limits |
| Embedding Generation (OpenAI) | ~1s per 10 chunks | OpenAI API |
| Embedding Generation (Azure) | ~1s per 10 chunks | Azure OpenAI API |
| Vector Search (k=4) | <50ms | PostgreSQL query |
| LLM Response (GPT-4o OpenAI) | 2-5s | OpenAI API |
| LLM Response (GPT-4o Azure) | 2-5s | Azure OpenAI API |
| End-to-End Query | 3-6s | LLM generation |

**Note**: Azure OpenAI performance is comparable to OpenAI, with the added benefit of regional proximity and dedicated capacity options.

## Scalability Considerations

```mermaid
graph TB
    subgraph "Current (Single Node)"
        Web[Web Container]
        DB[(PostgreSQL)]
        Web --> DB
    end

    subgraph "Future Scaling Options"
        LB[Load Balancer]
        Web1[Web Instance 1]
        Web2[Web Instance 2]
        Web3[Web Instance 3]
        
        PG_Primary[(PostgreSQL<br/>Primary)]
        PG_Replica1[(Read Replica 1)]
        PG_Replica2[(Read Replica 2)]
        
        Cache[Redis Cache<br/>for embeddings]
        
        LB --> Web1
        LB --> Web2
        LB --> Web3
        
        Web1 --> Cache
        Web2 --> Cache
        Web3 --> Cache
        
        Web1 --> PG_Primary
        Web2 --> PG_Primary
        Web3 --> PG_Primary
        
        PG_Primary -.->|Replication| PG_Replica1
        PG_Primary -.->|Replication| PG_Replica2
        
        Web1 --> PG_Replica1
        Web2 --> PG_Replica2
    end

    style Web fill:#fff4e6
    style DB fill:#fce4ec
    style LB fill:#e1f5ff
    style Cache fill:#f3e5f5
```

## Monitoring & Observability

### Logs
- **Container Logs**: `docker compose logs web -f`
- **Database Logs**: `docker compose logs db -f`
- **Format**: Uvicorn access logs + Python print statements

### Metrics to Monitor
1. **Ingestion**:
   - Documents ingested per hour
   - Average chunk count per document
   - Embedding generation time
   - Vector storage latency

2. **Query**:
   - Questions per minute
   - Average response time
   - Retrieval precision (user feedback)
   - OpenAI/Azure OpenAI API quota usage
   - Provider-specific error rates

3. **Infrastructure**:
   - Container CPU/memory usage
   - PostgreSQL query performance
   - Vector table size
   - Disk usage (pgdata volume)

## Error Handling Flow

```mermaid
flowchart TD
    Request[User Request] --> Validate{Input Valid?}
    Validate -->|No| Return400[400 Bad Request]
    Validate -->|Yes| Process[Process Request]
    
    Process --> CheckEnv{Dependencies OK?}
    CheckEnv -->|Missing tiktoken| Return500A[500: Install tiktoken]
    CheckEnv -->|No API Key| Return400A[400: Set OPENAI_API_KEY]
    CheckEnv -->|No DB| Return500B[500: DATABASE_URL not set]
    CheckEnv -->|OK| Execute[Execute Logic]
    
    Execute --> Runtime{Runtime Error?}
    Runtime -->|Vector Dim Mismatch| Return500C[500: Re-ingest with same embedder]
    Runtime -->|OpenAI Error| Return500D[500: OpenAI API error]
    Runtime -->|Success| Return200[200 OK + Response]
    
    Return400 --> End([Client Receives Error])
    Return400A --> End
    Return500A --> End
    Return500B --> End
    Return500C --> End
    Return500D --> End
    Return200 --> End

    style Return400 fill:#ffebee
    style Return400A fill:#ffebee
    style Return500A fill:#ffebee
    style Return500B fill:#ffebee
    style Return500C fill:#ffebee
    style Return500D fill:#ffebee
    style Return200 fill:#e8f5e9
```

## Development Workflow

```
Developer Workflow
──────────────────

1. Code Changes
   ├─ Edit backend/*.py
   ├─ Edit static/*.{html,js}
   └─ Edit requirements*.txt

2. Local Testing
   ├─ docker compose up --build
   └─ curl localhost:8000/chat

3. Logs & Debug
   └─ docker compose logs -f

4. Iterate
   ├─ Fix code
   └─ Rebuild container

5. Commit (git flow)
   ├─ .env stays local (gitignored)
   └─ Push changes
```

## API Endpoints Reference

### POST /ingest
**Purpose**: Upload and process a PDF document

**Request**:
```http
POST /ingest HTTP/1.1
Content-Type: multipart/form-data

pdf: <file>
```

**Response**:
```json
{
  "status": "ingested",
  "pdf_path": "./document.pdf"
}
```

### POST /chat
**Purpose**: Ask a question and get an AI-generated answer

**Request**:
```http
POST /chat HTTP/1.1
Content-Type: application/json

{
  "question": "What does Citus provide?"
}
```

**Response**:
```json
{
  "answer": "Citus provides...",
  "sources": [
    {
      "source": "./document.pdf",
      "text": "First 400 chars of relevant chunk..."
    }
  ]
}
```

### GET /
**Purpose**: Serve the web UI

**Response**: HTML page (index.html)

---

## Future Enhancements

1. **Authentication & Multi-tenancy**
   - User accounts
   - Per-user document isolation
   - API key management

2. **Advanced Retrieval**
   - Hybrid search (keyword + vector)
   - Reranking with cross-encoders
   - Query expansion

3. **Observability**
   - Prometheus metrics
   - Grafana dashboards
   - Distributed tracing

4. **Performance**
   - Embedding caching
   - Response streaming
   - Async batch processing

5. **Features**
   - Multi-document support
   - Citation tracking
   - Conversation history
   - Export to markdown/PDF
