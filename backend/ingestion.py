import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, Text
from sqlalchemy.dialects.postgresql import ARRAY
import sqlalchemy
try:
    # Newer langchain exposes PGVector / SQLAlchemy integration
    from langchain.vectorstores import PGVector
except Exception:
    PGVector = None
from pgvector.sqlalchemy import Vector
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Local LLM support (optional)
# Use a stronger Flan-T5 variant by default for better quality when running locally.
LOCAL_LLM_MODEL = os.environ.get('LOCAL_LLM_MODEL', 'google/flan-t5-large')
# LLM_PROVIDER will be loaded from database, env var is just the initial default
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'OPENAI')
LLAMA_MODEL_PATH = os.environ.get('LLAMA_MODEL_PATH')

# Local generation settings (tunable via env)
LOCAL_LLM_TEMPERATURE = float(os.environ.get('LOCAL_LLM_TEMPERATURE', '0.2'))
LOCAL_LLM_MAX_NEW_TOKENS = int(os.environ.get('LOCAL_LLM_MAX_NEW_TOKENS', '256'))
LLAMA_TEMPERATURE = float(os.environ.get('LLAMA_TEMPERATURE', '0.2'))
LLAMA_MAX_TOKENS = int(os.environ.get('LLAMA_MAX_TOKENS', '512'))
OPENAI_CHAT_MODEL = os.environ.get('OPENAI_CHAT_MODEL', 'gpt-4o-mini')

# cache for loaded local model
_local_llm = None

def _load_local_llm(model_name: str):
    """Lazy-load a seq2seq local model (Flan-T5 style). Returns (tokenizer, model).
    Raises EnvironmentError with a helpful message if transformers/torch are missing.
    """
    global _local_llm
    if _local_llm is not None:
        return _local_llm
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception:
        raise EnvironmentError("Local LLM requested but 'transformers' (and 'torch') are not installed. Add them to your environment to use a local LLM.")

    # Check if model_name is a local path or a HuggingFace model ID
    import os
    if os.path.isdir(model_name):
        # Local path exists - load from local directory
        print(f"Loading local model from: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    elif model_name.startswith('/') or model_name.startswith('./'):
        # Looks like a path but doesn't exist - show helpful error
        raise EnvironmentError(
            f"Local model path '{model_name}' does not exist. "
            f"Either create this directory with a valid model, or use a HuggingFace model ID like 'google/flan-t5-base' or 'google/flan-t5-small'."
        )
    else:
        # HuggingFace model ID - download if needed
        print(f"Loading HuggingFace model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    _local_llm = (tokenizer, model)
    return _local_llm


def _dedupe_documents(docs):
    """Return documents with duplicate page_content removed, preserving order."""
    seen = set()
    out = []
    for d in docs:
        text = (d.page_content or '').strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(d)
    return out

# Optional local embedding support
from langchain.embeddings import OpenAIEmbeddings
try:
    from langchain_embeddings.huggingface import HuggingFaceEmbeddings
except Exception:
    # fallback to langchain's HuggingFaceEmbeddings if available
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None

EMBEDDING_PROVIDER = os.environ.get('EMBEDDING_PROVIDER', 'OPENAI')
LOCAL_EMBEDDING_MODEL = os.environ.get('LOCAL_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

INDEX_DIR = os.path.join(os.getcwd(), "db")
DATABASE_URL = os.environ.get('DATABASE_URL')
PGVECTOR_TABLE = os.environ.get('PGVECTOR_TABLE', 'documents_vectors')

# Configuration management
_config_cache = None


def init_config_table():
    """Create the configuration table if it doesn't exist."""
    if DATABASE_URL is None:
        raise EnvironmentError("DATABASE_URL not set. Can't initialize config table.")
    
    from sqlalchemy import create_engine, text
    
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Create config table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS app_config (
                key VARCHAR(255) PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
        
        # Insert default values if they don't exist
        conn.execute(text("""
            INSERT INTO app_config (key, value) 
            VALUES ('LLM_PROVIDER', :llm_provider)
            ON CONFLICT (key) DO NOTHING
        """), {"llm_provider": os.environ.get('LLM_PROVIDER', 'OPENAI')})
        
        conn.execute(text("""
            INSERT INTO app_config (key, value) 
            VALUES ('EMBEDDING_PROVIDER', :embedding_provider)
            ON CONFLICT (key) DO NOTHING
        """), {"embedding_provider": os.environ.get('EMBEDDING_PROVIDER', 'OPENAI')})
        
        conn.commit()


def get_config_from_db():
    """Get configuration from database. Returns dict with LLM_PROVIDER and EMBEDDING_PROVIDER.
    Falls back to environment variables if database is not available.
    """
    global _config_cache
    
    if DATABASE_URL is None:
        return {
            "LLM_PROVIDER": os.environ.get('LLM_PROVIDER', 'OPENAI'),
            "EMBEDDING_PROVIDER": os.environ.get('EMBEDDING_PROVIDER', 'OPENAI')
        }
    
    try:
        from sqlalchemy import create_engine, text
        
        # Initialize table if needed
        init_config_table()
        
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT key, value FROM app_config 
                WHERE key IN ('LLM_PROVIDER', 'EMBEDDING_PROVIDER')
            """))
            
            config = {}
            for row in result:
                config[row[0]] = row[1]
            
            # Cache the config
            _config_cache = config
            
            return config
    except Exception as e:
        print(f"Warning: Failed to load config from database: {e}")
        # Fall back to environment variables
        return {
            "LLM_PROVIDER": os.environ.get('LLM_PROVIDER', 'OPENAI'),
            "EMBEDDING_PROVIDER": os.environ.get('EMBEDDING_PROVIDER', 'OPENAI')
        }


def update_config_in_db(key: str, value: str):
    """Update a configuration value in the database.
    
    Args:
        key: Configuration key (LLM_PROVIDER or EMBEDDING_PROVIDER)
        value: Configuration value (OPENAI or LOCAL)
    
    Returns:
        True if successful, False otherwise
    """
    global _config_cache, LLM_PROVIDER, EMBEDDING_PROVIDER, _local_llm
    
    if DATABASE_URL is None:
        raise EnvironmentError("DATABASE_URL not set. Can't update config.")
    
    if key not in ['LLM_PROVIDER', 'EMBEDDING_PROVIDER']:
        raise ValueError(f"Invalid config key: {key}")
    
    if value not in ['OPENAI', 'LOCAL']:
        raise ValueError(f"Invalid config value: {value}")
    
    try:
        from sqlalchemy import create_engine, text
        
        # Initialize table if needed
        init_config_table()
        
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO app_config (key, value, updated_at) 
                VALUES (:key, :value, CURRENT_TIMESTAMP)
                ON CONFLICT (key) DO UPDATE 
                SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP
            """), {"key": key, "value": value})
            
            conn.commit()
        
        # Update global variables
        if key == 'LLM_PROVIDER':
            LLM_PROVIDER = value
            # Reset cached local LLM if switching providers
            if value == 'OPENAI':
                _local_llm = None
        elif key == 'EMBEDDING_PROVIDER':
            EMBEDDING_PROVIDER = value
        
        # Invalidate cache
        _config_cache = None
        
        return True
    except Exception as e:
        print(f"Error updating config: {e}")
        raise


def load_config_from_db():
    """Load configuration from database and update global variables."""
    global LLM_PROVIDER, EMBEDDING_PROVIDER
    
    config = get_config_from_db()
    LLM_PROVIDER = config.get('LLM_PROVIDER', 'OPENAI')
    EMBEDDING_PROVIDER = config.get('EMBEDDING_PROVIDER', 'OPENAI')


def ensure_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    # Load configuration from database on startup
    try:
        load_config_from_db()
    except Exception as e:
        print(f"Warning: Could not load config from database: {e}")
        print("Using environment variables for configuration")


def log_ingestion(pdf_path: str):
    """Log an ingested PDF to a simple JSON file in the db directory."""
    import json
    from datetime import datetime
    
    log_file = os.path.join(INDEX_DIR, "ingestion_log.json")
    
    # Load existing log or start fresh
    log_data = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        except Exception:
            log_data = []
    
    # Append new entry
    entry = {
        "pdf_path": pdf_path,
        "filename": os.path.basename(pdf_path),
        "timestamp": datetime.utcnow().isoformat(),
    }
    log_data.append(entry)
    
    # Write back
    try:
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to log ingestion: {e}")


def get_ingestion_log():
    """Retrieve the list of ingested PDFs from the log file."""
    import json
    
    log_file = os.path.join(INDEX_DIR, "ingestion_log.json")
    if not os.path.exists(log_file):
        return []
    
    try:
        with open(log_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read ingestion log: {e}")
        return []


def clear_ingestion_log():
    """Clear the ingestion log file."""
    import json
    
    log_file = os.path.join(INDEX_DIR, "ingestion_log.json")
    
    try:
        # Write empty list to log file
        with open(log_file, 'w') as f:
            json.dump([], f, indent=2)
        return True
    except Exception as e:
        print(f"Warning: Failed to clear ingestion log: {e}")
        return False


def get_embedding_table_stats():
    """Get row counts for all embedding tables.
    
    Returns a dict mapping table names to their row counts.
    Raises EnvironmentError if DATABASE_URL is not set.
    """
    if DATABASE_URL is None:
        raise EnvironmentError("DATABASE_URL not set. Can't check embedding tables.")
    
    from sqlalchemy import create_engine, text, inspect
    
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    
    with engine.connect() as conn:
        tables = inspector.get_table_names()
        
        # Find embedding-related tables
        candidate_tables = [t for t in tables if t.startswith('langchain_pg') or 'embed' in t or 'vector' in t]
        
        result = {}
        for tbl in candidate_tables:
            try:
                r = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}"))
                row = r.fetchone()
                count = int(row[0]) if row is not None else 0
                result[tbl] = count
            except Exception as e:
                result[tbl] = {"error": str(e)}
        
        return result


def get_embeddings():
    """Return an embeddings instance based on configuration. Raises EnvironmentError with
    a helpful message if the chosen provider is unavailable.
    """
    if EMBEDDING_PROVIDER.upper() == 'LOCAL':
        if HuggingFaceEmbeddings is None:
            raise EnvironmentError("EMBEDDING_PROVIDER=LOCAL but no HuggingFace embeddings are available. Install sentence-transformers or langchain's huggingface integration.")
        print(f"Using local embeddings model: {LOCAL_EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)

    # default: OPENAI
    # Validate OPENAI_API_KEY presence early to produce a clear error instead of a Pydantic validation error
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        raise EnvironmentError("EMBEDDING_PROVIDER=OPENAI but OPENAI_API_KEY is not set. Set the env var or switch to EMBEDDING_PROVIDER=LOCAL.")
    print("Using OpenAI embeddings")
    return OpenAIEmbeddings()


def ingest_pdf(pdf_path: str):
    """Load PDF, split, embed and persist vectors to Postgres/pgvector."""
    print(f"Ingesting {pdf_path} ...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    # choose embeddings with a helper that validates availability and falls back when possible
    embeddings = get_embeddings()

    # Store vectors in Postgres + pgvector
    if DATABASE_URL is None:
        raise EnvironmentError("DATABASE_URL not set. Can't persist vectors to pgvector.")

    if PGVector is not None:
        # LangChain's PGVector wrapper
        print("Using LangChain PGVector to persist vectors")
        vectorstore = PGVector.from_documents(docs, embeddings, collection_name=PGVECTOR_TABLE, connection_string=DATABASE_URL)
        print(f"Persisted vectors to table {PGVECTOR_TABLE} in Postgres")
    else:
        # Fallback: simple SQLAlchemy table insert with pgvector Vector type
        print("Using fallback SQLAlchemy + pgvector to persist vectors")
        engine = create_engine(DATABASE_URL)
        meta = MetaData()

        # compute a sample embedding to determine dimension
        sample = docs[0].page_content if len(docs) > 0 else ""
        sample_emb = embeddings.embed_documents([sample])[0]
        dim = len(sample_emb)

        tbl = Table(
            PGVECTOR_TABLE,
            meta,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('text', Text),
            Column('metadata', Text),
            Column('vector', Vector(dim))
        )
        meta.create_all(engine)
        # compute embeddings and insert in batches
        with engine.connect() as conn:
            for d in docs:
                emb = embeddings.embed_documents([d.page_content])[0]
                ins = tbl.insert().values(text=d.page_content, metadata=str(d.metadata or {}), vector=emb)
                conn.execute(ins)
        print(f"Inserted {len(docs)} vectors into {PGVECTOR_TABLE}")


def load_retriever(k: int = 15):
    """Create and return a retriever backed by Postgres/pgvector."""
    if DATABASE_URL is None:
        raise EnvironmentError("DATABASE_URL not set. Can't load retriever from pgvector.")

    # get embeddings instance (will raise a clear error if none available)
    embeddings = get_embeddings()
    # Adapter object that provides the methods expected by PGVector's
    # `embedding_function` parameter (it expects an object with embed_query/embed_documents).
    class EmbeddingAdapter:
        def __init__(self, inner):
            self._inner = inner

        def embed_documents(self, texts):
            if hasattr(self._inner, 'embed_documents'):
                return self._inner.embed_documents(texts)
            # fall back to calling embed_query for each item
            if hasattr(self._inner, 'embed_query'):
                return [self._inner.embed_query(t) for t in texts]
            raise RuntimeError('Embedding backend has no embed_documents/embed_query method')

        def embed_query(self, text):
            if hasattr(self._inner, 'embed_query'):
                return self._inner.embed_query(text)
            if hasattr(self._inner, 'embed_documents'):
                return self._inner.embed_documents([text])[0]
            raise RuntimeError('Embedding backend has no embed_query/embed_documents method')

    adapter = EmbeddingAdapter(embeddings)

    if PGVector is not None:
        # Pass the adapter object required by PGVector implementations.
        vs = PGVector(collection_name=PGVECTOR_TABLE, connection_string=DATABASE_URL, embedding_function=adapter)
        return vs.as_retriever(search_kwargs={"k": k})
    else:
        # fallback: simple SQL-based nearest neighbor using pgvector operator
        # We implement a tiny retriever that queries pgvector for nearest vectors.
        engine = create_engine(DATABASE_URL)
        meta = MetaData()
        tbl = Table(PGVECTOR_TABLE, meta, autoload_with=engine)

        class SimpleRetriever:
            def __init__(self, engine, tbl, embeddings, k=4):
                self.engine = engine
                self.tbl = tbl
                self.embeddings = embeddings
                self.k = k

            def _embed_query(self, query):
                # use embed_query if available, else embed_documents
                if hasattr(self.embeddings, 'embed_query'):
                    return self.embeddings.embed_query(query)
                if hasattr(self.embeddings, 'embed_documents'):
                    return self.embeddings.embed_documents([query])[0]
                raise RuntimeError('Embedding backend has no embed_query/embed_documents')

            def get_relevant_documents(self, query):
                q_emb = self._embed_query(query)
                # pgvector SQL: ORDER BY vector <-> :vec LIMIT :k (distance operator)
                sql = sqlalchemy.text(f"SELECT text, metadata FROM {PGVECTOR_TABLE} ORDER BY vector <-> :vec LIMIT :k")
                with self.engine.connect() as conn:
                    rows = conn.execute(sql, {"vec": q_emb, "k": self.k}).fetchall()
                docs = []
                from langchain.schema import Document
                for r in rows:
                    docs.append(Document(page_content=r[0], metadata={"source": r[1]}))
                return docs

        return SimpleRetriever(engine, tbl, embeddings, k=k)


def chat_with_retriever(question: str, retriever, llm_temperature: float = 0.0):
    """
    Answer a question using the retriever. If an OpenAI API key is configured we call
    ChatOpenAI to generate a fluent answer. If no OpenAI key is present but local
    embeddings/retriever exist, we fall back to a simple extractive answer composed
    from the top retrieved documents so the endpoint works without an LLM.
    """
    openai_key = os.environ.get('OPENAI_API_KEY')

    # If LLM_PROVIDER is configured for local models, attempt to use the local LLM
    if LLM_PROVIDER and LLM_PROVIDER.upper() == 'LOCAL':
        try:
            tokenizer, model = _load_local_llm(LOCAL_LLM_MODEL)
        except EnvironmentError as e:
            # couldn't load local model; fall through to other options
            print(f"Local LLM requested but unavailable: {e}")
        else:
            # gather context from top retrieved docs
            try:
                docs = retriever.get_relevant_documents(question)
            except Exception:
                docs = retriever.get_documents(question)

            # dedupe retrieved docs to avoid repeated identical chunks
            docs = _dedupe_documents(docs)
            context = "\n\n".join([d.page_content for d in docs[:6]])
            # Stronger prompt: ask for a numbered, step-by-step answer, require citations
            # with page numbers when available, and return 'I don't know.' exactly if
            # the context does not contain the answer. This reduces hallucinations.
            prompt_text = (
                "You are an assistant that answers using only the provided context.\n\n"
                "Context:\n" + context + "\n\n"
                "Question: " + question + "\n\n"
                "Instructions:\n"
                "1) If the answer can be found in the context, provide a clear, numbered, step-by-step list (use numbers 1., 2., 3., ...).\n"
                "2) For each step, cite the source(s) in parentheses with page numbers or section titles when available (for example: (source: page 12)).\n"
                "3) If the context does NOT contain the answer, reply exactly: 'I don't know.' Do NOT make up facts.\n\n"
                "Answer:")

            try:
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # move model to device and set eval mode
                model.to(device)
                model.eval()

                # T5-like models typically work well with input max lengths ~512
                inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Use sampling for more fluent outputs with configurable temperature/top_p
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=LOCAL_LLM_TEMPERATURE,
                    top_p=0.95,
                    no_repeat_ngram_size=3,
                    num_return_sequences=1,
                )
                answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return {"answer": answer}
            except Exception as e:
                print(f"Local LLM generation failed: {e}")

    # Llama (llama.cpp via llama-cpp-python) support
    if LLM_PROVIDER and LLM_PROVIDER.upper() == 'LLAMA':
        if not LLAMA_MODEL_PATH:
            print("LLAMA_MODEL_PATH not set; cannot use Llama provider.")
        else:
            try:
                from llama_cpp import Llama
            except Exception as e:
                print(f"llama-cpp-python not installed: {e}")
            else:
                try:
                    llm = Llama(model_path=LLAMA_MODEL_PATH)
                    try:
                        docs = retriever.get_relevant_documents(question)
                    except Exception:
                        docs = retriever.get_documents(question)
                    docs = _dedupe_documents(docs)
                    context = "\n\n".join([d.page_content for d in docs[:8]])
                    prompt_text = (
                        "You are an assistant that answers using only the provided context.\n\n"
                        "Context:\n" + context + "\n\n"
                        "Question: " + question + "\n\n"
                        "Instructions:\n"
                        "1) If the answer can be found in the context, provide a clear, numbered, step-by-step list (use numbers 1., 2., 3., ...).\n"
                        "2) For each step, cite the source(s) in parentheses with page numbers or section titles when available (for example: (source: page 12)).\n"
                        "3) If the context does NOT contain the answer, reply exactly: 'I don't know.' Do NOT make up facts.\n\n"
                        "Answer:")
                    resp = llm.create(prompt=prompt_text, max_tokens=LLAMA_MAX_TOKENS, temperature=LLAMA_TEMPERATURE)
                    # llama-cpp-python response shape may vary; handle common forms
                    if isinstance(resp, dict) and 'choices' in resp and resp['choices']:
                        answer = resp['choices'][0].get('text') or resp['choices'][0].get('message', {}).get('content', '')
                    else:
                        answer = resp.get('text', '') if isinstance(resp, dict) else str(resp)
                    return {"answer": answer}
                except Exception as e:
                    print(f"Llama generation failed: {e}")

    # If OpenAI is available, use ChatOpenAI for a generative answer.
    if openai_key:
        # Prefer a stronger OpenAI chat model when available; fall back gracefully for older SDKs.
        model_name = OPENAI_CHAT_MODEL or 'gpt-4o-mini'
        try:
            llm = ChatOpenAI(temperature=llm_temperature, model=model_name)
        except TypeError:
            llm = ChatOpenAI(temperature=llm_temperature, model_name=model_name)
        template = (
            "You are a helpful assistant that must answer using only the retrieved documents.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Instructions:\n"
            "1) If the answer exists in the context, provide a clear, numbered, step-by-step list (1., 2., 3., ...).\n"
            "2) For each step, cite the source(s) in parentheses with page numbers or section titles when available (for example: (source: page 12)).\n"
            "3) If the context does NOT contain the answer, reply exactly: 'I don't know.' Do NOT make up facts.\n\n"
            "Answer:"
        )
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        try:
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})
            result = qa.run(question)
            return {"answer": result}
        except Exception as e:
            print(f"[ERROR] OpenAI LLM failed: {e}")
            import traceback
            traceback.print_exc()
            # Fall through to fallback
            pass

    # No LLM configured — perform a safe extractive fallback using retrieved docs.
    try:
        docs = retriever.get_relevant_documents(question)
    except Exception:
        # fallback method name
        docs = retriever.get_documents(question)

    if not docs:
        return {"answer": "I don't know. No relevant documents were found."}

    # Compose a short extractive answer from the top documents (first 1-3)
    pieces = []
    for d in docs[:3]:
        text = d.page_content.strip()
        snippets = text.split('\n')
        pieces.append(snippets[0][:400])

    answer = "\n\n---\n\n".join(pieces)
    return {"answer": f"Extractive answer (no LLM configured):\n\n{answer}"}


def delete_embeddings():
    """Delete (truncate) all embedding tables.
    
    - Try the configured PGVECTOR_TABLE first.
    - If not found, auto-detect candidate embedding tables (langchain defaults or names containing 'embed'/'vector').
    - Truncate all candidates with CASCADE to handle foreign key constraints.
    
    Returns a dict mapping truncated table names to number of rows deleted.
    Raises EnvironmentError on configuration or SQL errors.
    
    Security note: Table names are determined server-side only; no client parameters accepted.
    """
    if DATABASE_URL is None:
        raise EnvironmentError("DATABASE_URL not set. Can't delete embeddings from pgvector.")

    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.exc import ProgrammingError, OperationalError

    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    
    # Use engine.begin() for auto-commit transaction
    with engine.begin() as conn:
        try:
            tables = inspector.get_table_names()
        except Exception as e:
            raise EnvironmentError(f"Failed to list tables: {e}")

        # Try configured PGVECTOR_TABLE first
        to_delete = []
        if PGVECTOR_TABLE in tables:
            to_delete = [PGVECTOR_TABLE]
        else:
            # Auto-detect candidate embedding tables
            candidates = [t for t in tables if t.startswith('langchain_pg') or 'embed' in t or 'vector' in t]
            if not candidates:
                raise EnvironmentError(f"No embeddings table found (looked for '{PGVECTOR_TABLE}'). Available tables: {tables}")
            to_delete = candidates

        # Truncate all identified tables with CASCADE to handle foreign key constraints
        result = {}
        try:
            tbls = ','.join(to_delete)
            conn.execute(text(f"TRUNCATE TABLE {tbls} RESTART IDENTITY CASCADE"))
            # After truncation, counts are zero
            for t in to_delete:
                result[t] = 0
            
            # Also clear the ingestion log when embeddings are deleted
            clear_ingestion_log()
            
            return result
        except Exception as e:
            raise EnvironmentError(f"Failed to truncate tables {to_delete}: {e}")
