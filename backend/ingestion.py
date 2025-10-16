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
LOCAL_LLM_MODEL = os.environ.get('LOCAL_LLM_MODEL', 'google/flan-t5-small')
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'OPENAI')
LLAMA_MODEL_PATH = os.environ.get('LLAMA_MODEL_PATH')
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


def ensure_index():
    os.makedirs(INDEX_DIR, exist_ok=True)


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


def load_retriever(k: int = 4):
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
            prompt_text = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

            try:
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # move model to device and set eval mode
                model.to(device)
                model.eval()

                # T5-like models typically work well with input max lengths ~512
                inputs = tokenizer(prompt_text, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Use sampling for more fluent outputs with low temperature and nucleus sampling
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
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
                    prompt_text = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                    resp = llm.create(prompt=prompt_text, max_tokens=256, temperature=0.2)
                    answer = resp['choices'][0]['text'] if 'choices' in resp and resp['choices'] else resp.get('text', '')
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
        template = "You are a helpful assistant. Use the retrieved documents to answer the question. Include source page numbers if available.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})
        result = qa.run(question)
        return {"answer": result}

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
