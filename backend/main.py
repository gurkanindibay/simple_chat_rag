import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.ingestion import ensure_index, ingest_pdf, load_retriever, chat_with_retriever

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")


class ChatRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("./static/index.html")


@app.post("/ingest")
async def ingest(pdf: UploadFile = File(None)):
    # If uploaded file is provided, save and ingest; otherwise ingest from PDF_PATH env
    if pdf is not None:
        dest = os.path.join(os.getcwd(), pdf.filename)
        with open(dest, "wb") as f:
            f.write(await pdf.read())
        pdf_path = dest
    else:
        pdf_path = os.environ.get("PDF_PATH")
        if not pdf_path or not os.path.exists(pdf_path):
            return JSONResponse({"error": "No PDF provided and PDF_PATH not set or missing."}, status_code=400)

    ensure_index()
    ingest_pdf(pdf_path)
    return {"status": "ingested", "pdf_path": pdf_path}


@app.post("/chat")
async def chat(req: ChatRequest):
    retriever = load_retriever()
    # Get retrieved docs
    try:
        docs = retriever.get_relevant_documents(req.question)
    except Exception:
        # some retrievers use get_relevant_documents, others use get_documents
        docs = retriever.get_documents(req.question)

    # Build a small context string with sources
    sources = []
    for d in docs:
        src = None
        if isinstance(d.metadata, dict) and 'source' in d.metadata:
            src = d.metadata['source']
        elif isinstance(d.metadata, str):
            src = d.metadata
        sources.append({"source": src, "text": d.page_content[:400]})

    # dedupe sources by text to avoid returning repeated identical chunks
    seen_texts = set()
    deduped_sources = []
    for s in sources:
        t = s.get('text')
        if t in seen_texts:
            continue
        seen_texts.add(t)
        deduped_sources.append(s)

    try:
        answer = chat_with_retriever(req.question, retriever)
    except EnvironmentError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Internal error: {e}"}, status_code=500)

    answer['sources'] = deduped_sources
    return JSONResponse(answer)
