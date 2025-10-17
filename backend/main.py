import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.ingestion import ensure_index, ingest_pdf, load_retriever, chat_with_retriever
from backend.ingestion import delete_embeddings, log_ingestion, get_ingestion_log, get_embedding_table_stats

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")


class ChatRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("./static/index.html")


@app.get("/config")
async def config():
    # Return current provider configuration (for UI/debug)
    cfg = {
        "EMBEDDING_PROVIDER": os.environ.get("EMBEDDING_PROVIDER"),
        "LLM_PROVIDER": os.environ.get("LLM_PROVIDER"),
    }
    return cfg


@app.get("/info", response_class=HTMLResponse)
async def info_page():
    """Serve the info page with embedded configuration, ingestion, and embedding status."""
    import json
    
    # Fetch current config, ingestion data, and embedding stats on server side
    cfg = {
        "EMBEDDING_PROVIDER": os.environ.get("EMBEDDING_PROVIDER"),
        "LLM_PROVIDER": os.environ.get("LLM_PROVIDER"),
    }
    ingested = get_ingestion_log()
    
    # Get embedding table statistics
    try:
        embedding_stats = get_embedding_table_stats()
    except Exception as e:
        embedding_stats = {"error": str(e)}
    
    # Embed data as JSON in the page so JS has it immediately available
    cfg_json = json.dumps(cfg, indent=2)
    ingested_json = json.dumps(ingested)
    stats_json = json.dumps(embedding_stats)
    
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>RAG Chat — Info</title>
    <style>
      body {{ font-family: system-ui, sans-serif; max-width:800px;margin:40px auto;color:#111 }}
      pre {{ background:#f7f7f7; padding:12px; border:1px solid #eee }}
      button {{ margin-top:8px }}
      .stat-table {{ margin:12px 0 }}
      .stat-table tr {{ border-bottom:1px solid #ddd }}
      .stat-table td {{ padding:8px; }}
    </style>
  </head>
  <body>
    <h1>Configuration & Embeddings</h1>
    <p>This page shows current provider config and lets you delete stored embeddings (PGVector table).</p>

    <h2>Providers</h2>
    <pre id="cfg"></pre>

    <h2>Ingested PDFs</h2>
    <div id="ingested"></div>

    <h2>Embeddings in Database</h2>
    <table class="stat-table" id="embeddingStats">
      <thead><tr><th>Table</th><th>Row Count</th></tr></thead>
      <tbody id="embeddingStatsBody"></tbody>
    </table>

    <h2>Delete Embeddings</h2>
    <div>
      <button id="delete">Delete all embeddings (truncate table)</button>
      <div id="deleteResult"></div>
    </div>

    <p><a href="/">Back to chat</a></p>

    <script>
      // Data embedded from server
      const serverConfig = {cfg_json};
      const serverIngested = {ingested_json};
      const serverStats = {stats_json};

      function renderConfig() {{
        document.getElementById('cfg').textContent = JSON.stringify(serverConfig, null, 2);
      }}

      function renderIngestedPdfs() {{
        const list = serverIngested || [];
        const container = document.getElementById('ingested');
        if (list.length === 0) {{
          container.innerHTML = '<p><em>No PDFs ingested yet.</em></p>';
        }} else {{
          const html = '<ul>' + list.map(item => 
            `<li><strong>${{item.filename}}</strong><br/><small>${{item.pdf_path}}</small><br/><small>Ingested: ${{new Date(item.timestamp).toLocaleString()}}</small></li>`
          ).join('') + '</ul>';
          container.innerHTML = html;
        }}
      }}

      function renderEmbeddingStats() {{
        const stats = serverStats || {{}};
        const tbody = document.getElementById('embeddingStatsBody');
        
        if (stats.error) {{
          tbody.innerHTML = `<tr><td colspan="2"><em>Error: ${{stats.error}}</em></td></tr>`;
          return;
        }}
        
        const rows = Object.entries(stats).map(([table, count]) => 
          `<tr><td><strong>${{table}}</strong></td><td>${{typeof count === 'object' ? JSON.stringify(count) : count}}</td></tr>`
        ).join('');
        
        if (rows.length === 0) {{
          tbody.innerHTML = '<tr><td colspan="2"><em>No embedding tables found</em></td></tr>';
        }} else {{
          tbody.innerHTML = rows;
        }}
      }}

      async function deleteEmbeddings() {{
        if (!confirm('Delete all embeddings? This cannot be undone.')) return;
        try {{
          const res = await fetch('/embeddings/delete', {{ method: 'POST' }});
          if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
          const j = await res.json();
          const msg = j.error ? 'Error: ' + j.error : 'Deleted: ' + JSON.stringify(j.result);
          document.getElementById('deleteResult').textContent = msg;
          // Refresh stats after delete
          setTimeout(async () => {{
            try {{
              const res = await fetch('/embeddings/status');
              const j = await res.json();
              serverStats = j.tables || {{}};
              renderEmbeddingStats();
            }} catch (e) {{
              console.error('Error refreshing stats:', e);
            }}
          }}, 500);
        }} catch (e) {{
          console.error('Error deleting embeddings:', e);
          document.getElementById('deleteResult').textContent = 'Error: ' + e.message;
        }}
      }}

      // Initialize on page load
      function initPage() {{
        renderConfig();
        renderIngestedPdfs();
        renderEmbeddingStats();
        const deleteBtn = document.getElementById('delete');
        if (deleteBtn) {{
          deleteBtn.onclick = deleteEmbeddings;
        }}
      }}

      if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', initPage);
      }} else {{
        initPage();
      }}
    </script>
  </body>
</html>"""
    
    return html


@app.post("/embeddings/delete")
async def embeddings_delete():
    """Delete all embedding tables. Table selection is determined server-side (secure).
    
    Returns the names and counts of truncated tables.
    """
    try:
        deleted = delete_embeddings()
    except EnvironmentError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Internal error: {e}"}, status_code=500)

    return {"status": "deleted", "result": deleted}


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
    log_ingestion(pdf_path)
    return {"status": "ingested", "pdf_path": pdf_path}


@app.get("/ingestion-status")
async def ingestion_status():
    """Return the list of ingested PDFs."""
    return {"ingested": get_ingestion_log()}


@app.get("/embeddings/status")
async def embeddings_status():
    """Return the row counts for all embedding tables."""
    try:
        stats = get_embedding_table_stats()
        return {"status": "ok", "tables": stats}
    except EnvironmentError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Internal error: {e}"}, status_code=500)


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
