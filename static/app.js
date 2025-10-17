const chatEl = document.getElementById('chat');
const frm = document.getElementById('frm');
const q = document.getElementById('q');
const ingestBtn = document.getElementById('ingest');
const filein = document.getElementById('filein');

function append(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
  d.textContent = (role === 'user' ? 'You: ' : 'Bot: ') + text;
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
}

ingestBtn.onclick = async () => {
  const file = filein.files[0];
  const form = new FormData();
  if (file) form.append('pdf', file);
  append('user', 'Uploading and ingesting PDF...');
  const res = await fetch('/ingest', { method: 'POST', body: form });
  const j = await res.json();
  append('bot', JSON.stringify(j));
  // Refresh stats and ingestion log after ingest
  await refreshStats();
  await refreshIngestedPdfs();
};

frm.onsubmit = async (e) => {
  e.preventDefault();
  const question = q.value.trim();
  if (!question) return;
  append('user', question);
  q.value = '';
  const res = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }) });
  const j = await res.json();
  append('bot', j.answer || JSON.stringify(j));
  if (j.sources && j.sources.length) {
    const s = document.createElement('div');
    s.id = 'sources';
    s.innerHTML = '<strong>Sources:</strong>' + '<ul>' + j.sources.map(it => `<li><em>${it.source}</em>: ${it.text}</li>`).join('') + '</ul>';
    chatEl.appendChild(s);
  }
};

// Load and render configuration on page load
async function renderConfig() {
  try {
    const res = await fetch('/config');
    const j = await res.json();
    document.getElementById('cfg').textContent = JSON.stringify(j, null, 2);
  } catch (e) {
    document.getElementById('cfg').textContent = 'Error loading config: ' + e.message;
  }
}

// Load and render ingested PDFs
async function refreshIngestedPdfs() {
  try {
    const res = await fetch('/ingestion-status');
    const j = await res.json();
    const list = j.ingested || [];
    const container = document.getElementById('ingested');
    if (list.length === 0) {
      container.innerHTML = '<p><em>No PDFs ingested yet.</em></p>';
    } else {
      container.innerHTML = '<ul>' + list.map(item => `<li>${item.filename} <em>(${new Date(item.timestamp).toLocaleString()})</em></li>`).join('') + '</ul>';
    }
  } catch (e) {
    document.getElementById('ingested').innerHTML = '<p><em>Error loading ingestion status</em></p>';
  }
}

// Load and render embedding statistics
async function refreshStats() {
  try {
    const res = await fetch('/embeddings/status');
    const j = await res.json();
    const tbody = document.getElementById('embeddingStatsBody');
    
    if (j.error) {
      tbody.innerHTML = `<tr><td colspan="2"><em>Error: ${j.error}</em></td></tr>`;
      return;
    }
    
    const rows = Object.entries(j.tables || {}).map(([table, count]) => 
      `<tr style="border-bottom:1px solid #eee"><td style="padding:6px">${table}</td><td style="padding:6px">${count}</td></tr>`
    ).join('');
    tbody.innerHTML = rows || '<tr><td colspan="2"><em>No tables found</em></td></tr>';
  } catch (e) {
    document.getElementById('embeddingStatsBody').innerHTML = `<tr><td colspan="2"><em>Error: ${e.message}</em></td></tr>`;
  }
}

// Delete embeddings
document.getElementById('delete').onclick = async () => {
  if (!confirm('Are you sure you want to delete all embeddings? This cannot be undone.')) return;
  
  try {
    const res = await fetch('/embeddings/delete', { method: 'POST' });
    const j = await res.json();
    document.getElementById('deleteResult').textContent = JSON.stringify(j);
    // Refresh stats and ingestion log after deletion
    setTimeout(async () => {
      await refreshStats();
      await refreshIngestedPdfs();
    }, 500);
  } catch (e) {
    document.getElementById('deleteResult').textContent = 'Error: ' + e.message;
  }
};

// Initialize page: load config, ingestion status, and stats
async function initPage() {
  await renderConfig();
  await refreshIngestedPdfs();
  await refreshStats();
}

initPage();
