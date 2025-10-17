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

// optional: show a small banner if providers are set (fetch /config on load)
async function showProviders(){
  try{
    const res = await fetch('/config');
    if(!res.ok) return;
    const j = await res.json();
    const b = document.createElement('div');
    b.style.fontSize='0.9em'; b.style.color='#666'; b.style.marginTop='8px';
    b.textContent = `Providers — Embeddings: ${j.EMBEDDING_PROVIDER || 'N/A'} | LLM: ${j.LLM_PROVIDER || 'N/A'}`;
    document.body.insertBefore(b, document.body.firstChild);
  }catch(e){ /* ignore */ }
}

showProviders();
