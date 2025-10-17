// Determine API URL based on environment
let API_BASE_URL = '';

if (import.meta.env.VITE_API_URL) {
  API_BASE_URL = import.meta.env.VITE_API_URL;
} else if (import.meta.env.DEV) {
  // Development: use Vite proxy (relative paths)
  API_BASE_URL = '';
} else {
  // Production: use current domain or fallback
  API_BASE_URL = window.location.origin;
}

console.log('Environment:', {
  isDev: import.meta.env.DEV,
  isProd: import.meta.env.PROD,
  VITE_API_URL: import.meta.env.VITE_API_URL,
  API_BASE_URL
});

export const apiClient = {
  async getConfig() {
    const url = `${API_BASE_URL}/config`;
    console.log('Fetching:', url);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  },

  async getIngestionStatus() {
    const url = `${API_BASE_URL}/ingestion-status`;
    console.log('Fetching:', url);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  },

  async getEmbeddingsStatus() {
    const url = `${API_BASE_URL}/embeddings/status`;
    console.log('Fetching:', url);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  },

  async chat(question) {
    const url = `${API_BASE_URL}/chat`;
    console.log('Posting to:', url, { question });
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Chat error:', response.status, errorText);
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }
    const data = await response.json();
    console.log('Chat response:', data);
    return data;
  },

  async ingestPDF(file) {
    const formData = new FormData();
    formData.append('pdf', file);
    const response = await fetch(`${API_BASE_URL}/ingest`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  },

  async deleteEmbeddings() {
    const response = await fetch(`${API_BASE_URL}/embeddings/delete`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  },
};
