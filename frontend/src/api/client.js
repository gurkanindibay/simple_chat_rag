import { tokenRequest } from '../authConfig'

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

// Store the MSAL instance reference
let msalInstanceRef = null;

export const setMsalInstance = (instance) => {
  msalInstanceRef = instance;
};

const getAuthToken = async () => {
  if (!msalInstanceRef) {
    console.warn('MSAL instance not set');
    return null;
  }

  const accounts = msalInstanceRef.getAllAccounts();
  if (accounts.length === 0) {
    console.warn('No accounts found');
    return null;
  }

  try {
    // Prefer requesting the API scope defined in authConfig (falls back to User.Read)
    const scopesToRequest = (tokenRequest && tokenRequest.scopes && tokenRequest.scopes.length)
      ? tokenRequest.scopes
      : ['User.Read'];

    const response = await msalInstanceRef.acquireTokenSilent({
      scopes: scopesToRequest,
      account: accounts[0],
    });
    return response.accessToken;
  } catch (error) {
    console.error('Error acquiring token silently:', error);
    // If silent token acquisition fails, try interactive popup
    try {
      const scopesToRequest = (tokenRequest && tokenRequest.scopes && tokenRequest.scopes.length)
        ? tokenRequest.scopes
        : ['User.Read'];

      const response = await msalInstanceRef.acquireTokenPopup({
        scopes: scopesToRequest,
      });
      return response.accessToken;
    } catch (interactiveError) {
      console.error('Interactive token acquisition failed:', interactiveError);
      return null;
    }
  }
};

const getAuthHeaders = async () => {
  const token = await getAuthToken();
  const headers = {
    'Content-Type': 'application/json',
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  return headers;
};

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
    const headers = await getAuthHeaders();
    const response = await fetch(url, {
      method: 'POST',
      headers,
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
    const token = await getAuthToken();
    const headers = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    const response = await fetch(`${API_BASE_URL}/ingest`, {
      method: 'POST',
      headers,
      body: formData,
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  },

  async deleteEmbeddings() {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_BASE_URL}/embeddings/delete`, {
      method: 'POST',
      headers,
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  },

  async updateConfig(key, value) {
    const url = `${API_BASE_URL}/config/update`;
    console.log('Updating config:', { key, value });
    const headers = await getAuthHeaders();
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({ key, value }),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      console.error('Config update error:', response.status, errorData);
      throw new Error(errorData.error || `API error: ${response.status}`);
    }
    const data = await response.json();
    console.log('Config update response:', data);
    return data;
  },
};
