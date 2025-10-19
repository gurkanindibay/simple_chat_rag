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
  // Wire up window helpers when MSAL instance becomes available
  try { _maybeExposeAuthSim(); } catch (e) { /* ignore */ }
};

// Expose msal instance on window for manual debugging in the browser console
export const exposeMsalToWindow = () => {
  if (typeof window !== 'undefined' && msalInstanceRef) {
    window.__msal = msalInstanceRef;
    try { _maybeExposeAuthSim(); } catch (e) { /* ignore */ }
  }
};

// --- Token refresh / expiry simulation helpers ---
// These helpers let you force MSAL to refresh tokens or simulate expiry by
// removing the cached access token entry. Useful for debugging refresh logic.
let _periodicExpiryTimer = null;

/**
 * Force MSAL to acquire a fresh access token using acquireTokenSilent with forceRefresh:true.
 * Returns the access token string (or null on failure) and logs the result.
 */
export const forceRefreshTokenAndLog = async () => {
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
    const scopesToRequest = (tokenRequest && tokenRequest.scopes && tokenRequest.scopes.length)
      ? tokenRequest.scopes
      : ['User.Read'];
    console.info('[auth-sim] Forcing token refresh (acquireTokenSilent forceRefresh:true) for scopes:', scopesToRequest);
    const response = await msalInstanceRef.acquireTokenSilent({ scopes: scopesToRequest, account: accounts[0], forceRefresh: true });
    console.info('[auth-sim] Forced refresh response:', response);
    return response?.accessToken || null;
  } catch (err) {
    console.error('[auth-sim] Forced refresh failed, falling back to popup:', err);
    try {
      const response = await msalInstanceRef.acquireTokenPopup({ scopes: tokenRequest.scopes });
      console.info('[auth-sim] Popup response after forced refresh failure:', response);
      return response?.accessToken || null;
    } catch (popupErr) {
      console.error('[auth-sim] Popup also failed:', popupErr);
      return null;
    }
  }
};

/**
 * Remove the cached access token entry from MSAL cache so that next acquireTokenSilent
 * will attempt to refresh it (or fall back to interactive). Works with sessionStorage cacheLocation.
 * This simulates token expiry without waiting for the real expiry time.
 */
export const expireCachedAccessToken = () => {
  if (!msalInstanceRef) return false;
  try {
    // MSAL stores tokens in sessionStorage/localStorage under keys containing 'msal' and account/authority
    // We'll remove access token entries that include 'accessToken' to simulate expiry.
    const storage = msalInstanceRef.getLogger ? window.sessionStorage : window.sessionStorage;
    const keysToRemove = [];
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (!key) continue;
      // MSAL v2 keys typically include 'accessToken' or 'accesstoken'
      if (key.toLowerCase().includes('accesstoken') || key.toLowerCase().includes('access.token')) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(k => {
      console.debug('[auth-sim] Removing MSAL cache key to simulate expiry:', k);
      storage.removeItem(k);
    });
    return keysToRemove.length > 0;
  } catch (e) {
    console.error('[auth-sim] Failed to expire cached token:', e);
    return false;
  }
};

/**
 * Start a periodic job that expires cached access tokens every `intervalMs` milliseconds
 * (default 30000 = 30s). Returns true if started, false if already running or MSAL not set.
 */
export const startPeriodicExpiry = (intervalMs = 30000) => {
  if (!msalInstanceRef) {
    console.warn('MSAL instance not set; cannot start periodic expiry');
    return false;
  }
  if (_periodicExpiryTimer) {
    console.warn('Periodic expiry already running');
    return false;
  }
  console.info('[auth-sim] Starting periodic token expiry every', intervalMs, 'ms');
  _periodicExpiryTimer = setInterval(() => {
    const removed = expireCachedAccessToken();
    if (removed) {
      // After expiring cache, force a fresh token to be requested and logged (non-blocking)
      forceRefreshTokenAndLog().catch(err => console.error('[auth-sim] forceRefresh error:', err));
    } else {
      console.debug('[auth-sim] No access token cache keys found to remove this interval');
    }
  }, intervalMs);
  // expose simple control on window for convenience
  if (typeof window !== 'undefined') {
    window.__authSim = window.__authSim || {};
    window.__authSim.startPeriodicExpiry = () => startPeriodicExpiry(intervalMs);
    window.__authSim.stopPeriodicExpiry = stopPeriodicExpiry;
    window.__authSim.expireCachedAccessToken = expireCachedAccessToken;
    window.__authSim.forceRefreshTokenAndLog = forceRefreshTokenAndLog;
  }
  return true;
};

/**
 * Stop the periodic expiry job if running.
 */
export const stopPeriodicExpiry = () => {
  if (_periodicExpiryTimer) {
    clearInterval(_periodicExpiryTimer);
    _periodicExpiryTimer = null;
    console.info('[auth-sim] Stopped periodic token expiry');
    return true;
  }
  console.warn('[auth-sim] No periodic expiry was running');
  return false;
};

// Expose helpers automatically to the window when MSAL is set
const _maybeExposeAuthSim = () => {
  if (typeof window === 'undefined') return;
  window.__authSim = window.__authSim || {};
  window.__authSim.expireCachedAccessToken = expireCachedAccessToken;
  window.__authSim.forceRefreshTokenAndLog = forceRefreshTokenAndLog;
  window.__authSim.startPeriodicExpiry = startPeriodicExpiry;
  window.__authSim.stopPeriodicExpiry = stopPeriodicExpiry;
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

    console.debug('[auth] msalInstanceRef:', msalInstanceRef);
    console.debug('[auth] accounts:', accounts);
    console.debug('[auth] scopesToRequest:', scopesToRequest);

    const response = await msalInstanceRef.acquireTokenSilent({
      scopes: scopesToRequest,
      account: accounts[0],
    });
    console.debug('[auth] acquireTokenSilent response:', response);
    return response.accessToken;
  } catch (error) {
    console.error('Error acquiring token silently:', error);
    // If silent token acquisition fails, try interactive popup
    try {
      const scopesToRequest = (tokenRequest && tokenRequest.scopes && tokenRequest.scopes.length)
        ? tokenRequest.scopes
        : ['User.Read'];
      console.debug('[auth] Attempting interactive token acquisition with scopes:', scopesToRequest);
      const response = await msalInstanceRef.acquireTokenPopup({
        scopes: scopesToRequest,
      });
      console.debug('[auth] acquireTokenPopup response:', response);
      return response.accessToken;
    } catch (interactiveError) {
      console.error('Interactive token acquisition failed:', interactiveError);
      return null;
    }
  }
};

// Helper for manual debug from browser console
export const debugAuth = async () => {
  if (!msalInstanceRef) return { error: 'msal instance not set' };
  const accounts = msalInstanceRef.getAllAccounts();
  let tokenInfo = null;
  try {
    const scopesToRequest = (tokenRequest && tokenRequest.scopes && tokenRequest.scopes.length)
      ? tokenRequest.scopes
      : ['User.Read'];
    tokenInfo = await msalInstanceRef.acquireTokenSilent({ scopes: scopesToRequest, account: accounts[0] });
  } catch (e) {
    tokenInfo = { error: e?.message || String(e) };
  }
  return { accounts, tokenInfo };
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
    const headers = await getAuthHeaders();
    console.debug('[api] getConfig headers:', headers);
    const response = await fetch(url, { headers });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  },

  async getIngestionStatus() {
    const url = `${API_BASE_URL}/ingestion-status`;
    console.log('Fetching:', url);
    const headers = await getAuthHeaders();
    console.debug('[api] getIngestionStatus headers:', headers);
    const response = await fetch(url, { headers });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  },

  async getEmbeddingsStatus() {
    const url = `${API_BASE_URL}/embeddings/status`;
    console.log('Fetching:', url);
    const headers = await getAuthHeaders();
    console.debug('[api] getEmbeddingsStatus headers:', headers);
    const response = await fetch(url, { headers });
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
