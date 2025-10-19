import React, { useEffect, useState } from 'react';

// Determine API URL (same logic as api/client.js)
const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  } else if (import.meta.env.DEV) {
    return 'http://localhost:8000'; // Backend default in dev
  } else {
    return window.location.origin;
  }
};

export function DebugPanel() {
  const [claims, setClaims] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isCollapsed, setIsCollapsed] = useState(() => {
    try {
      const s = localStorage.getItem('debugPanelCollapsed');
      return s === null ? true : s === 'true';
    } catch (e) {
      return true;
    }
  });

  // Persist collapse state
  useEffect(() => {
    try { localStorage.setItem('debugPanelCollapsed', String(isCollapsed)); } catch (e) { /* ignore */ }
  }, [isCollapsed]);

  const fetchClaims = async () => {
    setLoading(true);
    setError(null);
    try {
      // Get access token using the helper
      let token = null;
      if (window.__authSim && window.__authSim.forceRefreshTokenAndLog) {
        token = await window.__authSim.forceRefreshTokenAndLog();
      }

      if (!token) {
        throw new Error('No access token available. Please sign in.');
      }

      const apiBaseUrl = getApiBaseUrl();
      const res = await fetch(`${apiBaseUrl}/auth/claims`, { 
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        } 
      });
      
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text.substring(0, 100)}`);
      }
      
      const json = await res.json();
      setClaims(json.claims || json);
    } catch (e) {
      setError(String(e));
      setClaims(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`debug-panel ${isCollapsed ? 'debug-panel-collapsed' : 'debug-panel-open'}`}>
      <div
        className="debug-panel-header"
        role="button"
        tabIndex={0}
        onClick={() => setIsCollapsed(!isCollapsed)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setIsCollapsed(!isCollapsed); }}
        aria-expanded={!isCollapsed}
      >
        <div className="debug-panel-title">Debug Panel</div>
        <div className={`debug-panel-chevron ${isCollapsed ? 'chev-collapsed' : 'chev-open'}`} aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>

      <div className="debug-panel-content" aria-hidden={isCollapsed}>
        <div className="debug-panel-controls">
          <button className="btn-primary" onClick={fetchClaims} disabled={loading}>
            {loading ? 'Loading...' : '🔍 Show Claims'}
          </button>
          <div className="debug-actions">
            <button onClick={() => window.__authSim?.startPeriodicExpiry?.(30000)} title="Start periodic token expiry every 30s">
              ▶️ Start 30s
            </button>
            <button onClick={() => window.__authSim?.stopPeriodicExpiry?.()} title="Stop periodic expiry">
              ⏹️ Stop
            </button>
            <button onClick={() => { window.__authSim?.expireCachedAccessToken?.(); setClaims(null); }} title="Expire cached token now">
              ⚡ Expire
            </button>
          </div>
        </div>

        {(error || claims) && (
          <div className="debug-panel-body">
            {error && <div className="debug-error">❌ Error: {error}</div>}
            {claims && <pre className="debug-claims">{JSON.stringify(claims, null, 2)}</pre>}
          </div>
        )}
      </div>
    </div>
  );
}
