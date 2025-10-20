import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { PublicClientApplication } from '@azure/msal-browser'
import { MsalProvider } from '@azure/msal-react'
import { msalConfig } from './authConfig'
import { setMsalInstance, exposeMsalToWindow } from './api/client'
import '@fortawesome/fontawesome-free/css/all.min.css'
import './index.css'
import App from './App.jsx'

// Initialize MSAL instance
const msalInstance = new PublicClientApplication(msalConfig);

const initAndRender = async () => {
  console.log('[MSAL Init] Starting MSAL initialization...');
  
  // register msal instance for api client and debugging
  setMsalInstance(msalInstance);
  exposeMsalToWindow();

  try {
    // MSAL v3+ requires calling initialize() first
    console.log('[MSAL Init] Calling initialize...');
    await msalInstance.initialize();
    console.log('[MSAL Init] Initialize complete');
    
    // Wait for any redirect result to be handled before rendering
    console.log('[MSAL Init] Calling handleRedirectPromise...');
    const redirectResult = await msalInstance.handleRedirectPromise();
    console.log('[MSAL Init] handleRedirectPromise result:', redirectResult);

    const accounts = msalInstance.getAllAccounts();
    console.log('[MSAL Init] Accounts after redirect:', accounts);
    
    if (accounts && accounts.length > 0) {
      console.log('[MSAL Init] ✓ Found', accounts.length, 'account(s), setting app_logged_in flag');
      try { localStorage.setItem('app_logged_in', '1'); } catch (e) { /* ignore */ }
    } else {
      // If previously logged in, attempt silent SSO restoration
      const tried = localStorage.getItem('app_logged_in');
      console.log('[MSAL Init] No accounts found. app_logged_in flag:', tried);
      
      if (tried) {
        try {
          console.log('[MSAL Init] Attempting ssoSilent...');
          if (typeof msalInstance.ssoSilent === 'function') {
            await msalInstance.ssoSilent();
          } else {
            console.log('[MSAL Init] ssoSilent not available');
          }
          const newAccounts = msalInstance.getAllAccounts();
          console.log('[MSAL Init] Accounts after ssoSilent:', newAccounts);
          
          if (newAccounts && newAccounts.length > 0) {
            try { localStorage.setItem('app_logged_in', '1'); } catch (e) { /* ignore */ }
          } else {
            console.log('[MSAL Init] No accounts after ssoSilent, clearing flag');
            try { localStorage.removeItem('app_logged_in'); } catch (e) { /* ignore */ }
          }
        } catch (e) {
          console.warn('[MSAL Init] ssoSilent failed:', e);
          try { localStorage.removeItem('app_logged_in'); } catch (er) { /* ignore */ }
        }
      }
    }
  } catch (err) {
    console.error('[MSAL Init] Error during initialization:', err);
  }
  
  console.log('[MSAL Init] Final account state:', msalInstance.getAllAccounts());

  // Now render the app
  createRoot(document.getElementById('root')).render(
    <StrictMode>
      <MsalProvider instance={msalInstance}>
        <App />
      </MsalProvider>
    </StrictMode>,
  );
};

initAndRender();
