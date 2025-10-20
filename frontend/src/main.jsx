import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { PublicClientApplication } from '@azure/msal-browser'
import { MsalProvider } from '@azure/msal-react'
import { msalConfig } from './authConfig'
import { setMsalInstance, exposeMsalToWindow } from './api/client'
import { restoreActiveAccountFromCache } from './msalCacheHelper'
import '@fortawesome/fontawesome-free/css/all.min.css'
import './index.css'
import App from './App.jsx'

// Initialize MSAL instance
const msalInstance = new PublicClientApplication(msalConfig);

const initAndRender = async () => {
  console.log('[MSAL Init] Starting MSAL initialization...');
  console.log('[MSAL Init] Config:', {
    clientId: msalConfig.auth.clientId,
    authority: msalConfig.auth.authority,
    redirectUri: msalConfig.auth.redirectUri,
    cacheLocation: msalConfig.cache.cacheLocation
  });
  
  // Check cache BEFORE initialization
  const keysBeforeInit = Object.keys(localStorage).filter(k => k.includes('msal'));
  console.log('[MSAL Init] localStorage keys BEFORE init:', keysBeforeInit.length, 'keys');
  if (keysBeforeInit.length > 0) {
    console.log('[MSAL Init] Keys before:', keysBeforeInit.slice(0, 5));
  }
  
  // register msal instance for api client and debugging
  setMsalInstance(msalInstance);
  exposeMsalToWindow();

  try {
    // MSAL v3+ requires calling initialize() first
    console.log('[MSAL Init] Calling initialize...');
    await msalInstance.initialize();
    console.log('[MSAL Init] Initialize complete');
    
    // Check if initialize cleared anything
    const keysAfterInit = Object.keys(localStorage).filter(k => k.includes('msal'));
    console.log('[MSAL Init] localStorage keys AFTER init:', keysAfterInit.length, 'keys');
    if (keysBeforeInit.length !== keysAfterInit.length) {
      console.warn('[MSAL Init] ⚠️ Cache size changed during init!', 
        'Before:', keysBeforeInit.length, 'After:', keysAfterInit.length);
    }
    
    // Small delay to allow MSAL to decrypt cache from localStorage
    // MSAL v4+ uses encrypted cache which needs time to decrypt
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Wait for any redirect result to be handled before rendering
    console.log('[MSAL Init] Calling handleRedirectPromise...');
    const redirectResult = await msalInstance.handleRedirectPromise();
    console.log('[MSAL Init] handleRedirectPromise result:', redirectResult);
    
    // If we got a result from redirect, set the account
    if (redirectResult && redirectResult.account) {
      console.log('[MSAL Init] ✓ Login successful via redirect, account:', redirectResult.account.username);
      msalInstance.setActiveAccount(redirectResult.account);
      try { localStorage.setItem('app_logged_in', '1'); } catch (e) { /* ignore */ }
    }

    // Check if there's cached account data before calling getAllAccounts
    const accountKeys = localStorage.getItem('msal.1.account.keys');
    console.log('[MSAL Init] Account keys in cache:', accountKeys);
    
    let accounts = msalInstance.getAllAccounts();
    console.log('[MSAL Init] Accounts after redirect:', accounts);
    
    // If no accounts but we have account keys, MSAL cache isn't loading properly
    if (accounts.length === 0 && accountKeys) {
      console.log('[MSAL Init] ⚠️ No accounts loaded but cache exists! This is the encrypted cache issue.');
      console.log('[MSAL Init] Waiting longer for cache decryption...');
      
      // Try multiple times with increasing delays to let MSAL decrypt the cache
      for (let i = 0; i < 5 && accounts.length === 0; i++) {
        await new Promise(resolve => setTimeout(resolve, 100 * (i + 1)));
        accounts = msalInstance.getAllAccounts();
        console.log(`[MSAL Init] Attempt ${i + 1}: ${accounts.length} accounts found`);
      }
      
      // If still no accounts, try the manual restore
      if (accounts.length === 0) {
        console.log('[MSAL Init] Attempting manual account restoration from cache...');
        restoreActiveAccountFromCache(msalInstance);
        accounts = msalInstance.getAllAccounts();
        console.log('[MSAL Init] Accounts after manual restore:', accounts.length);
      }
    }
    
    if (accounts && accounts.length > 0) {
      console.log('[MSAL Init] ✓ Found', accounts.length, 'account(s), setting app_logged_in flag');
      // Set the active account so MSAL knows which account to use
      if (!msalInstance.getActiveAccount()) {
        msalInstance.setActiveAccount(accounts[0]);
        console.log('[MSAL Init] Active account set to:', accounts[0].username);
      } else {
        console.log('[MSAL Init] Active account already set to:', msalInstance.getActiveAccount().username);
      }
      try { localStorage.setItem('app_logged_in', '1'); } catch (e) { /* ignore */ }
    } else {
      // If previously logged in, attempt silent SSO restoration
      const tried = localStorage.getItem('app_logged_in');
      console.log('[MSAL Init] No accounts found. app_logged_in flag:', tried);
      
      if (tried) {
        try {
          console.log('[MSAL Init] Attempting ssoSilent with basic scopes...');
          if (typeof msalInstance.ssoSilent === 'function') {
            // ssoSilent needs a proper request object with scopes
            await msalInstance.ssoSilent({ scopes: ['openid', 'profile'] });
          } else {
            console.log('[MSAL Init] ssoSilent not available');
          }
          const newAccounts = msalInstance.getAllAccounts();
          console.log('[MSAL Init] Accounts after ssoSilent:', newAccounts);
          
          if (newAccounts && newAccounts.length > 0) {
            msalInstance.setActiveAccount(newAccounts[0]);
            console.log('[MSAL Init] Active account restored via ssoSilent:', newAccounts[0].username);
            try { localStorage.setItem('app_logged_in', '1'); } catch (e) { /* ignore */ }
          } else {
            console.log('[MSAL Init] No accounts after ssoSilent, clearing flag');
            try { localStorage.removeItem('app_logged_in'); } catch (e) { /* ignore */ }
          }
        } catch (e) {
          console.warn('[MSAL Init] ssoSilent failed (this is normal if you need to re-login):', e.message || e);
          try { localStorage.removeItem('app_logged_in'); } catch (er) { /* ignore */ }
        }
      }
    }
  } catch (err) {
    console.error('[MSAL Init] Error during initialization:', err);
  }
  
  console.log('[MSAL Init] Final account state:', msalInstance.getAllAccounts());
  
  // Debug: Show what's in localStorage
  const msalKeys = Object.keys(localStorage).filter(key => key.includes('msal') || key.includes('login.windows'));
  console.log('[MSAL Init] MSAL localStorage keys:', msalKeys.length, 'keys found');
  if (msalKeys.length > 0) {
    console.log('[MSAL Init] Keys:', msalKeys);
    // Show content of keys to understand what's stored
    msalKeys.forEach(key => {
      try {
        const value = localStorage.getItem(key);
        const parsed = JSON.parse(value);
        console.log(`[MSAL Init] ${key}:`, typeof parsed === 'object' ? Object.keys(parsed) : parsed);
      } catch (e) {
        console.log(`[MSAL Init] ${key}: (not JSON or error)`);
      }
    });
  }

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
