import { useState } from 'react';
import { useMsal } from '@azure/msal-react';

export function ChatHeader({ userAccount }) {
  const { instance, accounts } = useMsal();
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  // Prefer passed userAccount, otherwise use first account from MSAL
  const acct = userAccount || (accounts && accounts[0]) || null;

  // Derive a friendly display name from common fields
  const displayName = acct
    ? (acct.name || acct.username || (acct.idTokenClaims && (acct.idTokenClaims.name || acct.idTokenClaims.preferred_username)) || acct.homeAccountId)
    : null;

  const handleLogout = async () => {
    if (isLoggingOut) return; // Prevent multiple clicks
    
    setIsLoggingOut(true);
    
    try {
      const logoutMarker = Date.now().toString();

      // Set global logout flag using localStorage (works across same browser)
      localStorage.setItem('msal_global_logout', logoutMarker);
      localStorage.setItem('msal_global_logout_processed', logoutMarker);
      
      // Also try cookie for cross-origin (will be rejected on localhost but that's ok)
      try {
        document.cookie = `msal_global_logout=${logoutMarker}; path=/; max-age=300`;
      } catch (cookieError) {
        console.warn('Unable to set msal_global_logout cookie during logout.', cookieError);
      }
      
      // Clear all MSAL-related localStorage and cookies for global logout across all apps
      const keysToRemove = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && (key.includes('msal') || key.includes('login.windows')) && key !== 'msal_global_logout' && key !== 'msal_global_logout_processed') {
          keysToRemove.push(key);
        }
      }
      keysToRemove.forEach(key => localStorage.removeItem(key));
      
      // Clear MSAL-related cookies
      document.cookie.split(';').forEach(cookie => {
        const cookieName = cookie.split('=')[0].trim();
        if ((cookieName.includes('msal') || cookieName.includes('login.windows')) && cookieName !== 'msal_global_logout') {
          document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
        }
      });
      
      await instance.logoutPopup({
        postLogoutRedirectUri: "/",
      });
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setIsLoggingOut(false);
    }
  };

  return (
    <div className="chat-header">
      <div className="chat-header-left">
        <i className="fas fa-robot"></i>
        <h1>RAG Chat Assistant</h1>
      </div>
      {acct && (
        <div className="chat-header-right">
          <div className="user-info">
            <span className="user-name">{displayName || 'User'}</span>
            <button 
              className="logout-button" 
              onClick={handleLogout} 
              disabled={isLoggingOut}
              title={isLoggingOut ? "Signing out..." : "Sign out"}
            >
              <i className={`fas ${isLoggingOut ? 'fa-spinner fa-spin' : 'fa-sign-out-alt'}`}></i>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
