import { useState } from 'react';
import { useMsal } from '@azure/msal-react';
import { setGlobalLogoutFlags } from '../utils/logout';

/**
 * Custom hook for handling logout using MSAL's official logout APIs.
 * 
 * According to Microsoft documentation:
 * https://learn.microsoft.com/en-us/entra/msal/javascript/browser/logout
 * 
 * logoutPopup() and logoutRedirect() automatically:
 * 1. Clear the MSAL cache (tokens, accounts, etc.)
 * 2. Clear the session on the identity server
 * 
 * No manual cache clearing is needed!
 * 
 * @param {Object} options - Configuration options
 * @param {string} options.logoutType - Type of logout: 'popup' or 'redirect' (default: 'popup')
 * @param {string} options.postLogoutRedirectUri - URI to redirect after logout (default: window.location.origin)
 * @returns {Object} - { logout: function, isLoggingOut: boolean }
 */
export function useLogout(options = {}) {
  const { 
    logoutType = 'popup', 
    postLogoutRedirectUri = window.location.origin
  } = options;
  
  const { instance } = useMsal();
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const logout = async () => {
    if (isLoggingOut) {
      console.warn('[useLogout] Logout already in progress, ignoring duplicate call');
      return;
    }
    
    setIsLoggingOut(true);
    
    try {
      // Step 1: Set global logout flags for cross-app coordination
      // NOTE: This is a custom feature, not provided by MSAL
      setGlobalLogoutFlags();
      
      // Step 2: Get the current account for proper logout
      const currentAccount = instance.getActiveAccount();
      
      // Step 3: Perform MSAL logout using official API
      // This automatically clears cache and server session
      const logoutRequest = {
        account: currentAccount, // Important: ensures proper account cleanup
        postLogoutRedirectUri: postLogoutRedirectUri,
      };
      
      if (logoutType === 'redirect') {
        console.log('[useLogout] Performing logout redirect...');
        // Note: logoutRedirect promise may not resolve before redirect happens
        await instance.logoutRedirect(logoutRequest);
      } else {
        console.log('[useLogout] Performing logout popup...');
        // For popup, we can add mainWindowRedirectUri if needed
        await instance.logoutPopup({
          ...logoutRequest,
          mainWindowRedirectUri: postLogoutRedirectUri, // Optional: redirect main window after popup closes
        });
      }
      
      console.log('[useLogout] Logout completed successfully');
    } catch (error) {
      console.error('[useLogout] Logout error:', error);
      
      // Even if logout fails, try to clean up local state
      console.warn('[useLogout] Logout failed, but local cleanup may have occurred');
      throw error;
    } finally {
      setIsLoggingOut(false);
    }
  };

  return { logout, isLoggingOut };
}
