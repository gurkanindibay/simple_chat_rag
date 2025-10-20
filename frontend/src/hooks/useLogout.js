import { useState } from 'react';
import { useMsal } from '@azure/msal-react';
import { setGlobalLogoutFlags } from '../utils/logout';

/**
 * Custom hook for handling logout across all applications.
 * Sets global logout flags so other apps on the same browser can detect the logout.
 * 
 * @param {Object} options - Configuration options
 * @param {string} options.logoutType - Type of logout: 'popup' or 'redirect' (default: 'popup')
 * @param {string} options.postLogoutRedirectUri - URI to redirect after logout (default: '/')
 * @returns {Object} - { logout: function, isLoggingOut: boolean }
 */
export function useLogout(options = {}) {
  const { 
    logoutType = 'popup', 
    postLogoutRedirectUri = '/' 
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
      // Set global logout flags for cross-app coordination
      setGlobalLogoutFlags();
      
      // Perform MSAL logout based on the specified type
      if (logoutType === 'redirect') {
        console.log('[useLogout] Performing logout redirect...');
        await instance.logoutRedirect({ postLogoutRedirectUri });
      } else {
        console.log('[useLogout] Performing logout popup...');
        await instance.logoutPopup({ postLogoutRedirectUri });
      }
      
      console.log('[useLogout] Logout completed successfully');
    } catch (error) {
      console.error('[useLogout] Logout error:', error);
      throw error; // Re-throw so calling component can handle if needed
    } finally {
      setIsLoggingOut(false);
    }
  };

  return { logout, isLoggingOut };
}
