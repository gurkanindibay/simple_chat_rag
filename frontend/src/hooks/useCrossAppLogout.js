import { useEffect } from 'react';
import { useMsal } from '@azure/msal-react';
import { checkGlobalLogoutFlag, markLogoutFlagProcessed } from '../utils/logout';

/**
 * Custom hook for cross-application logout coordination.
 * 
 * Monitors localStorage for logout signals from other applications/tabs.
 * When a logout is detected from another app, this hook triggers logout
 * in the current app.
 * 
 * How it works:
 * 1. Another app calls setGlobalLogoutFlags() when user logs out
 * 2. This hook polls localStorage every 5 seconds
 * 3. If logout flag is detected and not yet processed, trigger logout
 * 4. Mark the logout as processed to prevent duplicate logouts
 * 
 * Why we need this:
 * - MSAL instances are isolated per app/tab
 * - MSAL events don't cross tabs or different client IDs
 * - Provides true SSO experience (logout once, logout everywhere)
 * 
 * @example
 * ```javascript
 * function App() {
 *   useCrossAppLogout(); // That's it!
 *   
 *   return <YourAppContent />;
 * }
 * ```
 */
export function useCrossAppLogout() {
  const { instance, accounts } = useMsal();
  const isAuthenticated = accounts && accounts.length > 0;

  useEffect(() => {
    // Only run if user is authenticated
    if (!isAuthenticated) {
      return;
    }

    // Clear any stale logout flags on mount to prevent immediate logout
    // after a fresh login
    localStorage.removeItem('app_global_logout');
    localStorage.removeItem('app_global_logout_processed');

    /**
     * Check for global logout flag from other applications
     */
    const checkForCrossAppLogout = () => {
      const logoutTime = checkGlobalLogoutFlag();
      
      if (logoutTime !== null) {
        console.log('[useCrossAppLogout] Logout detected from another app, logging out...');
        
        // Mark as processed before logout to prevent race conditions
        markLogoutFlagProcessed(logoutTime);
        
        // Trigger logout in this app
        // Using logoutRedirect for reliability (popup might be blocked)
        instance.logoutRedirect({
          postLogoutRedirectUri: window.location.origin,
        });
      }
    };

    // Check immediately on mount
    checkForCrossAppLogout();

    // Then check every 5 seconds
    const intervalId = setInterval(checkForCrossAppLogout, 5000);

    // Cleanup interval on unmount
    return () => {
      clearInterval(intervalId);
    };
  }, [instance, isAuthenticated]);
}
