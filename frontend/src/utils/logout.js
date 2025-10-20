/**
 * Utility function to set global logout flags.
 * This is useful for coordinating logout across multiple applications.
 * 
 * NOTE: This is a CUSTOM feature for cross-app logout coordination.
 * MSAL does not provide cross-application logout coordination out of the box.
 * 
 * Call this BEFORE performing MSAL logout to notify other apps.
 * 
 * How it works:
 * 1. App A calls setGlobalLogoutFlags() → sets 'app_global_logout' with timestamp
 * 2. App B periodically checks 'app_global_logout' 
 * 3. If timestamp > 'app_global_logout_processed', App B knows it needs to logout
 * 4. App B logs out and calls markLogoutFlagProcessed() to mark it as handled
 */
export function setGlobalLogoutFlags() {
  const logoutMarker = Date.now().toString();
  
  console.log('[setGlobalLogoutFlags] Setting global logout flag for cross-app coordination...');
  
  try {
    // Set global logout flag in localStorage (works across same-browser apps)
    // Other apps will detect this and logout if they haven't processed this timestamp yet
    localStorage.setItem('app_global_logout', logoutMarker);
    
    // DO NOT set 'app_global_logout_processed' here!
    // Each app will mark it as processed after they handle the logout.
    
    // Clear app-specific logged-in marker
    try { 
      localStorage.removeItem('app_logged_in'); 
    } catch (e) { 
      console.warn('[setGlobalLogoutFlags] Unable to clear app_logged_in marker:', e);
    }
    
    console.log('[setGlobalLogoutFlags] Global logout flag set to:', logoutMarker);
    return true;
  } catch (error) {
    console.error('[setGlobalLogoutFlags] Error setting logout flags:', error);
    return false;
  }
}

/**
 * Check if a global logout flag has been set recently.
 * Returns the logout timestamp if found and not yet processed, null otherwise.
 * 
 * NOTE: This is for cross-app logout coordination only.
 * 
 * @param {number} maxAgeMs - Maximum age of the logout flag in milliseconds (default: 5 minutes)
 * @returns {number|null} - Logout timestamp if valid, null otherwise
 */
export function checkGlobalLogoutFlag(maxAgeMs = 300000) {
  const processedKey = 'app_global_logout_processed';
  const processedValue = localStorage.getItem(processedKey);
  const lastProcessed = processedValue ? parseInt(processedValue, 10) : 0;
  const now = Date.now();

  // Check localStorage for global logout flag
  const logoutFlag = localStorage.getItem('app_global_logout');
  if (logoutFlag) {
    const logoutTime = parseInt(logoutFlag, 10);
    
    // Check if it's recent and not yet processed
    if (!Number.isNaN(logoutTime) && 
        now - logoutTime < maxAgeMs && 
        logoutTime > lastProcessed) {
      return logoutTime;
    } else if (!Number.isNaN(logoutTime) && now - logoutTime >= maxAgeMs) {
      // Flag is old, remove it
      localStorage.removeItem('app_global_logout');
    }
  }
  
  return null;
}

/**
 * Mark a global logout flag as processed.
 * 
 * @param {number} logoutTime - The logout timestamp to mark as processed
 */
export function markLogoutFlagProcessed(logoutTime) {
  try {
    localStorage.setItem('app_global_logout_processed', logoutTime.toString());
    
    // Clear the flag
    localStorage.removeItem('app_global_logout');
    
    console.log('[markLogoutFlagProcessed] Logout flag marked as processed:', logoutTime);
  } catch (error) {
    console.error('[markLogoutFlagProcessed] Error marking flag as processed:', error);
  }
}

// NOTE: clearMSALStorage() function has been REMOVED.
// 
// According to Microsoft's official documentation:
// https://learn.microsoft.com/en-us/entra/msal/javascript/browser/logout
// 
// "The logout process for MSAL takes two steps:
//  1. Clear the MSAL cache.
//  2. Clear the session on the identity server."
// 
// Both steps are handled automatically by logoutRedirect() and logoutPopup().
// Manual cache clearing is NOT necessary and violates the principle of
// letting the library manage its own state.
// 
// If you need to clear cache without server logout, use:
//   instance.logoutRedirect({ onRedirectNavigate: () => false })
