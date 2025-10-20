/**
 * Utility function to set global logout flags.
 * This is useful for coordinating logout across multiple applications.
 * 
 * Call this BEFORE performing MSAL logout to ensure all apps are notified.
 */
export function setGlobalLogoutFlags() {
  const logoutMarker = Date.now().toString();
  
  console.log('[setGlobalLogoutFlags] Setting global logout flags...');
  
  try {
    // Set global logout flag in localStorage (works across same-browser apps)
    localStorage.setItem('msal_global_logout', logoutMarker);
    localStorage.setItem('msal_global_logout_processed', logoutMarker);
    
    // Also try to set cookie for potential cross-origin coordination
    try {
      document.cookie = `msal_global_logout=${logoutMarker}; path=/; max-age=300`;
    } catch (cookieError) {
      console.warn('[setGlobalLogoutFlags] Unable to set msal_global_logout cookie:', cookieError);
    }
    
    // Clear app-specific logged-in marker
    try { 
      localStorage.removeItem('app_logged_in'); 
    } catch (e) { 
      console.warn('[setGlobalLogoutFlags] Unable to clear app_logged_in marker:', e);
    }
    
    console.log('[setGlobalLogoutFlags] Global logout flags set successfully');
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
 * @param {number} maxAgeMs - Maximum age of the logout flag in milliseconds (default: 5 minutes)
 * @returns {number|null} - Logout timestamp if valid, null otherwise
 */
export function checkGlobalLogoutFlag(maxAgeMs = 300000) {
  const processedKey = 'msal_global_logout_processed';
  const processedValue = localStorage.getItem(processedKey);
  const lastProcessed = processedValue ? parseInt(processedValue, 10) : 0;
  const now = Date.now();

  // Check localStorage for global logout flag
  const logoutFlag = localStorage.getItem('msal_global_logout');
  if (logoutFlag) {
    const logoutTime = parseInt(logoutFlag, 10);
    
    // Check if it's recent and not yet processed
    if (!Number.isNaN(logoutTime) && 
        now - logoutTime < maxAgeMs && 
        logoutTime > lastProcessed) {
      return logoutTime;
    } else if (!Number.isNaN(logoutTime) && now - logoutTime >= maxAgeMs) {
      // Flag is old, remove it
      localStorage.removeItem('msal_global_logout');
    }
  }
  
  // Also check cookie as fallback
  try {
    const cookies = document.cookie.split(';');
    const logoutCookie = cookies.find(cookie => cookie.trim().startsWith('msal_global_logout='));
    if (logoutCookie) {
      const cookieValue = parseInt(logoutCookie.split('=')[1], 10);
      if (!Number.isNaN(cookieValue) && 
          cookieValue > lastProcessed && 
          now - cookieValue < maxAgeMs) {
        return cookieValue;
      }
    }
  } catch (cookieError) {
    console.warn('[checkGlobalLogoutFlag] Cookie check failed:', cookieError);
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
    localStorage.setItem('msal_global_logout_processed', logoutTime.toString());
    
    // Clear the flag
    localStorage.removeItem('msal_global_logout');
    
    // Expire the cookie
    document.cookie = 'msal_global_logout=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
    
    console.log('[markLogoutFlagProcessed] Logout flag marked as processed:', logoutTime);
  } catch (error) {
    console.error('[markLogoutFlagProcessed] Error marking flag as processed:', error);
  }
}

/**
 * Clear all MSAL-related storage (localStorage, sessionStorage, and cookies).
 * Use with caution as this will remove all authentication state.
 */
export function clearMSALStorage() {
  console.log('[clearMSALStorage] Clearing MSAL storage...');
  
  try {
    // Clear MSAL keys from localStorage
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && (key.includes('msal') || key.includes('login.windows')) && 
          key !== 'msal_global_logout' && 
          key !== 'msal_global_logout_processed') {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach(key => {
      localStorage.removeItem(key);
      console.log('[clearMSALStorage] Removed localStorage key:', key);
    });
    
    // Clear ALL MSAL keys from sessionStorage (including interaction status)
    const sessionKeysToRemove = [];
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key && (key.includes('msal') || key.includes('login.windows'))) {
        sessionKeysToRemove.push(key);
      }
    }
    sessionKeysToRemove.forEach(key => {
      sessionStorage.removeItem(key);
      console.log('[clearMSALStorage] Removed sessionStorage key:', key);
    });
    
    // Clear MSAL-related cookies
    document.cookie.split(';').forEach(cookie => {
      const cookieName = cookie.split('=')[0].trim();
      if ((cookieName.includes('msal') || cookieName.includes('login.windows')) && 
          cookieName !== 'msal_global_logout') {
        document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
        console.log('[clearMSALStorage] Cleared cookie:', cookieName);
      }
    });
    
    console.log('[clearMSALStorage] MSAL storage cleared successfully');
  } catch (error) {
    console.error('[clearMSALStorage] Error clearing MSAL storage:', error);
  }
}
