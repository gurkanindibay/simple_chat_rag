// MSAL Cache Helper - manually restore active account from encrypted cache
// This is a workaround for MSAL v4+ encrypted cache not auto-loading accounts

export const restoreActiveAccountFromCache = (msalInstance) => {
  try {
    const clientId = msalInstance.config.auth.clientId;
    
    // Check for active account filters
    const activeAccountKey = `msal.${clientId}.active-account-filters`;
    const activeAccountFilters = localStorage.getItem(activeAccountKey);
    
    if (!activeAccountFilters) {
      console.log('[Cache Helper] No active account filters found');
      return false;
    }
    
    const filters = JSON.parse(activeAccountFilters);
    console.log('[Cache Helper] Active account filters:', filters);
    
    // Try to get all accounts
    const accounts = msalInstance.getAllAccounts();
    
    if (accounts.length > 0) {
      // Find the account matching the filters
      const matchingAccount = accounts.find(acc => 
        acc.homeAccountId === filters.homeAccountId ||
        acc.localAccountId === filters.localAccountId
      );
      
      if (matchingAccount) {
        msalInstance.setActiveAccount(matchingAccount);
        console.log('[Cache Helper] ✓ Restored active account:', matchingAccount.username);
        return true;
      }
    }
    
    console.log('[Cache Helper] Could not find matching account');
    return false;
    
  } catch (e) {
    console.error('[Cache Helper] Error restoring account:', e);
    return false;
  }
};
