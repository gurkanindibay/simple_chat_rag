# Using Logout Utilities in Vanilla JavaScript

This guide shows how to use the logout utility functions from the React app in vanilla JavaScript contexts (like the SSO showcase app).

## Option 1: Copy the Utility Functions

Since the SSO showcase app doesn't use a build system, you can copy the core utility functions directly into your HTML file:

```html
<script>
  // Utility function to set global logout flags
  function setGlobalLogoutFlags() {
    const logoutMarker = Date.now().toString();
    
    console.log('[setGlobalLogoutFlags] Setting global logout flags...');
    
    try {
      // Set global logout flag in localStorage
      localStorage.setItem('msal_global_logout', logoutMarker);
      localStorage.setItem('msal_global_logout_processed', logoutMarker);
      
      // Also try to set cookie
      try {
        document.cookie = `msal_global_logout=${logoutMarker}; path=/; max-age=300`;
      } catch (cookieError) {
        console.warn('[setGlobalLogoutFlags] Unable to set cookie:', cookieError);
      }
      
      // Clear app-specific marker
      try { 
        localStorage.removeItem('app_logged_in'); 
      } catch (e) { 
        console.warn('[setGlobalLogoutFlags] Unable to clear app marker:', e);
      }
      
      return true;
    } catch (error) {
      console.error('[setGlobalLogoutFlags] Error setting flags:', error);
      return false;
    }
  }
  
  // Use it in your logout handler
  document.getElementById('logoutBtn').addEventListener('click', () => {
    // Set global flags first
    setGlobalLogoutFlags();
    
    // Then perform MSAL logout
    const account = msalInstance.getAllAccounts()[0];
    if (account) {
      msalInstance.logoutRedirect({
        account: account,
        postLogoutRedirectUri: 'http://localhost:8001'
      });
    }
  });
</script>
```

## Option 2: Create a Shared JavaScript File

Create a shared file that both apps can use:

### `/Users/gurkan_indibay/source/ai_tryouts/shared/logout-utils.js`

```javascript
// Shared logout utilities that work in any JavaScript context
window.LogoutUtils = {
  setGlobalLogoutFlags: function() {
    const logoutMarker = Date.now().toString();
    console.log('[LogoutUtils] Setting global logout flags...');
    
    try {
      localStorage.setItem('msal_global_logout', logoutMarker);
      localStorage.setItem('msal_global_logout_processed', logoutMarker);
      
      try {
        document.cookie = `msal_global_logout=${logoutMarker}; path=/; max-age=300`;
      } catch (cookieError) {
        console.warn('[LogoutUtils] Cookie error:', cookieError);
      }
      
      try { 
        localStorage.removeItem('app_logged_in'); 
      } catch (e) {}
      
      return true;
    } catch (error) {
      console.error('[LogoutUtils] Error:', error);
      return false;
    }
  },
  
  checkGlobalLogoutFlag: function(maxAgeMs = 300000) {
    const processedKey = 'msal_global_logout_processed';
    const lastProcessed = parseInt(localStorage.getItem(processedKey) || '0', 10);
    const now = Date.now();
    
    const logoutFlag = localStorage.getItem('msal_global_logout');
    if (logoutFlag) {
      const logoutTime = parseInt(logoutFlag, 10);
      
      if (!Number.isNaN(logoutTime) && 
          now - logoutTime < maxAgeMs && 
          logoutTime > lastProcessed) {
        return logoutTime;
      }
    }
    
    return null;
  },
  
  markLogoutFlagProcessed: function(logoutTime) {
    try {
      localStorage.setItem('msal_global_logout_processed', logoutTime.toString());
      localStorage.removeItem('msal_global_logout');
      document.cookie = 'msal_global_logout=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
    } catch (error) {
      console.error('[LogoutUtils] Error marking processed:', error);
    }
  }
};
```

Then include it in your HTML:

```html
<script src="../shared/logout-utils.js"></script>
<script>
  // Use it
  document.getElementById('logoutBtn').addEventListener('click', () => {
    LogoutUtils.setGlobalLogoutFlags();
    msalInstance.logoutRedirect({ postLogoutRedirectUri: 'http://localhost:8001' });
  });
  
  // Monitor for logout from other apps
  setInterval(() => {
    const logoutTime = LogoutUtils.checkGlobalLogoutFlag();
    if (logoutTime) {
      console.log('Logout detected from another app');
      // Update UI
      document.getElementById('statusDot').className = 'status-dot error';
      document.getElementById('statusText').textContent = 'Logged out - Please sign in';
      // Mark as processed
      LogoutUtils.markLogoutFlagProcessed(logoutTime);
    }
  }, 5000);
</script>
```

## Current Implementation in SSO Showcase

The SSO showcase app (`/sso-showcase-spa/index.html`) currently has the logout logic inline. Here's what it does:

1. **On Logout Button Click**: Sets the global logout flags before calling MSAL logout
2. **Periodic Check**: Every 5 seconds, checks for the global logout flag
3. **On Detection**: Updates UI and clears local MSAL storage

To use the reusable utilities, you would replace the inline code with calls to the utility functions.

## Benefits of Using Utilities

1. **Consistency**: Same logout behavior across all apps
2. **Maintainability**: Fix bugs in one place
3. **Testability**: Utility functions are easier to test
4. **Documentation**: Centralized documentation
5. **Reusability**: Can be used in any JavaScript context

## Migration Example

### Before (Current SSO Showcase)

```javascript
document.getElementById('logoutBtn').addEventListener('click', () => {
  const logoutMarker = Date.now().toString();
  localStorage.setItem('msal_global_logout', logoutMarker);
  localStorage.setItem('msal_global_logout_processed', logoutMarker);
  // ... more code ...
  msalInstance.logoutRedirect({ postLogoutRedirectUri: 'http://localhost:8001' });
});
```

### After (Using Utilities)

```javascript
document.getElementById('logoutBtn').addEventListener('click', () => {
  LogoutUtils.setGlobalLogoutFlags();
  msalInstance.logoutRedirect({ postLogoutRedirectUri: 'http://localhost:8001' });
});
```
