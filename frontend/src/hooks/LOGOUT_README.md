# Logout Utilities

This directory contains reusable logout utilities for coordinating logout across multiple applications that share the same authentication context.

## Files

### `useLogout.js` - React Hook
A custom React hook that provides logout functionality with cross-app coordination.

#### Usage

```jsx
import { useLogout } from '../hooks/useLogout';

function MyComponent() {
  const { logout, isLoggingOut } = useLogout({ 
    logoutType: 'popup',  // or 'redirect'
    postLogoutRedirectUri: '/' 
  });

  const handleSignOut = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  return (
    <button onClick={handleSignOut} disabled={isLoggingOut}>
      {isLoggingOut ? 'Signing out...' : 'Sign Out'}
    </button>
  );
}
```

#### Options

- `logoutType`: `'popup'` or `'redirect'` (default: `'popup'`)
- `postLogoutRedirectUri`: URI to redirect to after logout (default: `'/'`)

#### Returns

- `logout`: Async function to perform logout
- `isLoggingOut`: Boolean indicating if logout is in progress

### `logout.js` - Utility Functions
Standalone utility functions for managing logout flags and storage. Useful for vanilla JavaScript contexts or advanced scenarios.

#### Functions

##### `setGlobalLogoutFlags()`
Sets global logout flags in localStorage and cookies to notify other apps.

```javascript
import { setGlobalLogoutFlags } from '../utils/logout';

// Before performing MSAL logout
setGlobalLogoutFlags();
msalInstance.logoutRedirect();
```

##### `checkGlobalLogoutFlag(maxAgeMs = 300000)`
Checks if a global logout flag has been set recently.

```javascript
import { checkGlobalLogoutFlag } from '../utils/logout';

const logoutTime = checkGlobalLogoutFlag();
if (logoutTime) {
  console.log('Global logout detected at:', new Date(logoutTime));
  // Perform local logout
}
```

##### `markLogoutFlagProcessed(logoutTime)`
Marks a logout flag as processed to prevent duplicate handling.

```javascript
import { markLogoutFlagProcessed } from '../utils/logout';

const logoutTime = checkGlobalLogoutFlag();
if (logoutTime) {
  // Handle logout
  markLogoutFlagProcessed(logoutTime);
}
```

##### `clearMSALStorage()`
Clears all MSAL-related localStorage and cookies.

```javascript
import { clearMSALStorage } from '../utils/logout';

// Use with caution - removes all auth state
clearMSALStorage();
```

## How It Works

### Cross-App Logout Coordination

1. When a user logs out from one app, it sets a timestamp in:
   - `localStorage.msal_global_logout`
   - `localStorage.msal_global_logout_processed`
   - `document.cookie.msal_global_logout` (for cross-origin attempts)

2. Other apps periodically check for this flag using `checkGlobalLogoutFlag()`

3. If a recent logout flag is found, the app:
   - Logs out locally
   - Marks the flag as processed using `markLogoutFlagProcessed()`
   - Clears MSAL storage using `clearMSALStorage()`

### Storage Keys Used

- `msal_global_logout` - Timestamp of logout event
- `msal_global_logout_processed` - Timestamp of last processed logout
- `app_logged_in` - App-specific login marker

## Examples

### Example 1: React Component with Popup Logout

```jsx
import { useLogout } from '../hooks/useLogout';

function Header() {
  const { logout, isLoggingOut } = useLogout({ logoutType: 'popup' });

  return (
    <button onClick={logout} disabled={isLoggingOut}>
      <i className={`fas ${isLoggingOut ? 'fa-spinner fa-spin' : 'fa-sign-out-alt'}`}></i>
      {isLoggingOut ? 'Signing out...' : 'Sign out'}
    </button>
  );
}
```

### Example 2: React Component with Redirect Logout

```jsx
import { useLogout } from '../hooks/useLogout';

function Sidebar() {
  const { logout, isLoggingOut } = useLogout({ 
    logoutType: 'redirect',
    postLogoutRedirectUri: '/logged-out'
  });

  return (
    <button onClick={logout} disabled={isLoggingOut}>
      Sign Out
    </button>
  );
}
```

### Example 3: Vanilla JavaScript

```javascript
import { setGlobalLogoutFlags } from '../utils/logout';

// In your logout handler
document.getElementById('logoutBtn').addEventListener('click', () => {
  // Set global flags first
  setGlobalLogoutFlags();
  
  // Then perform MSAL logout
  msalInstance.logoutRedirect({
    postLogoutRedirectUri: 'http://localhost:8001'
  });
});
```

### Example 4: Monitoring for Logout in Another App

```javascript
import { checkGlobalLogoutFlag, markLogoutFlagProcessed, clearMSALStorage } from '../utils/logout';

// Check periodically (e.g., every 5 seconds)
setInterval(() => {
  const logoutTime = checkGlobalLogoutFlag();
  
  if (logoutTime) {
    console.log('Logout detected from another app');
    
    // Clear local auth state
    clearMSALStorage();
    
    // Update UI
    showLoggedOutState();
    
    // Mark as processed
    markLogoutFlagProcessed(logoutTime);
  }
}, 5000);
```

## Best Practices

1. **Use the hook in React components**: Always prefer `useLogout` hook over manual implementation
2. **Set flags before MSAL logout**: Always call `setGlobalLogoutFlags()` before MSAL logout operations
3. **Handle errors**: Wrap logout calls in try-catch blocks
4. **Disable UI during logout**: Use `isLoggingOut` to disable buttons and show loading states
5. **Choose appropriate logout type**: 
   - Use `'popup'` for better UX (no page reload)
   - Use `'redirect'` for guaranteed cleanup
6. **Monitor for global logout**: Check flags periodically in long-running apps
7. **Clean up processed flags**: Always call `markLogoutFlagProcessed()` after handling a logout

## Migration Guide

### Before (Old Code)

```jsx
const handleLogout = async () => {
  try {
    const logoutMarker = Date.now().toString();
    localStorage.setItem('msal_global_logout', logoutMarker);
    localStorage.setItem('msal_global_logout_processed', logoutMarker);
    document.cookie = `msal_global_logout=${logoutMarker}; path=/; max-age=300`;
    localStorage.removeItem('app_logged_in');
    await instance.logoutPopup({ postLogoutRedirectUri: '/' });
  } catch (error) {
    console.error('Logout error:', error);
  }
};
```

### After (New Code)

```jsx
import { useLogout } from '../hooks/useLogout';

const { logout, isLoggingOut } = useLogout({ logoutType: 'popup' });

const handleLogout = async () => {
  try {
    await logout();
  } catch (error) {
    console.error('Logout error:', error);
  }
};
```

## Troubleshooting

### Logout not propagating to other apps

1. Verify both apps are checking for the logout flag
2. Check console logs for `[setGlobalLogoutFlags]` messages
3. Ensure localStorage is accessible (not blocked by browser)
4. Verify both apps are on the same origin (for localStorage sharing)

### Multiple logout triggers

1. Check if `markLogoutFlagProcessed()` is being called
2. Verify the flag expiration time (default: 5 minutes)
3. Look for duplicate event listeners or periodic checks

### Logout fails silently

1. Check browser console for error messages
2. Verify MSAL instance is properly initialized
3. Check network tab for failed Microsoft logout requests
