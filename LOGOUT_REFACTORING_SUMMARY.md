# Logout Refactoring Summary

## Overview
Refactored the logout functionality across the application to be reusable, maintainable, and consistent. Created both React hooks and vanilla JavaScript utilities for cross-app logout coordination.

## Changes Made

### 1. Created Reusable React Hook
**File**: `/frontend/src/hooks/useLogout.js`

- Custom hook that handles logout for React components
- Supports both popup and redirect logout types
- Sets global logout flags for cross-app coordination
- Provides loading state (`isLoggingOut`)
- Properly handles errors and cleanup

**Usage**:
```jsx
const { logout, isLoggingOut } = useLogout({ logoutType: 'popup' });
```

### 2. Created Utility Functions
**File**: `/frontend/src/utils/logout.js`

Standalone utility functions for any JavaScript context:
- `setGlobalLogoutFlags()` - Sets logout markers
- `checkGlobalLogoutFlag()` - Checks for logout events
- `markLogoutFlagProcessed()` - Marks logout as handled
- `clearMSALStorage()` - Clears all MSAL data

**Usage**:
```javascript
import { setGlobalLogoutFlags } from '../utils/logout';
setGlobalLogoutFlags();
```

### 3. Updated Components

#### ChatHeader.jsx
**Before**: 25 lines of logout logic inline
**After**: 5 lines using `useLogout` hook

```jsx
// Before
const handleLogout = async () => {
  if (isLoggingOut) return;
  setIsLoggingOut(true);
  try {
    const logoutMarker = Date.now().toString();
    localStorage.setItem('msal_global_logout', logoutMarker);
    // ... 15+ more lines
  } catch (error) { ... }
};

// After
const { logout, isLoggingOut } = useLogout({ logoutType: 'popup' });
const handleLogout = async () => {
  try { await logout(); } 
  catch (error) { console.error('Logout error:', error); }
};
```

#### Sidebar.jsx
**Before**: 20 lines of logout logic inline
**After**: 5 lines using `useLogout` hook

- Simplified from 20+ lines to 5 lines
- Added loading state to button (`isLoggingOut`)
- Added spinner icon during logout
- Consistent with ChatHeader implementation

### 4. Created Documentation

#### `/frontend/src/hooks/LOGOUT_README.md`
Comprehensive documentation including:
- API reference for hook and utilities
- Usage examples for React and vanilla JS
- How the cross-app coordination works
- Best practices and troubleshooting
- Migration guide from old code

#### `/sso-showcase-spa/LOGOUT_UTILS_USAGE.md`
Guide for using logout utilities in vanilla JavaScript:
- How to copy utility functions
- Creating shared JavaScript files
- Integration examples for SSO showcase app
- Migration examples

## Benefits

### Code Quality
- **Reduced Duplication**: Logout logic defined once, used everywhere
- **Cleaner Components**: Components went from 20+ lines to 5 lines
- **Better Separation**: Business logic separated from UI components
- **Type Safety**: Well-documented parameters and return values

### Maintainability
- **Single Source of Truth**: Fix bugs in one place
- **Consistent Behavior**: Same logout flow across all components
- **Easy Testing**: Utility functions are easier to unit test
- **Clear Documentation**: Comprehensive guides for developers

### User Experience
- **Loading States**: All logout buttons show loading spinner
- **Disabled State**: Prevents multiple logout clicks
- **Error Handling**: Proper error catching and logging
- **Cross-App Sync**: Logout in one app logs out all apps

### Developer Experience
- **Simple API**: Easy to understand and use
- **Flexible Options**: Support for popup and redirect
- **Good Defaults**: Works out of the box with sensible defaults
- **Extensive Examples**: Multiple usage examples provided

## File Structure

```
frontend/
├── src/
│   ├── hooks/
│   │   ├── useLogout.js          # React hook for logout
│   │   └── LOGOUT_README.md      # Comprehensive documentation
│   ├── utils/
│   │   └── logout.js             # Standalone utility functions
│   └── components/
│       ├── ChatHeader.jsx        # Updated to use useLogout
│       └── Sidebar.jsx           # Updated to use useLogout

sso-showcase-spa/
└── LOGOUT_UTILS_USAGE.md         # Guide for vanilla JS usage
```

## Cross-App Logout Flow

### When User Clicks Logout (App A)
1. `useLogout` hook called
2. `setGlobalLogoutFlags()` sets:
   - `localStorage.msal_global_logout` = timestamp
   - `localStorage.msal_global_logout_processed` = timestamp
   - `document.cookie.msal_global_logout` = timestamp
3. MSAL logout performed (popup or redirect)

### Detection in Other App (App B)
1. Periodic check (every 5 seconds) calls `checkGlobalLogoutFlag()`
2. Finds recent logout timestamp
3. Updates UI to logged-out state
4. Clears MSAL storage using `clearMSALStorage()`
5. Marks as processed using `markLogoutFlagProcessed()`

## Testing Checklist

- [ ] Logout from ChatHeader in main app
  - [ ] Logs out main app
  - [ ] Logs out SSO showcase app
- [ ] Logout from Sidebar in main app
  - [ ] Logs out main app
  - [ ] Logs out SSO showcase app
- [ ] Logout from SSO showcase app
  - [ ] Logs out SSO showcase app
  - [ ] Logs out main app
- [ ] Loading states work correctly
- [ ] Cannot click logout multiple times
- [ ] Error handling works
- [ ] Console logs are clean and informative

## Future Enhancements

### Potential Improvements
1. **Event-based communication**: Use `BroadcastChannel` API for instant cross-tab communication
2. **Service Worker**: Centralized logout coordination
3. **Analytics**: Track logout events and reasons
4. **Graceful degradation**: Handle localStorage blocked scenarios
5. **Logout reasons**: Pass reason codes (user action, timeout, error, etc.)

### Additional Features
1. **Confirmation dialog**: "Are you sure you want to log out?"
2. **Logout all devices**: Server-side session invalidation
3. **Remember me**: Option to stay logged in
4. **Auto-logout**: After period of inactivity

## Migration Instructions

### For Existing Components

1. **Import the hook**:
   ```jsx
   import { useLogout } from '../hooks/useLogout';
   ```

2. **Use the hook**:
   ```jsx
   const { logout, isLoggingOut } = useLogout({ 
     logoutType: 'popup'  // or 'redirect'
   });
   ```

3. **Update button**:
   ```jsx
   <button onClick={logout} disabled={isLoggingOut}>
     {isLoggingOut ? 'Signing out...' : 'Sign out'}
   </button>
   ```

4. **Remove old code**:
   - Remove manual localStorage manipulation
   - Remove manual MSAL instance usage
   - Remove manual loading state management

### For Vanilla JavaScript

See `/sso-showcase-spa/LOGOUT_UTILS_USAGE.md` for detailed instructions.

## Related Files

- Fixed issue: ChatHeader was not setting global logout flags
- Standardized: Both logout buttons now use same underlying logic
- Documented: Comprehensive README files for all use cases

## Questions?

Refer to:
- `/frontend/src/hooks/LOGOUT_README.md` - Complete API documentation
- `/sso-showcase-spa/LOGOUT_UTILS_USAGE.md` - Vanilla JS usage guide
- Component source code - Real-world usage examples
