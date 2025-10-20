# Cross-App "interaction_in_progress" Error Fix

## Problem
After signing out from localhost:5173, attempting to login from localhost:8001 resulted in an "interaction_in_progress" error:
```
Not authenticated - Please sign in
ℹ️ Another Microsoft authentication flow is already running. Complete or close it, then try again.
```

The workaround was to open localhost:5173 and perform a login operation there first.

## Root Cause
The issue occurred because:

1. **sessionStorage is NOT shared between different origins** (localhost:5173 vs localhost:8001)
2. **localStorage IS shared** across the same domain (localhost)
3. When logging out from 5173:
   - Global logout flags were set in localStorage (visible to 8001) ✓
   - localStorage tokens were cleared ✓
   - But sessionStorage interaction states were NOT cleared ✗
4. When trying to login from 8001:
   - It detected the logout flag via localStorage
   - It cleared localStorage tokens
   - But **stale MSAL sessionStorage entries remained** in 8001's own sessionStorage
   - These stale entries included `msal.interaction.status` and other MSAL state
   - MSAL saw these stale entries and thought an interaction was already in progress

## Solution
The fix ensures that **ALL MSAL-related sessionStorage is cleared** at multiple points:

### 1. During Logout (both apps)
**Files Changed:**
- `frontend/src/hooks/useLogout.js`
- `frontend/src/utils/logout.js`
- `sso-showcase-spa/index.html`

**Changes:**
- Added `clearMSALStorage()` call in `useLogout` hook before performing MSAL logout
- Updated `clearMSALStorage()` to clear ALL sessionStorage entries (not just localStorage)
- Both localStorage AND sessionStorage are now cleared during logout

### 2. When Detecting Global Logout (localhost:8001)
**File Changed:**
- `sso-showcase-spa/index.html`

**Changes:**
- When the periodic check detects a global logout flag
- When `attemptSSOSilent()` detects a global logout flag
- Now clears ALL MSAL sessionStorage entries in addition to localStorage

### 3. Before Login Attempt (both apps)
**Files Changed:**
- `frontend/src/components/Login.jsx`
- `sso-showcase-spa/index.html`

**Changes:**
- Changed from clearing only `msal.interaction.status` 
- To clearing ALL MSAL-related sessionStorage entries (`msal` or `login.windows`)
- This ensures no stale state can cause "interaction_in_progress" errors

## Key Changes

### frontend/src/utils/logout.js
```javascript
export function clearMSALStorage() {
  // ... existing localStorage clearing ...
  
  // NEW: Clear ALL MSAL keys from sessionStorage (including interaction status)
  const sessionKeysToRemove = [];
  for (let i = 0; i < sessionStorage.length; i++) {
    const key = sessionStorage.key(i);
    if (key && (key.includes('msal') || key.includes('login.windows'))) {
      sessionKeysToRemove.push(key);
    }
  }
  sessionKeysToRemove.forEach(key => sessionStorage.removeItem(key));
}
```

### frontend/src/hooks/useLogout.js
```javascript
const logout = async () => {
  // Set global logout flags for cross-app coordination
  setGlobalLogoutFlags();
  
  // NEW: Clear all MSAL storage including sessionStorage
  clearMSALStorage();
  
  // Perform MSAL logout
  // ...
};
```

### frontend/src/components/Login.jsx & sso-showcase-spa/index.html
```javascript
// Before: Only cleared msal.interaction.status
if (k.includes('msal.interaction.status')) { /* clear */ }

// After: Clear ALL MSAL sessionStorage
if (k.includes('msal') || k.includes('login.windows')) { /* clear */ }
```

## Testing the Fix

1. **Start both apps:**
   ```bash
   # Terminal 1 - Main app
   cd frontend
   npm run dev  # localhost:5173
   
   # Terminal 2 - SSO showcase
   cd sso-showcase-spa
   python serve.py  # localhost:8001
   ```

2. **Test the scenario:**
   - Login to localhost:5173
   - Logout from localhost:5173
   - Go to localhost:8001
   - Click "Sign in with Microsoft"
   - **Expected:** Login should work without "interaction_in_progress" error

3. **Verify in browser console:**
   - Check for "Cleared sessionStorage key" messages
   - Should see multiple MSAL sessionStorage keys being cleared
   - No "interaction_in_progress" errors should appear

## Why This Works

### Before Fix:
```
Logout from 5173:
├── localStorage cleared ✓
├── sessionStorage NOT cleared ✗
└── localStorage flags set ✓

Login to 8001:
├── Detects logout flag ✓
├── Clears localStorage ✓
├── sessionStorage STILL has stale entries ✗
└── MSAL error: "interaction_in_progress" ❌
```

### After Fix:
```
Logout from 5173:
├── localStorage cleared ✓
├── sessionStorage cleared ✓ (NEW)
└── localStorage flags set ✓

Login to 8001:
├── Detects logout flag ✓
├── Clears localStorage ✓
├── Clears sessionStorage ✓ (NEW)
└── Login succeeds ✓
```

## Important Notes

1. **localStorage tokens are preserved during login** - We only clear sessionStorage, not localStorage, when preparing for login. This preserves valid tokens.

2. **sessionStorage is cleared during logout** - During logout we clear EVERYTHING including sessionStorage to ensure clean state.

3. **Works across both apps** - The fix is applied to both localhost:5173 (React app) and localhost:8001 (SSO showcase) to ensure consistent behavior.

4. **Backwards compatible** - The fix doesn't break existing functionality; it only adds more aggressive cleanup of stale state.

## Related Files
- `frontend/src/hooks/useLogout.js` - React hook for logout
- `frontend/src/utils/logout.js` - Shared logout utilities
- `frontend/src/components/Login.jsx` - React login component
- `sso-showcase-spa/index.html` - Vanilla JS SSO showcase app
