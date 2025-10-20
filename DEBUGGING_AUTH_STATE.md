# Debugging Authentication State Persistence

## Current Issue Analysis

Based on the logs you provided, MSAL is not finding any accounts in localStorage after refresh:

```
[MSAL Init] Accounts after redirect: []
[MSAL Init] No accounts found. app_logged_in flag: null
```

This indicates one of two things:
1. **You haven't logged in yet** after the code changes, OR
2. **MSAL is not properly storing accounts** in localStorage

## Recent Changes Made

### 1. Added Active Account Setting (`main.jsx`)
```javascript
// After finding accounts, explicitly set the active account
if (accounts && accounts.length > 0) {
  msalInstance.setActiveAccount(accounts[0]);
  console.log('[MSAL Init] Active account set to:', accounts[0].username);
  localStorage.setItem('app_logged_in', '1');
}
```

### 2. Enhanced Login Success Handler (`Login.jsx`)
```javascript
// After successful popup login, set the active account
instance.loginPopup(loginRequest)
  .then((response) => {
    if (response && response.account) {
      instance.setActiveAccount(response.account);
      console.log('[Login] Active account set to:', response.account.username);
    }
    localStorage.setItem('app_logged_in', '1');
  })
```

### 3. Added Detailed Debug Logging (`main.jsx`)
Now logs:
- MSAL configuration details
- Number of MSAL keys in localStorage
- Sample localStorage key names

## Testing Steps

### Step 1: Clear Everything and Start Fresh
Open your browser console (F12) and run:
```javascript
// Clear all MSAL data
Object.keys(localStorage).forEach(key => {
  if (key.includes('msal') || key.includes('login.windows')) {
    localStorage.removeItem(key);
  }
});
localStorage.removeItem('app_logged_in');
localStorage.clear();
location.reload();
```

### Step 2: Perform Fresh Login
1. Go to http://localhost:5173
2. Click "Sign in with Microsoft (Popup)" or "Sign in with Microsoft (Redirect)"
3. Complete the authentication flow
4. Watch the console logs - you should see:
   ```
   [Login] Popup login successful: {account: {...}, ...}
   [Login] Active account set to: your-email@domain.com
   ```

### Step 3: Check localStorage
In the browser console, run:
```javascript
// Check if MSAL stored anything
Object.keys(localStorage).filter(key => key.includes('msal')).length
// Should return a number > 0

// Check accounts
window.__msal.getAllAccounts()
// Should return array with your account

// Check active account
window.__msal.getActiveAccount()
// Should return your account object
```

### Step 4: Test Refresh
1. **Refresh the page (F5 or Cmd+R)**
2. Watch the console logs - you should now see:
   ```
   [MSAL Init] Accounts after redirect: [{...}]
   [MSAL Init] ✓ Found 1 account(s), setting app_logged_in flag
   [MSAL Init] Active account set to: your-email@domain.com
   [MSAL Init] MSAL localStorage keys: X keys found
   ```
3. The app should load **without** asking you to login again

## What to Look For in Logs

### ✅ Good Signs (Authentication Working)
```
[MSAL Init] Config: {clientId: "a8a16485-...", authority: "https://login.microsoftonline.com/...", ...}
[Login] Popup login successful: {...}
[Login] Active account set to: user@domain.com
[MSAL Init] Accounts after redirect: [{username: "user@domain.com", ...}]
[MSAL Init] Active account set to: user@domain.com
[MSAL Init] MSAL localStorage keys: 5 keys found
[App] Render - isAuthenticatedHook: true accounts: 1 isAuthenticated: true
```

### ❌ Bad Signs (Still Broken)
```
[MSAL Init] Config: {clientId: "", ...}  // Empty clientId!
[MSAL Init] Accounts after redirect: []
[MSAL Init] MSAL localStorage keys: 0 keys found
[App] Render - isAuthenticatedHook: false accounts: 0 isAuthenticated: false
```

## Common Issues and Solutions

### Issue 1: Empty Client ID
**Symptom:** `clientId: ""`
**Cause:** Environment variables not loaded
**Solution:** 
1. Check that `/Users/gurkan_indibay/source/ai_tryouts/frontend/.env` exists
2. Restart the Vite dev server: Stop (Ctrl+C) and run `npm run dev` again
3. Clear cache: `rm -rf node_modules/.vite`

### Issue 2: No MSAL Keys in localStorage After Login
**Symptom:** `MSAL localStorage keys: 0 keys found` even after login
**Causes:**
- Browser privacy settings blocking localStorage
- Incorrect redirect URI configuration
- MSAL cache location misconfigured

**Solution:**
1. Check browser console for errors during login
2. Verify redirect URI in Azure Portal matches `http://localhost:5173`
3. Try in a different browser or incognito mode

### Issue 3: Accounts Found but Lost After Refresh
**Symptom:** Login works, but refresh loses state
**Cause:** Active account not set, or MSAL not reading from cache properly
**Solution:** Already fixed by adding `setActiveAccount()` calls

## Manual Testing in Console

Open browser console and test MSAL directly:

```javascript
// Check MSAL instance
window.__msal

// Get all accounts
window.__msal.getAllAccounts()

// Get active account
window.__msal.getActiveAccount()

// Set active account manually (if you have accounts but no active one)
const accounts = window.__msal.getAllAccounts();
if (accounts.length > 0) {
  window.__msal.setActiveAccount(accounts[0]);
}

// Check localStorage
Object.keys(localStorage).filter(k => k.includes('msal'))

// Try silent token acquisition
window.__msal.acquireTokenSilent({
  scopes: ['User.Read'],
  account: window.__msal.getActiveAccount()
}).then(r => console.log('Token:', r))
```

## Next Steps

1. **Clear browser data** and perform a fresh login
2. **Watch the console logs** carefully during login and refresh
3. **Run the console tests** above to verify MSAL state
4. If still not working, share the new console logs showing:
   - What happens during login (look for `[Login]` messages)
   - What's in localStorage after login
   - What happens on refresh
