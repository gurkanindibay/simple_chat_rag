# CRITICAL: MSAL Not Storing Accounts in localStorage

## 🔴 Root Cause Identified

Based on your latest logs:
```
[MSAL Init] Accounts after redirect: Array(0)
[MSAL Init] No accounts found. app_logged_in flag: 1
[MSAL Init] MSAL localStorage keys: 2 keys found
```

**The problem:** MSAL is finding only 2 localStorage keys (likely just config), but NO account data. This means:
1. ✅ MSAL is initialized correctly
2. ✅ Config is stored in localStorage
3. ❌ **Account/token data is NOT being stored**

## 🎯 Latest Fixes Applied

### 1. Fixed ssoSilent Error
**Before:** `TypeError: Cannot read properties of undefined (reading 'prompt')`
**After:** Added proper scopes parameter to ssoSilent call

### 2. Enhanced Redirect Handling
Now properly captures and stores account from redirect result:
```javascript
if (redirectResult && redirectResult.account) {
  msalInstance.setActiveAccount(redirectResult.account);
  localStorage.setItem('app_logged_in', '1');
}
```

### 3. Added Detailed localStorage Debugging
Shows what's actually stored in those 2 localStorage keys

## 🧪 Diagnostic Steps

### Step 1: Use the Diagnostic Tool
I've created a diagnostic HTML file. Access it at:
```
http://localhost:5173/diagnostic.html
```

This tool will show you:
- Exactly what MSAL keys exist in localStorage
- Whether account data is present
- Token expiry information
- MSAL configuration

### Step 2: Check the Updated Logs

Reload your app and look for these NEW log messages:
```
[MSAL Init] Config: {clientId: "...", authority: "...", ...}
[MSAL Init] Keys: ["key1", "key2"]
[MSAL Init] key1: {...}
[MSAL Init] key2: {...}
```

This will tell us what those 2 keys actually contain.

### Step 3: Fresh Login Test

1. **Clear everything:**
   ```javascript
   localStorage.clear();
   location.reload();
   ```

2. **Login using REDIRECT method** (more reliable than popup):
   - Click "Sign in with Microsoft (Redirect)"
   - Complete authentication
   - You'll be redirected back to the app

3. **Check the logs immediately after redirect:**
   - Look for: `[MSAL Init] ✓ Login successful via redirect, account: ...`
   - Run diagnostic: http://localhost:5173/diagnostic.html

## 🔍 Possible Root Causes

### Cause 1: Browser Privacy Settings
**Symptom:** Only 2 config keys, no account data
**Check:** 
```javascript
// Run in console
localStorage.setItem('test', 'value');
console.log(localStorage.getItem('test'));
localStorage.removeItem('test');
```
If this fails, browser is blocking localStorage.

**Solution:** 
- Check browser privacy settings
- Disable "Block third-party cookies"
- Try in a different browser
- Try in Incognito/Private mode (but with cookies enabled)

### Cause 2: Azure App Registration Issue
**Symptom:** Login completes but no tokens stored
**Check:** 
1. Azure Portal → App registrations → Your app
2. Authentication → Implicit grant and hybrid flows
3. **Enable:** "Access tokens" and "ID tokens"

### Cause 3: CORS or Redirect URI Mismatch
**Symptom:** Login redirects but loses state
**Check:**
1. Azure Portal → App registrations → Your app → Authentication
2. Redirect URIs must include: `http://localhost:5173`
3. Logout URL: Leave empty or set to `http://localhost:5173`

### Cause 4: MSAL Cache Not Working
**Symptom:** Everything works but localStorage doesn't persist
**Check:** Configuration in authConfig.js
```javascript
cache: {
  cacheLocation: 'localStorage',  // ✓ Correct
  storeAuthStateInCookie: true,   // ✓ Correct
}
```

## 📊 What Should localStorage Look Like When Working?

After successful login, you should see keys like:
```
msal.account.keys
msal.token.keys.{clientId}
{clientId}.{tenantId}-login.windows.net-accesstoken-...
{clientId}.{tenantId}-login.windows.net-idtoken-...
{clientId}.{tenantId}-login.windows.net-refreshtoken-...
{homeAccountId}-login.windows.net-{tenantId}  // Account info
```

Typically 5-10+ keys for a successful login.

## 🎬 Next Steps

1. **Check new console logs** after reload (see what those 2 keys contain)
2. **Open diagnostic tool** at http://localhost:5173/diagnostic.html
3. **Clear and re-login** using REDIRECT method
4. **Share the output** of:
   - New console logs showing key contents
   - Diagnostic tool results
   - Browser you're using
   - Any browser extensions that might block cookies/localStorage

## 🚨 Quick Test Commands

Run these in browser console after login:

```javascript
// Check what's stored
Object.keys(localStorage).filter(k => k.includes('msal')).forEach(k => {
  console.log(k, ':', localStorage.getItem(k).substring(0, 100));
});

// Check MSAL state
console.log('Accounts:', window.__msal.getAllAccounts());
console.log('Active:', window.__msal.getActiveAccount());

// Check config
console.log('Config:', window.__msal.config);

// Manual token acquisition test
window.__msal.acquireTokenSilent({
  scopes: ['User.Read'],
  account: window.__msal.getAllAccounts()[0]
}).then(r => console.log('Token acquired:', r)).catch(e => console.error('Token failed:', e));
```

## 💡 Workaround: sessionStorage

If localStorage continues to fail, we can switch to sessionStorage (won't persist across tabs but will survive refresh):

```javascript
cache: {
  cacheLocation: 'sessionStorage',  // Alternative
  storeAuthStateInCookie: true,
}
```

This won't give you SSO across tabs, but it will survive page refresh.
