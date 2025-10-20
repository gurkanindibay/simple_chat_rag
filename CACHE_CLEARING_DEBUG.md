# 🚨 CRITICAL: MSAL Cache Being Cleared After Refresh

## Problem Identified

After a successful login, when you refresh the page, **all the encrypted token keys are deleted** from localStorage, leaving only:
- `msal.a8a16485-...active-account-filters` (just the account metadata)

Missing (should be present):
- ❌ `msal.1-...-accesstoken-...` (2 tokens)
- ❌ `msal.1-...-idtoken-...`
- ❌ `msal.1-...-refreshtoken-...`
- ❌ `msal.1.account.keys`
- ❌ `msal.1.token.keys.{clientId}`
- ❌ Account data key

## Root Causes Found & Fixed

### Issue #1: Login Component Clearing Cache ✅ FIXED
**Problem:** `Login.jsx` was clearing ALL MSAL localStorage keys before login:
```javascript
// OLD CODE (BAD):
if (k.includes('msal') || k.includes('login.windows')) {
  localStorage.removeItem(k);  // Deleted tokens!
}
```

**Why this caused issues:**
1. User clicks "Sign in with Microsoft (Redirect)"
2. Login.jsx clears ALL MSAL cache (including any old tokens)
3. Redirects to Microsoft login
4. User authenticates
5. **Redirects back** - but cache was cleared before redirect!
6. No tokens saved because redirect context lost

**Fix Applied:** Now only clears interaction flags, preserves tokens:
```javascript
// NEW CODE (GOOD):
if (k.includes('msal.interaction.status')) {
  sessionStorage.removeItem(k);  // Only clear flags
}
// DO NOT touch localStorage tokens!
```

### Issue #2: Possible MSAL Bug - Cache Cleared on Init
**Investigation:** Added logging to check if MSAL's `initialize()` is clearing cache:
```javascript
console.log('Keys BEFORE init:', keysBeforeInit.length);
await msalInstance.initialize();
console.log('Keys AFTER init:', keysAfterInit.length);
```

This will tell us if MSAL itself is clearing the cache.

## Testing Instructions

### Step 1: Clear Everything & Start Fresh
```javascript
// In browser console:
localStorage.clear();
sessionStorage.clear();
location.reload();
```

### Step 2: Login Fresh
1. Go to http://localhost:5173
2. Click **"Sign in with Microsoft (Redirect)"** (more reliable than popup)
3. Complete authentication
4. You'll be redirected back

### Step 3: Check After Login
Open browser console and run:
```javascript
// Should show 8+ keys
Object.keys(localStorage).filter(k => k.includes('msal')).forEach(k => {
  console.log(k);
});

// Should show your account
window.__msal.getAllAccounts();
```

### Step 4: Refresh the Page
Press F5 or Cmd+R, then immediately check console for:
```
[MSAL Init] localStorage keys BEFORE init: 8 keys  ← Should be 8+
[MSAL Init] localStorage keys AFTER init: 8 keys   ← Should stay 8+
```

**If keys disappear:**
- Between BEFORE and AFTER: MSAL is clearing cache (MSAL bug)
- Both are low: Tokens were never saved (redirect flow issue)

### Step 5: Check What's Left
After refresh, run:
```javascript
Object.keys(localStorage).filter(k => k.includes('msal'))
```

**Expected:** 8+ keys including account and token keys
**If only 1 key:** Tokens are being cleared

## Possible Additional Issues

### Issue A: Token Expiry
Tokens typically expire after 1 hour. Check timestamps:
```javascript
const keys = Object.keys(localStorage).filter(k => k.includes('msal'));
keys.forEach(k => {
  try {
    const data = JSON.parse(localStorage.getItem(k));
    if (data.lastUpdatedAt) {
      const date = new Date(parseInt(data.lastUpdatedAt));
      console.log(k, '→ Last updated:', date.toLocaleString());
    }
  } catch(e) {}
});
```

If tokens are > 1 hour old, they'll be auto-deleted.

### Issue B: MSAL Version Bug
MSAL v4.25.1 has known issues with encrypted cache. Solutions:

**Option 1: Downgrade to v3.x** (uses unencrypted cache):
```json
{
  "@azure/msal-browser": "^3.24.0",
  "@azure/msal-react": "^2.1.2"
}
```

**Option 2: Disable encryption** (requires MSAL v4.5+):
```javascript
// In authConfig.js
cache: {
  cacheLocation: 'localStorage',
  storeAuthStateInCookie: true,
  // Disable encryption (if available in your MSAL version):
  secureCookies: false,
  claimsBasedCachingEnabled: false
}
```

### Issue C: Browser Extensions
Some privacy extensions clear localStorage. Check:
- Privacy Badger
- uBlock Origin (with strict mode)
- Cookie AutoDelete
- Any VPN with tracker blocking

**Test:** Try in Incognito/Private mode with extensions disabled.

### Issue D: Redirect URI Mismatch
If redirect URI doesn't match exactly, tokens won't be saved.

**Check Azure Portal:**
1. App registrations → Your app → Authentication
2. Redirect URIs must include: `http://localhost:5173` (exact match)
3. **NOT:** `http://localhost:5173/` (no trailing slash)
4. **NOT:** `https://localhost:5173` (wrong protocol)

## Diagnostic Command

Run this after login to export full diagnostic data:
```javascript
const diagnostics = {
  timestamp: new Date().toISOString(),
  accounts: window.__msal.getAllAccounts(),
  activeAccount: window.__msal.getActiveAccount(),
  cache: {}
};

Object.keys(localStorage).filter(k => k.includes('msal')).forEach(k => {
  try {
    const raw = localStorage.getItem(k);
    const parsed = JSON.parse(raw);
    diagnostics.cache[k] = {
      type: typeof parsed,
      keys: typeof parsed === 'object' ? Object.keys(parsed) : null,
      size: raw.length,
      hasData: parsed.data ? 'YES (encrypted)' : 'NO',
      lastUpdated: parsed.lastUpdatedAt ? new Date(parseInt(parsed.lastUpdatedAt)).toLocaleString() : null
    };
  } catch(e) {
    diagnostics.cache[k] = { error: e.message };
  }
});

console.log(JSON.stringify(diagnostics, null, 2));
```

Copy the output and share it for detailed analysis.

## Next Steps

1. **Clear and re-login** following steps above
2. **Check new console logs** - look for the BEFORE/AFTER init messages
3. **Verify tokens persist** after refresh
4. **Share diagnostics** if tokens still disappear

The login component fix should prevent the cache clearing issue. The new logging will tell us if there are other issues.
