# Cross-App Logout Bug Fix

## Problem Description

When clicking "Sign in with Microsoft" on `localhost:8001` (SSO showcase app), the authentication state at `localhost:5173` (main app) was being destroyed. Users would see localStorage keys being deleted under `localhost:5173`.

## Root Cause

In `/sso-showcase-spa/index.html`, the login button handler (around line 648-690) was clearing **ALL MSAL-related localStorage keys**, not just interaction flags:

```javascript
// PROBLEMATIC CODE (REMOVED):
localKeys.forEach(k => {
    if (!k) return;
    if (k.includes('msal') || k.includes('login.windows')) {
        try { localStorage.removeItem(k); } catch (e) {}
    }
});
```

### Why This Was Destructive:

1. **localStorage Behavior**: On `localhost`, even though the apps run on different ports (5173 vs 8001), localStorage keys with certain patterns could affect each other
2. **Token Destruction**: The code was deleting actual authentication tokens, not just interaction flags
3. **Cross-App Impact**: This affected the main app at `localhost:5173` that was already logged in

## The Fix

Changed the login button handler to **ONLY clear sessionStorage interaction flags**, preserving all localStorage tokens:

```javascript
// FIXED CODE:
sessKeys.forEach(k => {
    if (!k) return;
    // Remove ONLY interaction status flags
    if (k.includes('msal.interaction.status')) {
        try { sessionStorage.removeItem(k); } catch (e) {}
    }
});

// DO NOT clear localStorage - it contains encrypted tokens
console.debug('[Login] Preserving localStorage tokens for session persistence');
```

### Key Changes:

1. ✅ **Only clear sessionStorage** - interaction flags live here
2. ✅ **Only clear `msal.interaction.status`** - not all MSAL keys
3. ✅ **Preserve localStorage completely** - tokens must persist
4. ✅ **Removed cookie clearing** - not needed and potentially harmful
5. ✅ **Added explanatory comments** - prevent future mistakes

## Testing

After applying this fix:

1. **Sign in to localhost:5173** (main app)
2. **Open localhost:8001** (SSO showcase)
3. **Click "Sign in with Microsoft"** on localhost:8001
4. **Verify**: localStorage keys in localhost:5173 should remain intact
5. **Expected**: User stays logged in on localhost:5173

## Why localStorage Clearing Was Wrong

### What Should Be Cleared:
- ❌ **sessionStorage interaction flags** - temporary state for auth flows
- ✅ Only `msal.interaction.status` keys

### What Should NEVER Be Cleared:
- ❌ **localStorage tokens** - contain encrypted authentication tokens
- ❌ **MSAL cache entries** - needed for SSO and refresh tokens
- ❌ **Account information** - required for seamless authentication

## Similar Issue in Main App

Note: The main app (`frontend/src/components/Login.jsx`) already had the correct implementation:

```javascript
// Only clear interaction status, not actual cache
if (k.includes('msal.interaction.status')) {
    sessionStorage.removeItem(k);
}
// DO NOT clear localStorage - it contains the encrypted tokens!
```

The SSO showcase app should follow this same pattern.

## Lessons Learned

1. **Be Surgical, Not Aggressive**: Only clear what's necessary (interaction flags)
2. **Understand Storage Scopes**: localStorage persists, sessionStorage is temporary
3. **Test Cross-App Impact**: Changes in one app can affect another on the same domain
4. **Preserve Tokens**: Never clear authentication tokens unless explicitly logging out
5. **Document Intent**: Comments prevent future developers from making the same mistake

## Files Modified

- `/Users/gurkan_indibay/source/ai_tryouts/sso-showcase-spa/index.html` (lines 648-680)

## Date Fixed

October 20, 2025
