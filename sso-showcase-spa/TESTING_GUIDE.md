# ✅ SSO Configuration Complete - Testing Guide

## What Was Fixed

### Problem
You were seeing "Not authenticated - Please sign in" because:
1. **Main app** used `sessionStorage` (isolated per tab)
2. **SSO showcase** used `localStorage` (shared across tabs)
3. **Different storage** = No shared authentication ❌

### Solution Applied
✅ Both apps now use **`localStorage`** with the **same Client ID**:
- **Main App**: `localStorage` + Client ID `a8a16485-0827-46c6-b3e0-91fca5966341`
- **SSO Showcase**: `localStorage` + Client ID `a8a16485-0827-46c6-b3e0-91fca5966341`

## 🚨 Critical: Add Redirect URI in Azure

Before testing, you **MUST** add the redirect URI in Azure Portal:

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Microsoft Entra ID** → **App registrations**
3. Find app: **Client ID `a8a16485-0827-46c6-b3e0-91fca5966341`**
4. Click **Authentication** → Under **Single-page application** → **Add URI**
5. Add: `http://localhost:8001`
6. Click **Save**

**Without this step, you'll get "AADSTS50011: Redirect URI mismatch" error.**

## 🧪 How to Test SSO (After Azure Configuration)

### Preparation
1. **Close ALL browser tabs** for both apps
2. **Clear browser cache** (Cmd+Shift+Delete on Mac, Ctrl+Shift+Delete on Windows)
3. Make sure **frontend is running** on port 5173
4. Make sure **SSO showcase server is running** on port 8001

### Start Servers

Terminal 1 - Main App:
```bash
cd frontend
npm run dev
```

Terminal 2 - SSO Showcase:
```bash
cd sso-showcase-spa
python3 serve.py
```

### Test Scenario 1: Main → SSO Showcase ✅

1. **Open main app**: `http://localhost:5173`
2. **Sign in** with Microsoft
3. **See your chat interface** (you're logged in)
4. **Open new tab** in same browser
5. **Go to**: `http://localhost:8001`
6. **Expected**: ✅ Automatically shows your user info! No login prompt!

### Test Scenario 2: SSO Showcase → Main 🔄

1. **Close all tabs** and **clear cache** again
2. **Open SSO showcase first**: `http://localhost:8001`
3. **Click "Sign in with Microsoft"**
4. Complete login
5. **See your user info** displayed
6. **Open new tab** → `http://localhost:5173`
7. **Expected**: ✅ Main app automatically authenticated!

## ✅ Success Indicators

You know SSO is working when:
- ✅ No password prompt on second app
- ✅ User info appears immediately
- ✅ Same email/name shown in both apps
- ✅ Status shows "✅ SSO Successful! You were automatically authenticated."

## 🐛 Troubleshooting

### Still seeing "Not authenticated"?

**Quick Checklist**:
```bash
# 1. Verify main app uses localStorage
cd frontend
grep "cacheLocation:" src/authConfig.js
# Should show: cacheLocation: 'localStorage'

# 2. Verify SSO showcase uses localStorage
cd ../sso-showcase-spa
grep "cacheLocation:" index.html
# Should show: cacheLocation: 'localStorage'

# 3. Verify both use same client ID
grep "clientId:" index.html
# Should show: clientId: 'a8a16485-0827-46c6-b3e0-91fca5966341'
```

### Error: "AADSTS50011: Redirect URI mismatch"
**Cause**: You haven't added `http://localhost:8001` to Azure  
**Fix**: Follow the Azure configuration steps above

### Error: "interaction_required"
**Possible causes**:
1. **Browser cache not cleared** → Clear it completely
2. **Different browser used** → Use same browser for both apps
3. **Private/Incognito mode** → Don't use private mode (it isolates storage)
4. **localStorage disabled** → Check browser settings

**Fix**:
```bash
# Clear localStorage in browser console (F12):
localStorage.clear()
# Then refresh and try again
```

### Frontend not reflecting changes?
```bash
cd frontend
# Stop the dev server (Ctrl+C)
# Start it again
npm run dev
```

### Main app still uses sessionStorage?
If you edited `authConfig.js` but it's not taking effect:
1. Stop the frontend dev server
2. Clear browser cache
3. Start frontend again
4. Hard refresh (Cmd+Shift+R or Ctrl+Shift+R)

## 📊 Verification Commands

Run these to verify your configuration:

```bash
# Show all configuration
cd /Users/gurkan_indibay/source/ai_tryouts

echo "=== Configuration Summary ==="
echo "Main App Cache:"
grep "cacheLocation:" frontend/src/authConfig.js

echo -e "\nSSO Showcase Cache:"
grep "cacheLocation:" sso-showcase-spa/index.html

echo -e "\nMain App Client ID:"
grep "VITE_AZURE_FRONTEND_CLIENT_ID=" frontend/.env

echo -e "\nSSO Showcase Client ID:"
grep "clientId: 'a8a16485" sso-showcase-spa/index.html
```

Expected output:
```
=== Configuration Summary ===
Main App Cache:
    cacheLocation: 'localStorage',

SSO Showcase Cache:
                cacheLocation: 'localStorage',

Main App Client ID:
VITE_AZURE_FRONTEND_CLIENT_ID=a8a16485-0827-46c6-b3e0-91fca5966341

SSO Showcase Client ID:
                clientId: 'a8a16485-0827-46c6-b3e0-91fca5966341',
```

## 🎯 What Happens Behind the Scenes

When you sign into the **main app**:
1. MSAL.js gets tokens from Azure AD
2. Tokens stored in `localStorage` with key based on client ID
3. `localStorage` is shared across all tabs in same browser

When you open **SSO showcase**:
1. MSAL.js calls `ssoSilent()`
2. Checks `localStorage` for existing tokens
3. Finds tokens for client ID `a8a16485...`
4. Validates tokens are still valid
5. Uses tokens without asking for password ✅

## 📸 What You Should See

### Main App (After Login)
- Your chat interface
- Your name/email in top right
- "Sign Out" button visible

### SSO Showcase (With SSO Working)
```
🔐 SSO Showcase Application
Second SPA Demo
Testing Single Sign-On Between Applications

✅ Authenticated Successfully

✅ SSO Successful! You were automatically authenticated.

Display Name: Your Name
Email: your.email@domain.com
User ID: [user-id]
Environment: login.windows.net
```

## 🎉 Next Steps

Once SSO is working:

1. **Test both directions** (Main→SSO and SSO→Main)
2. **Test in different tabs** (open multiple tabs)
3. **Test sign out** (sign out in one, check if other also signs out)
4. **Document your findings** (screenshot the automatic authentication)

---

**Status**: ✅ Code changes complete, waiting for Azure redirect URI configuration  
**Next Action**: Add `http://localhost:8001` to Azure Portal  
**Expected Result**: SSO will work automatically after Azure configuration
