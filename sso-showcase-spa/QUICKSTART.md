# SSO Showcase - Quick Start Guide

## 🎯 Goal
Demonstrate **true SSO** between two separate Single Page Applications (SPAs):
1. **Main App** (React, port 5173) - Full RAG chat application
2. **SSO Showcase** (Vanilla JS, port 8001) - Simple demo app

## 📋 Prerequisites Checklist

Before starting, you need:
- [ ] Access to Azure Portal with app registration permissions
- [ ] Main application already configured with Azure AD (Client ID: `a8a16485-0827-46c6-b3e0-91fca5966341`)
- [ ] Python 3.x installed
- [ ] Tenant ID: `066690f2-a8a6-4889-852e-124371dcbd6f`

## 🚀 Step-by-Step Setup

### Step 1: Register Second SPA in Azure Portal

1. Open [Azure Portal](https://portal.azure.com)
2. Navigate to **Microsoft Entra ID** → **App registrations**
3. Click **+ New registration**
4. Fill in details:
   ```
   Name: SSO Showcase SPA
   Supported account types: Single tenant
   Redirect URI: 
     - Platform: Single-page application
     - URI: http://localhost:8001
   ```
5. Click **Register**
6. **IMPORTANT**: Copy the **Application (client) ID** from the Overview page
   - Example: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

### Step 2: Configure the SSO Showcase SPA

```bash
cd sso-showcase-spa

# Configure with your new Client ID
python3 configure.py YOUR-CLIENT-ID-HERE

# Verify configuration
python3 configure.py --show
```

### Step 3: Start the SSO Showcase Server

```bash
# From the sso-showcase-spa directory
python3 serve.py
```

You should see:
```
🚀 SSO Showcase SPA Server running at http://localhost:8001
```

### Step 4: Test True SSO Behavior

#### 🧪 Test Scenario A: Main App → SSO Showcase

1. **Sign in to main app:**
   - Open `http://localhost:5173`
   - Click "Sign in with Microsoft"
   - Complete authentication

2. **Open SSO Showcase:**
   - Open `http://localhost:8001` in the **same browser**
   - **Expected**: Automatically authenticated! No login prompt!
   - **You should see**: Your user info displayed immediately

#### 🧪 Test Scenario B: SSO Showcase → Main App

1. **Clear browser cache** (to reset)
   
2. **Sign in to SSO Showcase first:**
   - Open `http://localhost:8001`
   - Click "Sign in with Microsoft"
   - Complete authentication

3. **Open Main App:**
   - Open `http://localhost:5173`
   - **Expected**: Automatically authenticated! No login prompt!

## ✅ Success Criteria

You've successfully configured SSO when:
- ✅ Signing into one app automatically authenticates the other
- ✅ No password prompt when switching between apps
- ✅ User info appears automatically on the second app
- ✅ Both apps show the same user email/name

## 🔍 Verification

### Check Current Configuration

```bash
cd sso-showcase-spa
python3 configure.py --show
```

Expected output:
```
📋 Current SSO Showcase SPA Configuration:
============================================================
✅ Client ID: <your-new-client-id>
✅ Redirect URI: http://localhost:8001
✅ Authority: https://login.microsoftonline.com/066690f2-a8a6-4889-852e-124371dcbd6f
============================================================
```

### Test Server is Running

```bash
curl http://localhost:8001
```

Should return HTML content.

## 🐛 Troubleshooting

### Problem: "SSO_SHOWCASE_CLIENT_ID" appears in error
**Solution**: You forgot to configure the Client ID
```bash
python3 configure.py YOUR-CLIENT-ID
```

### Problem: "AADSTS50011: Redirect URI mismatch"
**Solution**: Add redirect URI in Azure Portal
1. Go to your app registration
2. Authentication → Add URI → `http://localhost:8001`
3. Save

### Problem: "interaction_required" error
**Solution**: No active session exists
1. Sign in to the main app first
2. Then open the SSO showcase
3. Or click "Sign in with Microsoft" on the SSO page

### Problem: SSO doesn't work (always asks for login)
**Possible causes:**
1. Different browsers used for each app
2. Private/Incognito mode (doesn't share localStorage)
3. Browser cleared cache between attempts
4. Different Client IDs configured

**Solution**: 
- Use same browser, same window/tab session
- Check both apps use `localStorage` (not `sessionStorage`)
- Verify both use same Tenant ID

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│      Microsoft Entra ID (Azure AD)      │
│         Tenant: 066690f2-...            │
└──────────────┬─────────────┬────────────┘
               │             │
               │             │
    ┌──────────▼─────┐  ┌───▼──────────┐
    │   Main App     │  │ SSO Showcase │
    │  Port: 5173    │  │ Port: 8001   │
    │  Client ID: a8 │  │ Client ID:   │
    │  (React+Vite)  │  │ NEW (Vanilla)│
    └────────────────┘  └──────────────┘
           │                    │
           └─────────┬──────────┘
                     │
              localStorage
         (Shared SSO session)
```

## 📚 Files Overview

```
sso-showcase-spa/
├── index.html       - Complete SPA (HTML/CSS/JS + MSAL)
├── serve.py         - Simple HTTP server (port 8001)
├── configure.py     - Configuration helper script
├── README.md        - Detailed documentation
└── QUICKSTART.md    - This file
```

## 🎓 What You're Testing

1. **Cross-Application SSO**: Two different apps, two different client IDs
2. **Shared Session**: Both apps recognize the same Microsoft session
3. **Silent Authentication**: No password prompt on second app
4. **localStorage Persistence**: Session survives page refreshes
5. **ssoSilent() Method**: MSAL checks for existing session

## 🔐 Security Notes

- Each app has its own Client ID (security isolation)
- Both apps must be in the same Azure AD tenant
- Redirect URIs must be pre-registered (prevents phishing)
- Tokens are stored securely in localStorage
- Users can revoke access in Azure portal at any time

## 📞 Next Steps After Testing

Once SSO is working locally:

1. **Document the flow** - Screenshot the automatic authentication
2. **Test edge cases** - Try different browsers, sign-out scenarios
3. **Production setup** - Register production redirect URIs
4. **Security review** - Ensure token handling meets your requirements

---

**Need Help?** Check `README.md` for detailed troubleshooting or review the SSO flow diagrams in the main docs.
