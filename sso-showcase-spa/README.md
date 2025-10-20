# SSO Showcase - Second SPA

This is a standalone Single Page Application (SPA) designed to demonstrate **true Single Sign-On (SSO)** behavior between two different applications.

## Purpose

This application showcases that when a user signs into one application (the main RAG chat app at `localhost:5173`), they are automatically authenticated in this second application **without needing to sign in again**.

## Architecture

- **Technology**: Pure HTML/CSS/JavaScript with MSAL.js
- **Port**: 8001
- **Client ID**: Separate Azure AD app registration (different from main app)
- **SSO Mechanism**: Shared Microsoft session via `localStorage` and `ssoSilent()` method

## Setup Instructions

### 1. Register the Application in Azure Portal

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Microsoft Entra ID** > **App registrations**
3. Click **New registration**
4. Configure:
   - **Name**: "SSO Showcase SPA" (or similar)
   - **Supported account types**: "Accounts in this organizational directory only"
   - **Redirect URI**: 
     - Platform: **Single-page application (SPA)**
     - URI: `http://localhost:8001`
5. Click **Register**
6. Copy the **Application (client) ID** - you'll need this next

### 2. Configure the Client ID

Edit `sso-showcase-spa/index.html` and replace the placeholder:

```javascript
clientId: 'SSO_SHOWCASE_CLIENT_ID'
```

With your actual Client ID:

```javascript
clientId: 'YOUR-ACTUAL-CLIENT-ID-HERE'
```

### 3. Start the Server

From the `sso-showcase-spa` directory:

```bash
python3 serve.py
```

The server will start on `http://localhost:8001`

### 4. Test SSO Flow

#### Option A: Existing Session (True SSO)
1. **First**, sign in to the main application at `http://localhost:5173`
2. **Then**, open `http://localhost:8001` in the same browser
3. **Expected Result**: You should be automatically authenticated without seeing a login prompt!

#### Option B: Fresh Session
1. Open `http://localhost:8001` directly (without signing into main app first)
2. Click "Sign in with Microsoft"
3. Complete authentication
4. **Then**, open the main app at `http://localhost:5173`
5. **Expected Result**: The main app should also automatically authenticate you!

## How SSO Works

1. **Shared Microsoft Session**: Both apps use the same Microsoft Entra ID tenant
2. **localStorage**: Both apps store tokens in `localStorage` (not `sessionStorage`)
3. **ssoSilent()**: The MSAL library checks for an existing Microsoft session
4. **No Password Prompt**: If a session exists, you get a token silently without re-authenticating

## Key Differences from Main App

| Aspect | Main App (localhost:5173) | SSO Showcase (localhost:8001) |
|--------|---------------------------|-------------------------------|
| Technology | React + Vite | Pure HTML/CSS/JS |
| Port | 5173 | 8001 |
| Client ID | `a8a16485-0827-46c6-b3e0-91fca5966341` | *New Client ID* (to be registered) |
| Purpose | Full RAG chat application | SSO demonstration only |

## Troubleshooting

### "interaction_required" Error
- This means no active Microsoft session exists
- Sign in to the main app first, then try again
- Or click "Sign in with Microsoft" on this page

### "Redirect URI Mismatch" Error
- Ensure `http://localhost:8001` is added as a redirect URI in Azure Portal
- Check that the Client ID in `index.html` matches your Azure app registration

### SSO Not Working
1. Verify both apps use `localStorage` (not `sessionStorage`)
2. Ensure both apps use the same tenant ID
3. Clear browser cache and try again
4. Check browser console for errors

## Files

- `index.html` - The complete SPA (HTML/CSS/JS)
- `serve.py` - Simple Python HTTP server
- `README.md` - This file

## Next Steps

After testing locally:
1. Deploy both SPAs to production
2. Add production redirect URIs to Azure app registrations
3. Update `redirectUri` in both apps to use production URLs
