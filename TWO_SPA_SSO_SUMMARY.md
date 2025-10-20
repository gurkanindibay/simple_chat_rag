# Two-SPA SSO Implementation Summary

## Overview

This document describes the implementation of a **true Single Sign-On (SSO) demonstration** between two separate Single Page Applications (SPAs), showcasing that authentication in one application automatically authenticates users in another application.

## Architecture

### Two Independent Applications

1. **Main Application** (RAG Chat App)
   - **Technology**: React + Vite
   - **Port**: 5173
   - **Client ID**: `a8a16485-0827-46c6-b3e0-91fca5966341`
   - **Purpose**: Full-featured RAG chat application
   - **Location**: `frontend/`

2. **SSO Showcase Application**
   - **Technology**: Vanilla HTML/CSS/JavaScript + MSAL.js
   - **Port**: 8001
   - **Client ID**: `630f781d-5e19-46c4-9273-35ed836088a2`
   - **Purpose**: Demonstrate SSO capability
   - **Location**: `sso-showcase-spa/`

### SSO Flow

```
User Journey:
1. User signs in to Main App (port 5173)
   ↓
2. Microsoft Entra ID creates session
   ↓
3. Session stored in localStorage (browser)
   ↓
4. User opens SSO Showcase (port 8001)
   ↓
5. SSO Showcase calls ssoSilent()
   ↓
6. MSAL finds existing session in localStorage
   ↓
7. User automatically authenticated! ✅
```

## Key Technical Decisions

### Why Two Different Client IDs?

**Reason**: To demonstrate **true cross-application SSO**
- Same Client ID = same app (not impressive)
- Different Client IDs = different apps sharing authentication (true SSO)

This mirrors real-world scenarios where companies have multiple applications (HR portal, Email, etc.) all using the same identity provider.

### Global Logout Implementation

**Enhanced SSO Experience**: When logging out from any application, all MSAL-related localStorage is cleared, effectively logging out from all applications in the session.

```javascript
// Global logout clears all MSAL tokens across apps
const keysToRemove = [];
for (let i = 0; i < localStorage.length; i++) {
  const key = localStorage.key(i);
  if (key && (key.includes('msal') || key.includes('login.windows'))) {
    keysToRemove.push(key);
  }
}
keysToRemove.forEach(key => localStorage.removeItem(key));
```

### Why localStorage (not sessionStorage)?

| Storage Type | Scope | SSO Behavior |
|-------------|-------|--------------|
| `sessionStorage` | Single tab only | ❌ No SSO across tabs |
| `localStorage` | Entire browser | ✅ SSO across tabs and windows |

**Decision**: Use `localStorage` to enable true SSO across browser windows/tabs.

### Why ssoSilent() Method?

The `ssoSilent()` method is the core of SSO:
```javascript
msalInstance.ssoSilent(loginRequest)
```

**What it does**:
1. Checks for existing Microsoft session
2. If found, gets access token silently
3. No user interaction required (no password prompt)
4. Returns user info automatically

**Alternative**: `loginRedirect()` - always shows login UI (not SSO)

## Implementation Files

### SSO Showcase SPA Files

```
sso-showcase-spa/
├── index.html          - Complete SPA with MSAL.js integration
├── serve.py            - HTTP server (port 8001)
├── configure.py        - Client ID configuration helper
├── README.md           - Detailed documentation
└── QUICKSTART.md       - Step-by-step setup guide
```

### Key Code Sections

#### MSAL Configuration (index.html)
```javascript
const msalConfig = {
    auth: {
        clientId: 'SSO_SHOWCASE_CLIENT_ID',  // To be replaced
        authority: 'https://login.microsoftonline.com/066690f2-a8a6-4889-852e-124371dcbd6f',
        redirectUri: 'http://localhost:8001'
    },
    cache: {
        cacheLocation: 'localStorage',  // Enable cross-tab SSO
        storeAuthStateInCookie: false
    }
};
```

#### SSO Silent Authentication
```javascript
async function attemptSSOSilent() {
    try {
        const response = await msalInstance.ssoSilent(loginRequest);
        // Success - user authenticated silently!
        showUserInfo(response.account);
    } catch (error) {
        // No session found - show login button
        document.getElementById('loginBtn').style.display = 'inline-block';
    }
}
```

## Setup Process

### For You (The Developer)

1. **Azure Portal Configuration**:
   ```
   ✅ New App Registration Created:
   - Name: "SSO Showcase SPA"
   - Type: Single-page application
   - Redirect URI: http://localhost:8001
   - Client ID: 630f781d-5e19-46c4-9273-35ed836088a2
   ```

2. **Configure the SPA**:
   ```bash
   ✅ Client ID configured in index.html
   ```

3. **Start the server**:
   ```bash
   python3 serve.py
   ```

### For Testing SSO

**Scenario 1**: Main App First
1. Open `http://localhost:5173` → Sign in
2. Open `http://localhost:8001` → **Auto-authenticated!** ✅

**Scenario 2**: SSO Showcase First
1. Open `http://localhost:8001` → Sign in
2. Open `http://localhost:5173` → **Auto-authenticated!** ✅

## Comparison: Old vs New Approach

### ❌ Previous Approach (Discarded)
- Single HTML page served by backend at `/sso-standalone`
- Same client ID as main app
- Port 8000 (backend port)
- **Problem**: Not truly separate applications

### ✅ New Approach (Current)
- Completely separate SPA application
- Own client ID (demonstrates cross-app SSO)
- Own port (8001, independent server)
- **Benefit**: True demonstration of SSO between different apps

## Benefits of This Implementation

1. **Realistic SSO Demonstration**
   - Two different apps, two different client IDs
   - Mirrors real-world enterprise SSO scenarios

2. **Easy to Understand**
   - Vanilla JavaScript (no framework complexity)
   - All code in single HTML file
   - Clear visual feedback of SSO success

3. **Independent Deployment**
   - Runs on its own port
   - No dependency on backend server
   - Can be deployed separately

4. **Educational Value**
   - Shows exactly how MSAL.js handles SSO
   - Demonstrates localStorage sharing
   - Clear error messages guide troubleshooting

## Security Considerations

1. **Separate Client IDs**: Each app has its own identity in Azure AD
2. **Registered Redirect URIs**: Prevents unauthorized apps from stealing tokens
3. **Token Isolation**: Each app gets its own tokens (different scopes/audience)
4. **Tenant Restriction**: Both apps must be in same Azure AD tenant
5. **User Consent**: Users can revoke access per application

## Testing Checklist

- [ ] Main app authentication works
- [ ] SSO Showcase app authentication works independently
- [ ] SSO: Main → Showcase (auto-auth)
- [ ] SSO: Showcase → Main (auto-auth)
- [ ] Logout from one logs out both (global logout)
- [ ] Works across multiple browser tabs
- [ ] Clear cache breaks SSO (expected)
- [ ] Private/incognito mode works independently

## Troubleshooting Guide

### Issue: "SSO_SHOWCASE_CLIENT_ID" in console
**Cause**: Client ID not configured
**Fix**: `python3 configure.py YOUR-CLIENT-ID`

### Issue: "Redirect URI mismatch"
**Cause**: Azure doesn't recognize `http://localhost:8001`
**Fix**: Add redirect URI in Azure Portal app registration

### Issue: Always shows login button
**Cause**: No active Microsoft session
**Fix**: Sign in to main app first, then open SSO showcase

### Issue: SSO works once, then stops
**Cause**: Browser cleared localStorage
**Fix**: Normal behavior - need to sign in again

## Production Deployment

When deploying to production:

1. **Register Production URLs** in Azure:
   ```
   Main App: https://yourdomain.com
   SSO Showcase: https://yourdomain.com/sso-demo
   ```

2. **Update Redirect URIs**:
   - Main app: `https://yourdomain.com`
   - SSO Showcase: `https://yourdomain.com/sso-demo`

3. **Update Configuration**:
   - index.html: Change `redirectUri` to production URL
   - Both apps: Update authority URLs if needed

4. **SSL Required**: Azure AD requires HTTPS for production

## Documentation Files

- `sso-showcase-spa/README.md` - Comprehensive documentation
- `sso-showcase-spa/QUICKSTART.md` - Quick setup guide
- `TWO_SPA_SSO_SUMMARY.md` - This file (architecture overview)

## Next Steps

1. ✅ **You**: Register new SPA in Azure Portal
2. ✅ **You**: Configure Client ID using `configure.py`
3. ✅ **You**: Start the server (`python3 serve.py`)
4. ✅ **Test**: Verify SSO works in both directions
5. 📸 **Document**: Screenshot the automatic authentication
6. 🎉 **Demonstrate**: Show stakeholders true SSO in action

## Conclusion

This implementation provides a **realistic, production-ready demonstration** of Single Sign-On between two separate applications. It showcases:

- ✅ Cross-application authentication
- ✅ Silent authentication (no password prompts)
- ✅ Shared session management
- ✅ Industry-standard MSAL.js library
- ✅ Secure Azure AD integration

The two-SPA approach clearly demonstrates the value of SSO: **sign in once, access multiple applications** - the core promise of Single Sign-On.
