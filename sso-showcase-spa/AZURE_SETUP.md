# Azure Configuration for SSO Showcase

## Current Setup Status

✅ **Client ID**: `a8a16485-0827-46c6-b3e0-91fca5966341` (same as main app)  
✅ **Tenant ID**: `066690f2-a8a6-4889-852e-124371dcbd6f`  
⚠️ **Redirect URI**: `http://localhost:8001` - **NEEDS TO BE ADDED IN AZURE**

## Why Same Client ID?

We're using the **same client ID** for both apps because:

1. **Immediate SSO**: Authentication in one app automatically works in the other
2. **Shared Token Cache**: Both apps access the same tokens in localStorage
3. **No Additional Configuration**: No need to configure API permissions between apps
4. **Real-world Scenario**: This is how companies deploy the same app on multiple domains

## Required Azure Configuration

### Step 1: Add Redirect URI

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Microsoft Entra ID** → **App registrations**
3. Find the app with Client ID: `a8a16485-0827-46c6-b3e0-91fca5966341`
4. Click **Authentication** in the left menu
5. Under **Single-page application**, click **Add URI**
6. Add: `http://localhost:8001`
7. Click **Save**

### Step 2: Verify Existing Configuration

Make sure these redirect URIs are also configured:
- ✅ `http://localhost:5173` (main app)
- ✅ `http://localhost:8001` (SSO showcase) ← **Add this one**

## Testing SSO Flow

Once the redirect URI is added:

### Test 1: Main App → SSO Showcase
1. **Clear browser cache** (important!)
2. Open `http://localhost:5173`
3. Sign in with Microsoft
4. Open `http://localhost:8001` in a **new tab** (same browser)
5. **Expected**: Automatically authenticated! ✅

### Test 2: SSO Showcase → Main App
1. **Clear browser cache**
2. Open `http://localhost:8001`
3. Sign in with Microsoft
4. Open `http://localhost:5173` in a **new tab**
5. **Expected**: Automatically authenticated! ✅

## How It Works

```
┌─────────────────────────────────────────┐
│    Microsoft Entra ID (Azure AD)        │
│    Tenant: 066690f2-...                 │
│    Client ID: a8a16485-... (shared)     │
└──────────────┬──────────────────────────┘
               │
               │ Same tokens stored in
               │ localStorage (shared)
               │
    ┌──────────▼─────┐       ┌─────▼──────────┐
    │   Main App     │       │  SSO Showcase  │
    │  Port: 5173    │◄─────►│  Port: 8001    │
    │  (React+Vite)  │  SSO  │  (Vanilla JS)  │
    └────────────────┘       └────────────────┘
         Both use localStorage with same clientId
```

## Troubleshooting

### "AADSTS50011: Redirect URI mismatch"
**Cause**: `http://localhost:8001` not added to Azure  
**Fix**: Follow Step 1 above

### "interaction_required" Error
**Cause**: No tokens in localStorage  
**Fix**: 
1. Clear browser cache
2. Close ALL tabs for both apps
3. Sign in to main app first
4. Then open SSO showcase

### SSO Still Not Working?
**Checklist**:
- [ ] Both apps use `localStorage` (not `sessionStorage`)
- [ ] Both apps use same client ID: `a8a16485-0827-46c6-b3e0-91fca5966341`
- [ ] Both redirect URIs added in Azure
- [ ] Using same browser (not different browsers)
- [ ] Not in private/incognito mode
- [ ] Browser cache cleared before testing

### Verify Configuration

Check main app:
```bash
cd frontend
grep "cacheLocation" src/authConfig.js
# Should show: cacheLocation: 'localStorage'
```

Check SSO showcase:
```bash
cd sso-showcase-spa
grep "cacheLocation" index.html
# Should show: cacheLocation: 'localStorage'
```

## Production Considerations

When deploying to production:

1. **Add Production Redirect URIs**:
   ```
   https://yourdomain.com
   https://yourdomain.com/sso-showcase
   ```

2. **Update index.html**:
   ```javascript
   redirectUri: 'https://yourdomain.com/sso-showcase'
   ```

3. **HTTPS Required**: Azure requires HTTPS in production

4. **Same Origin**: For best SSO experience, host both apps on same domain:
   - Main: `https://yourdomain.com`
   - SSO: `https://yourdomain.com/sso`

## Alternative: Different Client IDs

If you want to test with **different client IDs** (true cross-app SSO):

1. Create second app registration in Azure
2. Configure API permissions (App1 can access App2)
3. Grant admin consent
4. Both apps must use `localStorage`
5. Test with `ssoSilent()` - may require user consent

**Note**: This is more complex and requires additional Azure configuration. The same-client-ID approach demonstrates SSO just as effectively.
