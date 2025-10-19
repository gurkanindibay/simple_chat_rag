# Microsoft Entra (Azure AD) Authentication Setup

This document provides a complete guide to setting up Microsoft Entra (formerly Azure Active Directory) authentication for the RAG Chat application.

## Table of Contents
1. [Overview](#overview)
2. [Azure Portal Setup](#azure-portal-setup)
3. [Backend Configuration](#backend-configuration)
4. [Frontend Configuration](#frontend-configuration)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

## Overview

The application uses Microsoft Entra for authentication with the following architecture:
- **Frontend**: Uses MSAL (Microsoft Authentication Library) for browser-based authentication
- **Backend**: Validates JWT tokens issued by Microsoft Entra
- **Flow**: OAuth 2.0 Authorization Code Flow with PKCE

## Azure Portal Setup

### Step 1: Create an App Registration

1. Navigate to the [Azure Portal](https://portal.azure.com)
2. Go to **Azure Active Directory** → **App registrations** → **New registration**
3. Fill in the following:
   - **Name**: RAG Chat Application (or your preferred name)
   - **Supported account types**: Choose based on your needs:
     - Single tenant (only your organization)
     - Multi-tenant (any organization)
     - Personal Microsoft accounts (for testing)
   - **Redirect URI**: 
     - Platform: Single-page application (SPA)
     - URI: `http://localhost:5173` (for development)
4. Click **Register**

### Step 2: Configure Authentication

1. In your app registration, go to **Authentication**
2. Under **Single-page application**, add redirect URIs:
   - `http://localhost:5173` (development)
   - Your production URL (e.g., `https://yourapp.com`)
# Microsoft Entra (Azure AD) Authentication — Setup & Verification

This document shows the exact steps to register the frontend (SPA) and backend (API) applications in Microsoft Entra (Azure AD), configure environment variables used by this project, and verify the end-to-end authentication flow locally.

This repository expects the following env var conventions:

- Frontend (Vite) envs (file: `frontend/.env`):
   - `VITE_AZURE_FRONTEND_TENANT_ID` — Directory (tenant) ID
   - `VITE_AZURE_FRONTEND_CLIENT_ID` — Frontend (SPA) Application (client) ID
   - `VITE_REDIRECT_URI` — SPA redirect URI (development default: `http://localhost:5173`)
   - `VITE_AZURE_BACKEND_CLIENT_ID` — Backend (API) Application (client) ID (used to build scope)
   - `VITE_AZURE_BACKEND_SCOPE` — Full scope URI (optional). If not set, the code builds `api://<VITE_AZURE_BACKEND_CLIENT_ID>/access_as_user`
   - `VITE_API_URL` — Backend base URL (e.g. `http://localhost:8000`)

- Backend envs (file: `.env` at repository root or process env):
   - `AZURE_TENANT_ID` — Directory (tenant) ID (must match the frontend tenant)
   - `AZURE_CLIENT_ID` — Backend (API) Application (client) ID
   - (optional) `AZURE_CLIENT_SECRET` — only needed for confidential flows on the backend

Everything below assumes a dev setup on `localhost` (frontend on :5173, backend on :8000).

---

## 1) Register Backend (API) app in Azure

1. In Azure Portal → Azure Active Directory → App registrations → New registration
    - Name: `rag-chat-backend` (or your name)
    - Supported account types: choose as needed (usually Single tenant for internal use)
    - Redirect URI: *leave blank* for API app (not required)
    - Click Register

2. Record values from the Overview page:
    - Directory (tenant) ID → use for `AZURE_TENANT_ID`
    - Application (client) ID → use for `AZURE_CLIENT_ID` and `VITE_AZURE_BACKEND_CLIENT_ID`

3. Expose an API (create a delegated scope):
    - In the backend app registration → **Expose an API**
    - If Application ID URI is empty, set it to `api://<APPLICATION_CLIENT_ID>` (Azure will suggest one)
    - Click **Add a scope**
       - Scope name: `access_as_user`
       - Who can consent: `Admins and users`
       - Admin consent display name: `Access RAG Chat API`
       - Admin consent description: `Allow the app to access the RAG Chat API on behalf of the signed-in user.`
       - Click **Add scope**
    - After this, your full scope URI will be `api://<BACKEND_CLIENT_ID>/access_as_user`

4. (Optional) App roles: if you want RBAC, add `App roles` in the manifest or via the App roles UI and assign users/groups in Enterprise Applications.

---

## 2) Register Frontend (SPA) app in Azure

1. Azure AD → App registrations → New registration
    - Name: `rag-chat-frontend` (or your name)
    - Supported account types: same tenant as backend (recommended)
    - Redirect URI: Platform: Single-page application (SPA)
       - URI: `http://localhost:5173`
    - Click Register

2. Configure Authentication
    - In the frontend app registration → **Authentication**
       - Under **Platform configurations**, ensure **Single-page application** platform is configured and `http://localhost:5173` is added as a redirect URI
       - Do NOT enable the implicit grant options; MSAL (Authorization Code + PKCE) is used.

3. Configure API permissions
    - In **API permissions** → **Add a permission** → **My APIs** → select the backend app you created
    - Choose **Delegated permissions** → check the `access_as_user` scope you created
    - Also add **Microsoft Graph → Delegated → User.Read** (used for display name/profile)
    - Click **Add permissions**
    - Click **Grant admin consent** (requires an admin) — this avoids per-user consent prompts in dev.

4. Note values to set in frontend `.env`:
    - Directory (tenant) ID → `VITE_AZURE_FRONTEND_TENANT_ID`
    - Application (client) ID → `VITE_AZURE_FRONTEND_CLIENT_ID`
    - Backend client id (from the backend app) → `VITE_AZURE_BACKEND_CLIENT_ID`
    - Optionally, `VITE_AZURE_BACKEND_SCOPE` → `api://<BACKEND_CLIENT_ID>/access_as_user`

---

## 3) Configure the repository environment files

- `frontend/.env` (example):

```ini
# Frontend dev
VITE_API_URL=http://localhost:8000
VITE_AZURE_FRONTEND_TENANT_ID=<TENANT_ID>
VITE_AZURE_FRONTEND_CLIENT_ID=<FRONTEND_CLIENT_ID>
VITE_REDIRECT_URI=http://localhost:5173
# Backend info used by frontend to request scope
VITE_AZURE_BACKEND_CLIENT_ID=<BACKEND_CLIENT_ID>
VITE_AZURE_BACKEND_SCOPE=api://<BACKEND_CLIENT_ID>/access_as_user
```

- Root `.env` (backend) example:

```ini
# Backend
AZURE_TENANT_ID=<TENANT_ID>
AZURE_CLIENT_ID=<BACKEND_CLIENT_ID>
# Optionally AZURE_CLIENT_SECRET if you use confidential flows
```

Notes:
- The backend `AZURE_CLIENT_ID` must match the backend app Application ID.
- `VITE_AZURE_BACKEND_SCOPE` must use the backend app id (not the tenant id).

---

## 4) How the code uses these values

- Frontend: `frontend/src/authConfig.js` builds `tokenRequest.scopes` from `VITE_AZURE_BACKEND_SCOPE` or `api://<VITE_AZURE_BACKEND_CLIENT_ID>/access_as_user`. It also uses `VITE_AZURE_FRONTEND_CLIENT_ID` and `VITE_AZURE_FRONTEND_TENANT_ID` for the MSAL client configuration.
- Backend: `backend/auth.py` loads `AZURE_TENANT_ID` and `AZURE_CLIENT_ID` and validates incoming access tokens' `issuer` and `aud` (audience) claims accordingly.

---

## 5) Local testing checklist

1. Start backend (reload env) — ensure `.env` has the backend values set

```bash
cd /path/to/repo
source .venv/bin/activate
# ensure .env is loaded by your process manager or export env vars manually
PYTHONPATH=$(pwd) uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

2. Start frontend

```bash
cd frontend
npm install
npm run dev
```

3. Use the SPA to sign in. In DevTools Console run:

```js
// show MSAL accounts
window.__msal?.getAllAccounts().then(console.log)
// debug token acquisition (helper provided by project)
debugAuth().then(console.log)
```

4. Inspect access token
- Decode it (jwt.ms or jwt.io) and confirm:
   - `iss` = `https://login.microsoftonline.com/<TENANT_ID>/v2.0`
   - `aud` = `<BACKEND_CLIENT_ID>` (or the Application ID URI expected by your backend)
   - `scp` contains `api://<BACKEND_CLIENT_ID>/access_as_user`

5. Make API call with token (curl example)

```bash
curl -H "Authorization: Bearer <ACCESS_TOKEN>" http://localhost:8000/config
```

Expect HTTP 200 and valid JSON response.

---

## 6) Troubleshooting quick guide

- 401 Unauthorized
   - Verify `AZURE_TENANT_ID` and `AZURE_CLIENT_ID` are set in the backend environment and match the values from Azure.
   - Verify the token `aud` claim matches `AZURE_CLIENT_ID` or the Application ID URI your backend uses.
   - Verify the frontend requested the correct scope (`VITE_AZURE_BACKEND_SCOPE`).

- Token not being sent from the SPA
   - Verify `window.__msal` exists in the browser console
   - Ensure `debugAuth()` returns `tokenInfo.accessToken` — if not, user is not signed in or the token request failed
   - Check network tab: `Authorization` header should be present

- AADSTS50011 / reply URL mismatch
   - Ensure `VITE_REDIRECT_URI` matches the redirect URI registered in the SPA app registration

- Cross-origin or CORS errors
   - For local dev the backend allows all origins. For production add your frontend origin to the backend's CORS allow list

---

## 7) Security notes

- Do not commit `.env` files with secrets to version control
- For production, use Azure Key Vault and secure secret storage
- Restrict CORS and use HTTPS in production

---

## 8) Quick reference — env vars to set

Frontend: `frontend/.env`
```ini
VITE_AZURE_FRONTEND_TENANT_ID=<TENANT_ID>
VITE_AZURE_FRONTEND_CLIENT_ID=<FRONTEND_CLIENT_ID>
VITE_REDIRECT_URI=http://localhost:5173
VITE_AZURE_BACKEND_CLIENT_ID=<BACKEND_CLIENT_ID>
VITE_AZURE_BACKEND_SCOPE=api://<BACKEND_CLIENT_ID>/access_as_user
VITE_API_URL=http://localhost:8000
```

Backend: `.env` (repo root)
```ini
AZURE_TENANT_ID=<TENANT_ID>
AZURE_CLIENT_ID=<BACKEND_CLIENT_ID>
```

---

If you want, I can also add a small script to automate creating the `.env` files from the App Registration IDs you provide. Just tell me which IDs to use and I will create the `.env` files for you (local-only change, not committed credentials).
```
