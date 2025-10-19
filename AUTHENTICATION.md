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
3. Under **Implicit grant and hybrid flows**, ensure these are **unchecked**:
   - Access tokens
   - ID tokens
   (MSAL.js 2.0+ uses auth code flow with PKCE, not implicit flow)
4. Click **Save**

### Step 3: Configure API Permissions

1. Go to **API permissions**
2. Add the following permissions:
   - **Microsoft Graph** → **Delegated permissions** → `User.Read`
3. Click **Add permissions**
4. (Optional) Click **Grant admin consent** if required by your organization

### Step 4: Expose an API (Optional, for API scopes)

If you want to define custom scopes for your backend API:

1. Go to **Expose an API**
2. Click **Add a scope**
3. Accept the default Application ID URI or customize it
4. Add a scope:
   - **Scope name**: `access_as_user`
   - **Who can consent**: Admins and users
   - **Admin consent display name**: Access RAG Chat API
   - **Admin consent description**: Allows the app to access the RAG Chat API on behalf of the user
5. Click **Add scope**

### Step 5: Note Your Configuration Values

Copy the following values (you'll need them for configuration):
- **Directory (tenant) ID**: Found on the **Overview** page
- **Application (client) ID**: Found on the **Overview** page

## Backend Configuration

### Step 1: Install Dependencies

The required packages are already in `requirements.txt`:

```bash
cd /path/to/your/project
source .venv/bin/activate  # or your virtual environment
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

Create or update your `.env` file in the project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Provider Configuration
EMBEDDING_PROVIDER=OPENAI
LLM_PROVIDER=OPENAI

# Microsoft Entra Authentication
AZURE_TENANT_ID=your-tenant-id-from-azure-portal
AZURE_CLIENT_ID=your-client-id-from-azure-portal
AZURE_CLIENT_SECRET=optional-client-secret  # Only needed for backend-only flows
```

Replace:
- `your-tenant-id-from-azure-portal` with your **Directory (tenant) ID**
- `your-client-id-from-azure-portal` with your **Application (client) ID**

### Step 3: Protected Endpoints

The following endpoints now require authentication:
- `POST /chat` - Send chat messages
- `POST /ingest` - Upload and ingest PDFs
- `POST /config/update` - Update configuration
- `POST /embeddings/delete` - Delete embeddings

Public endpoints (no authentication required):
- `GET /config` - Get configuration
- `GET /ingestion-status` - Get ingestion status
- `GET /embeddings/status` - Get embeddings status

## Frontend Configuration

### Step 1: Install Dependencies

```bash
cd frontend
npm install
```

Dependencies are already in `package.json`:
- `@azure/msal-browser`
- `@azure/msal-react`

### Step 2: Configure Environment Variables

Create a `.env` file in the `frontend/` directory:

```bash
# Microsoft Entra Configuration
VITE_AZURE_TENANT_ID=your-tenant-id-from-azure-portal
VITE_AZURE_CLIENT_ID=your-client-id-from-azure-portal
VITE_REDIRECT_URI=http://localhost:5173

# Backend API URL
VITE_API_URL=http://localhost:8000
```

Replace:
- `your-tenant-id-from-azure-portal` with your **Directory (tenant) ID**
- `your-client-id-from-azure-portal` with your **Application (client) ID**

### Step 3: Update Configuration (if needed)

The authentication configuration is in `frontend/src/authConfig.js`. You can customize:
- Cache location (sessionStorage vs localStorage)
- Additional scopes
- Token request configuration

## Testing

### Step 1: Start the Backend

```bash
# Ensure PostgreSQL database is running
docker compose up -d db

# Start the backend with authentication
source .venv/bin/activate
PYTHONPATH=/path/to/your/project python backend/main.py
```

### Step 2: Start the Frontend

```bash
cd frontend
npm run dev
```

### Step 3: Test Authentication Flow

1. Open your browser to `http://localhost:5173`
2. You should see the login page
3. Click **Sign in with Microsoft**
4. Complete the Microsoft login process
5. You should be redirected back to the application
6. The chat interface should now be visible
7. Your name and a logout button should appear in the header

### Step 4: Verify Token Authentication

Open browser developer tools and check:
1. Network tab → Look for API calls to `/chat`, `/ingest`, etc.
2. Verify that the `Authorization: Bearer <token>` header is present
3. Check the response - should be 200 OK (not 401 Unauthorized)

## Troubleshooting

### Issue: "AADSTS50011: The reply URL specified in the request does not match"

**Solution**: Ensure the redirect URI in Azure Portal matches exactly what you're using:
- Development: `http://localhost:5173`
- Production: Your production URL

### Issue: "401 Unauthorized" when calling backend APIs

**Possible causes**:
1. **Token not being sent**: Check browser network tab
2. **Token validation failing**: Check backend logs
3. **Environment variables not set**: Verify `.env` files

**Solutions**:
- Ensure `AZURE_TENANT_ID` and `AZURE_CLIENT_ID` match in both frontend and backend
- Check that tokens are being acquired in the browser console
- Verify the audience (aud) claim in the JWT matches your client ID

### Issue: Token validation errors in backend

**Check**:
```bash
# View backend logs for detailed error messages
python backend/main.py
```

Common errors:
- **Invalid issuer**: Tenant ID mismatch
- **Invalid audience**: Client ID mismatch
- **Expired token**: Token lifetime expired (usually 1 hour)

### Issue: MSAL initialization errors

**Solution**: Check `frontend/src/authConfig.js`:
- Ensure environment variables are prefixed with `VITE_`
- Restart the frontend dev server after changing `.env`

### Issue: CORS errors

**Solution**: The backend is configured to allow all origins in development:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, restrict to specific origins.

## Security Best Practices

### 1. Production Configuration

For production deployments:
- Use HTTPS only
- Restrict CORS to specific domains
- Use environment-specific redirect URIs
- Enable logging and monitoring

### 2. Token Storage

- Tokens are stored in `sessionStorage` by default (cleared on tab close)
- For persistent login, change to `localStorage` in `authConfig.js`
- Never store tokens in cookies without proper security headers

### 3. API Permissions

- Request minimum necessary permissions
- Use delegated permissions (not application permissions) for user-context APIs
- Regularly review and audit granted permissions

### 4. Secret Management

- Never commit `.env` files to version control
- Use Azure Key Vault or similar for production secrets
- Rotate secrets regularly

## Advanced Configuration

### Role-Based Access Control (RBAC)

The backend includes role checking utilities. To use roles:

1. In Azure Portal, add app roles to your app registration
2. Assign users to roles
3. In backend, protect endpoints with roles:

```python
from backend.auth import require_role

@app.post("/admin-only", dependencies=[Depends(require_role("Admin"))])
async def admin_endpoint():
    return {"message": "Admin access granted"}
```

### Custom Scopes

To use custom API scopes:

1. Expose API and define scopes in Azure Portal
2. Update `frontend/src/authConfig.js`:

```javascript
export const tokenRequest = {
  scopes: [
    `api://${import.meta.env.VITE_AZURE_CLIENT_ID}/access_as_user`,
    'User.Read'
  ],
};
```

## References

- [Microsoft Identity Platform Documentation](https://docs.microsoft.com/en-us/azure/active-directory/develop/)
- [MSAL.js Documentation](https://github.com/AzureAD/microsoft-authentication-library-for-js)
- [PyJWT Documentation](https://pyjwt.readthedocs.io/)

## Support

For issues specific to:
- **Azure AD/Entra**: Contact your Azure administrator
- **MSAL**: Check [MSAL.js GitHub issues](https://github.com/AzureAD/microsoft-authentication-library-for-js/issues)
- **Application**: Create an issue in this repository
