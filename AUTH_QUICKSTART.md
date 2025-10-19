# Microsoft Entra Authentication - Quick Start

This is a condensed version of the authentication setup. For full details, see [AUTHENTICATION.md](./AUTHENTICATION.md).

## 1. Azure Portal Setup (5 minutes)

1. Go to [Azure Portal](https://portal.azure.com) → **Azure Active Directory** → **App registrations** → **New registration**
2. Set:
   - **Name**: RAG Chat Application
   - **Account type**: Single tenant (your org)
   - **Redirect URI**: Single-page application → `http://localhost:5173`
3. Click **Register**
4. Go to **Authentication** → Add redirect URI for production if needed
5. Go to **API permissions** → Add → **Microsoft Graph** → **Delegated** → `User.Read`
6. Copy from **Overview** page:
   - **Directory (tenant) ID**
   - **Application (client) ID**

## 2. Backend Configuration (2 minutes)

Edit `.env` in project root:

```bash
# Add these lines
AZURE_TENANT_ID=your-tenant-id-here
AZURE_CLIENT_ID=your-client-id-here
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Frontend Configuration (2 minutes)

Create `frontend/.env`:

```bash
VITE_AZURE_TENANT_ID=your-tenant-id-here
VITE_AZURE_CLIENT_ID=your-client-id-here
VITE_REDIRECT_URI=http://localhost:5173
VITE_API_URL=http://localhost:8000
```

Install dependencies:

```bash
cd frontend
npm install
```

## 4. Test (1 minute)

```bash
# Terminal 1 - Start backend
docker compose up -d db
source .venv/bin/activate
PYTHONPATH=$(pwd) python backend/main.py

# Terminal 2 - Start frontend
cd frontend
npm run dev

# Open browser to http://localhost:5173
# Click "Sign in with Microsoft"
# Complete login
# Start chatting!
```

## Common Issues

| Issue | Solution |
|-------|----------|
| "Reply URL mismatch" | Ensure redirect URI in Azure matches `http://localhost:5173` exactly |
| 401 Unauthorized | Check AZURE_TENANT_ID and AZURE_CLIENT_ID match in both .env files |
| MSAL errors | Restart frontend after changing .env: `npm run dev` |

## Disable Authentication for Testing

Comment out authentication in `backend/main.py`:

```python
# Change this:
async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):

# To this:
async def chat(req: ChatRequest):
```

## Next Steps

- See [AUTHENTICATION.md](./AUTHENTICATION.md) for:
  - Role-based access control (RBAC)
  - Custom scopes
  - Production deployment
  - Troubleshooting guide
  - Security best practices
