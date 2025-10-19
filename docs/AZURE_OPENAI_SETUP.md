# Azure OpenAI Integration Guide

This guide explains how to integrate Azure OpenAI (via Azure AI Foundry) into your RAG application and configure user permissions.

## Table of Contents
1. [Azure AI Foundry Setup](#azure-ai-foundry-setup)
2. [Azure OpenAI Configuration](#azure-openai-configuration)
3. [User Permission Management](#user-permission-management)
4. [Application Configuration](#application-configuration)
5. [Testing the Integration](#testing-the-integration)

---

## Azure AI Foundry Setup

### Step 1: Create Azure OpenAI Resource

1. **Navigate to Azure AI Foundry**:
   - Go to [Azure AI Foundry](https://ai.azure.com/)
   - Or use the Azure Portal: https://portal.azure.com

2. **Create an Azure OpenAI Service**:
   ```bash
   # Via Azure CLI
   az cognitiveservices account create \
     --name <your-openai-resource-name> \
     --resource-group <your-resource-group> \
     --kind OpenAI \
     --sku S0 \
     --location eastus
   ```

3. **Deploy Models**:
   - In Azure AI Foundry, go to your OpenAI resource
   - Navigate to "Model Deployments"
   - Deploy the models you need:
     - **For Chat/LLM**: `gpt-4o`, `gpt-4`, `gpt-35-turbo`
     - **For Embeddings**: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`

4. **Get Your Credentials**:
   - **Endpoint**: Found in "Keys and Endpoint" section (e.g., `https://<your-resource>.openai.azure.com/`)
   - **API Key**: One of the two keys provided
   - **API Version**: Current version (e.g., `2024-02-15-preview`)
   - **Deployment Names**: The names you gave to your deployed models

---

## Azure OpenAI Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Deployment Names (as configured in Azure)
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Set providers to use Azure OpenAI
LLM_PROVIDER=AZURE_OPENAI
EMBEDDING_PROVIDER=AZURE_OPENAI
```

### Docker Configuration

If using Docker, update your `docker-compose.yml`:

```yaml
services:
  app:
    environment:
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION}
      - AZURE_OPENAI_CHAT_DEPLOYMENT=${AZURE_OPENAI_CHAT_DEPLOYMENT}
      - AZURE_OPENAI_EMBEDDING_DEPLOYMENT=${AZURE_OPENAI_EMBEDDING_DEPLOYMENT}
      - LLM_PROVIDER=AZURE_OPENAI
      - EMBEDDING_PROVIDER=AZURE_OPENAI
```

---

## User Permission Management

### Quick Reference - Finding Cognitive Services Permissions

**Can't find "Cognitive Services OpenAI Contributor" role?**

👉 **Quick Steps:**
1. Azure Portal → Your OpenAI Resource
2. Left menu → **Access control (IAM)**
3. Click **+ Add** → **Add role assignment**
4. In the search box, type: **"Cognitive Services"** or **"OpenAI"**
5. Select: **Cognitive Services OpenAI Contributor**
6. Add your users → Review + assign

If you still can't find it, see the detailed guide below with troubleshooting steps.

---

### Option 1: Azure RBAC (Recommended for Production)

Azure provides fine-grained role-based access control for OpenAI resources.

#### Available Roles

1. **Cognitive Services OpenAI User**: Read-only access to view resources
2. **Cognitive Services OpenAI Contributor**: Can use the API and manage deployments
3. **Cognitive Services Contributor**: Full access to the resource

#### Assign Roles via Azure Portal

**Step-by-Step Guide:**

1. **Navigate to Azure Portal**: Go to [https://portal.azure.com](https://portal.azure.com)

2. **Find Your Azure OpenAI Resource**:
   - In the search bar at the top, type "Azure OpenAI"
   - Or navigate to "All resources" and find your OpenAI resource

3. **Open Access Control (IAM)**:
   - Click on your Azure OpenAI resource name
   - In the left sidebar, scroll down and click **"Access control (IAM)"**
   - This is usually in the middle section of the left menu

4. **Add Role Assignment**:
   - Click the **"+ Add"** button at the top
   - Select **"Add role assignment"** from the dropdown

5. **Select the Role** (This is where the Cognitive Services roles are):
   - You'll see three tabs: "Role", "Members", "Review + assign"
   - On the **"Role" tab**, you'll see a search box
   - Type **"Cognitive Services"** in the search box
   - You'll see several options appear:
     - ✅ **Cognitive Services OpenAI Contributor** (RECOMMENDED - allows API usage)
     - **Cognitive Services OpenAI User** (read-only)
     - **Cognitive Services Contributor** (full access)
     - **Cognitive Services User** (basic access)
   - Select **"Cognitive Services OpenAI Contributor"**
   - Click **"Next"** button

6. **Add Members**:
   - On the **"Members" tab**, click **"+ Select members"**
   - In the "Select members" panel that opens on the right:
     - Search for user email addresses (e.g., user@yourcompany.com)
     - Or search for Azure AD groups
     - Or search for managed identities (for applications)
   - Click on the names to add them to "Selected members"
   - Click **"Select"** button

7. **Review and Assign**:
   - Click **"Next"** to go to "Review + assign" tab
   - Review your selections
   - Click **"Review + assign"** button to finalize

**Troubleshooting - Can't Find the Roles?**

If you don't see the Cognitive Services roles:

- **Option 1**: Use the "Job function roles" tab instead of "All roles"
- **Option 2**: Filter by typing "OpenAI" or "Cognitive" in the search
- **Option 3**: Scroll down - there are many roles, use Page Down or scroll wheel
- **Option 4**: Check that you have Owner or User Access Administrator role on the subscription

**Visual Guide - Where to Find Cognitive Services Roles:**

```
Azure Portal → Your OpenAI Resource → Access control (IAM)
├─ + Add (button at top)
│  └─ Add role assignment
│     └─ Role tab
│        ├─ Search box: Type "Cognitive Services" or "OpenAI"
│        └─ Results will show:
│           ├─ Cognitive Services OpenAI Contributor ✓ (Select this)
│           ├─ Cognitive Services OpenAI User
│           ├─ Cognitive Services Contributor
│           └─ Cognitive Services User
```

**Alternative Method Using Azure CLI** (if Portal is not working):

```bash
# List all available Cognitive Services roles
az role definition list --query "[?contains(roleName, 'Cognitive')].{Name:roleName, Id:name}" --output table

# You should see:
# - Cognitive Services OpenAI Contributor
# - Cognitive Services OpenAI User
# - Cognitive Services Contributor
# - Cognitive Services User

# Assign role to a user (replace placeholders)
az role assignment create \
  --assignee user@domain.com \
  --role "Cognitive Services OpenAI Contributor" \
  --scope /subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/YOUR_RESOURCE_GROUP/providers/Microsoft.CognitiveServices/accounts/YOUR_OPENAI_RESOURCE_NAME
```

#### Common Issues and Solutions

**Issue 1: "I don't have permission to add role assignments"**
- **Solution**: Ask your Azure subscription administrator to either:
  - Grant you "Owner" or "User Access Administrator" role on the OpenAI resource
  - Add the users for you

**Issue 2: "The role list is empty or doesn't show Cognitive Services roles"**
- **Solution**: 
  - Make sure you're in the correct Azure subscription (check top-right corner)
  - Ensure the Azure OpenAI resource is properly created
  - Try using Azure CLI method instead

**Issue 3: "User can't access Azure OpenAI even after granting permission"**
- **Solution**: 
  - Wait 5-10 minutes for permissions to propagate
  - Ask user to sign out and sign back in to Azure
  - Verify the role was applied at the correct scope (resource level, not subscription)

#### Assign Roles via Azure CLI

```bash
# Get your Azure OpenAI resource ID
RESOURCE_ID=$(az cognitiveservices account show \
  --name <your-openai-resource-name> \
  --resource-group <your-resource-group> \
  --query id -o tsv)

# Assign role to a user
az role assignment create \
  --assignee <user-email@domain.com> \
  --role "Cognitive Services OpenAI Contributor" \
  --scope $RESOURCE_ID

# Assign role to a service principal (for your app)
az role assignment create \
  --assignee <service-principal-id> \
  --role "Cognitive Services OpenAI Contributor" \
  --scope $RESOURCE_ID
```

### Option 2: Using Azure Entra ID (Azure AD) Authentication

For enhanced security, use Azure Entra ID instead of API keys:

#### Setup Steps

1. **Create or Use Existing App Registration**:
   ```bash
   # Create app registration
   az ad app create --display-name "RAG-Chat-App"
   
   # Get the Application (client) ID
   APP_ID=$(az ad app list --display-name "RAG-Chat-App" --query [0].appId -o tsv)
   ```

2. **Grant API Permissions**:
   - In Azure Portal → Azure Active Directory → App registrations
   - Select your app
   - Go to "API permissions"
   - Add "Cognitive Services" permissions

3. **Update Your Application Configuration**:
   ```bash
   # .env file
   AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
   AZURE_OPENAI_USE_AAD=true
   AZURE_CLIENT_ID=<your-app-client-id>
   AZURE_CLIENT_SECRET=<your-app-client-secret>
   AZURE_TENANT_ID=<your-tenant-id>
   ```

### Option 3: Application-Level Permission Control

Leverage your existing authentication system to control who can use Azure OpenAI:

#### In Your Application (`backend/auth.py`)

The application already has role-based access control. Map Azure OpenAI usage to specific roles:

```python
# Users with 'rag_chat_user' role can query
# Users with 'rag_admin' role can configure providers
```

#### Custom Permission Logic

You can add additional checks in `backend/main.py`:

```python
@app.post("/chat")
async def chat(req: ChatRequest, current_user: dict = Depends(require_role('rag_chat_user'))):
    # Check if user is allowed to use Azure OpenAI
    if LLM_PROVIDER == 'AZURE_OPENAI':
        # Add custom logic here
        # e.g., check user's subscription tier, usage limits, etc.
        pass
```

---

## Application Configuration

### Managing Providers via UI

1. Open your application at `http://localhost:5173`
2. Navigate to the **Configuration** card in the sidebar
3. Select **Azure OpenAI** from the dropdown for:
   - LLM Provider
   - Embedding Provider

### Managing Providers via API

```bash
# Update LLM provider
curl -X POST http://localhost:8000/config/update \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{
    "key": "LLM_PROVIDER",
    "value": "AZURE_OPENAI"
  }'

# Update embedding provider
curl -X POST http://localhost:8000/config/update \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{
    "key": "EMBEDDING_PROVIDER",
    "value": "AZURE_OPENAI"
  }'
```

---

## Testing the Integration

### 1. Test Azure OpenAI Connection

```bash
# Test chat completion
curl -X POST "https://<your-resource>.openai.azure.com/openai/deployments/<deployment-name>/chat/completions?api-version=2024-02-15-preview" \
  -H "Content-Type: application/json" \
  -H "api-key: <your-api-key>" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ],
    "max_tokens": 100
  }'
```

### 2. Test Embeddings

```bash
curl -X POST "https://<your-resource>.openai.azure.com/openai/deployments/<embedding-deployment>/embeddings?api-version=2024-02-15-preview" \
  -H "Content-Type: application/json" \
  -H "api-key: <your-api-key>" \
  -d '{
    "input": "Test embedding"
  }'
```

### 3. Test Application Chat

```bash
# After configuring the app
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{
    "question": "What is this document about?"
  }'
```

### 4. Check Configuration

```bash
curl http://localhost:8000/config \
  -H "Authorization: Bearer <your-token>"
```

Expected response:
```json
{
  "LLM_PROVIDER": "AZURE_OPENAI",
  "EMBEDDING_PROVIDER": "AZURE_OPENAI"
}
```

---

## Cost Management

Azure OpenAI pricing is different from OpenAI:

- **Pay-per-use**: Charged per 1,000 tokens
- **Provisioned Throughput**: Reserved capacity for predictable costs

### Monitoring Usage

1. **Azure Portal**:
   - Navigate to your OpenAI resource
   - Go to "Metrics"
   - View token usage, request counts

2. **Set Budget Alerts**:
   ```bash
   az consumption budget create \
     --name "OpenAI-Monthly-Budget" \
     --amount 100 \
     --time-grain Monthly \
     --category Cost
   ```

3. **Application-Level Tracking**:
   - Log all API calls
   - Track tokens per user
   - Implement rate limiting

---

## Comparison: Azure OpenAI vs OpenAI vs Local

| Feature | Azure OpenAI | OpenAI | Local |
|---------|--------------|--------|-------|
| **Cost** | Pay-per-token | Pay-per-token | Free (hardware) |
| **Data Privacy** | Azure compliance | OpenAI servers | Fully local |
| **Performance** | Fast | Fast | Depends on hardware |
| **Availability** | 99.9% SLA | High | Local only |
| **Enterprise Features** | RBAC, VNet, Private Link | Limited | N/A |
| **Setup Complexity** | Medium | Easy | High |
| **Best For** | Enterprise, compliance | Quick start | Privacy, offline |

---

## Troubleshooting

### Error: "Resource not found"
- **Solution**: Verify your deployment names match exactly

### Error: "Access denied"
- **Solution**: Check RBAC permissions, ensure user has "Cognitive Services OpenAI Contributor" role

### Error: "Rate limit exceeded"
- **Solution**: 
  - Check quota limits in Azure Portal
  - Consider provisioned throughput
  - Implement retry logic with exponential backoff

### Error: "Invalid API version"
- **Solution**: Update `AZURE_OPENAI_API_VERSION` to a supported version

---

## Security Best Practices

1. **Never commit API keys**: Use environment variables or Azure Key Vault
2. **Use Azure Entra ID**: Prefer AAD authentication over API keys
3. **Enable Private Link**: For production, use private endpoints
4. **Rotate Keys Regularly**: Set up key rotation policies
5. **Monitor Access Logs**: Enable diagnostic settings
6. **Implement Rate Limiting**: Protect against abuse
7. **Use Managed Identities**: For Azure-hosted applications

---

## Additional Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure AI Foundry](https://ai.azure.com/)
- [Azure RBAC Documentation](https://learn.microsoft.com/en-us/azure/role-based-access-control/)
- [LangChain Azure OpenAI Integration](https://python.langchain.com/docs/integrations/llms/azure_openai)
