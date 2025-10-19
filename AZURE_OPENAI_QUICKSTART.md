# Azure OpenAI Quick Start Guide

This guide will help you quickly set up Azure OpenAI for your RAG application.

## Prerequisites

1. **Azure Subscription**: You need an active Azure subscription
2. **Azure OpenAI Access**: Request access to Azure OpenAI Service (if not already granted)
3. **Application already running**: Your RAG chat application should be already set up

## Quick Setup (5 minutes)

### Step 1: Get Azure OpenAI Credentials

#### Option A: Using Azure AI Foundry (Recommended)
1. Go to [Azure AI Foundry](https://ai.azure.com/)
2. Navigate to your project or create a new one
3. Deploy models:
   - For chat: Deploy `gpt-4o` or `gpt-35-turbo`
   - For embeddings: Deploy `text-embedding-ada-002`
4. Get your credentials:
   - **Endpoint**: e.g., `https://your-resource.openai.azure.com/`
   - **API Key**: Found in "Keys and Endpoint"
   - **Deployment Names**: The names you gave to your models

#### Option B: Using Azure Portal
1. Go to [Azure Portal](https://portal.azure.com)
2. Search for "Azure OpenAI"
3. Create a new Azure OpenAI resource or select existing one
4. Navigate to "Keys and Endpoint"
5. Deploy models in the "Model deployments" section

### Step 2: Configure Your Application

Add these lines to your `.env` file:

```bash
# Set providers to Azure OpenAI
LLM_PROVIDER=AZURE_OPENAI
EMBEDDING_PROVIDER=AZURE_OPENAI

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Model Deployment Names (as configured in Azure)
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

**Important**: Replace the placeholder values with your actual Azure OpenAI credentials!

### Step 3: Install Required Dependencies

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install Azure OpenAI support
pip install langchain-openai azure-identity
```

### Step 4: Restart Your Application

```bash
# If running with Python
pkill -f uvicorn  # Stop existing server
source .venv/bin/activate
PYTHONPATH=/Users/gurkan_indibay/source/ai_tryouts uvicorn backend.main:app --host 0.0.0.0 --port 8000

# If running with Docker
docker-compose down
docker-compose up --build
```

### Step 5: Test the Integration

```bash
# Test configuration endpoint
curl http://localhost:8000/config \
  -H "Authorization: Bearer YOUR_TOKEN"

# Expected response:
# {
#   "LLM_PROVIDER": "AZURE_OPENAI",
#   "EMBEDDING_PROVIDER": "AZURE_OPENAI"
# }

# Test chat (make sure you have PDFs ingested first)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"question": "What is this document about?"}'
```

## Using the UI

You can also switch providers via the web interface:

1. Open `http://localhost:5173`
2. Log in with your credentials
3. Find the **Configuration** card in the sidebar
4. Select "Azure OpenAI" from the dropdowns for:
   - LLM Provider
   - Embedding Provider
5. Changes are saved automatically

## Managing User Permissions

### Option 1: Azure RBAC (Recommended)

Grant users access to your Azure OpenAI resource:

```bash
# Assign role to a user
az role assignment create \
  --assignee user@domain.com \
  --role "Cognitive Services OpenAI Contributor" \
  --scope /subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.CognitiveServices/accounts/{openai-resource-name}

# Assign role to your application
az role assignment create \
  --assignee {service-principal-id} \
  --role "Cognitive Services OpenAI Contributor" \
  --scope /subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.CognitiveServices/accounts/{openai-resource-name}
```

### Option 2: Application-Level Control

Your application already has role-based access control via Microsoft Entra ID:

- Users with `rag_chat_user` role can use the chat
- Users with `rag_admin` role can change configuration

No additional Azure permissions needed if you're using API keys.

## Important Notes

### Re-ingesting PDFs

**IMPORTANT**: If you switch between embedding providers, you MUST re-ingest your PDFs because different providers create different vector dimensions:

- OpenAI: 1536 dimensions
- Azure OpenAI: 1536 dimensions (same as OpenAI)
- Local models: Typically 384 dimensions

Since Azure OpenAI uses the same dimensions as OpenAI, you can switch between them without re-ingesting IF you're using the same embedding model.

### Cost Considerations

Azure OpenAI pricing is per-token:
- **GPT-4**: ~$0.03-0.06 per 1,000 tokens
- **GPT-3.5-Turbo**: ~$0.0015-0.002 per 1,000 tokens
- **Embeddings**: ~$0.0001 per 1,000 tokens

Monitor usage in Azure Portal → Your OpenAI Resource → Metrics

## Troubleshooting

### Error: "langchain-openai is not installed"
```bash
pip install langchain-openai azure-identity
```

### Error: "AZURE_OPENAI_ENDPOINT is not set"
Check your `.env` file and make sure all Azure variables are set correctly.

### Error: "Deployment not found"
Verify your deployment names in Azure match exactly what's in your `.env` file.

### Error: "Access denied" 
Check:
1. API key is correct
2. Azure RBAC permissions are granted
3. Resource is not in a restricted region

## Next Steps

- **Read Full Documentation**: See [docs/AZURE_OPENAI_SETUP.md](./docs/AZURE_OPENAI_SETUP.md) for detailed setup
- **Security Setup**: Configure Private Link, VNet integration
- **Cost Optimization**: Set up budget alerts and quotas
- **Monitoring**: Enable diagnostic logs and Application Insights

## Comparison with Regular OpenAI

| Feature | Azure OpenAI | OpenAI |
|---------|--------------|--------|
| **Data Residency** | Your Azure region | OpenAI servers (US) |
| **Compliance** | HIPAA, SOC 2, ISO 27001 | Limited |
| **Private Network** | Yes (Private Link) | No |
| **SLA** | 99.9% | Best effort |
| **Cost** | Pay-per-token | Pay-per-token |
| **Models** | Same models | Same models |
| **Setup Complexity** | Medium | Easy |

## Support

For issues:
1. Check logs: `tail -f /var/log/yourapp.log`
2. Review [AZURE_OPENAI_SETUP.md](./docs/AZURE_OPENAI_SETUP.md)
3. Azure Support: [Azure Portal](https://portal.azure.com) → Support

---

**Estimated Setup Time**: 5-10 minutes
**Recommended For**: Enterprise deployments, compliance-sensitive applications, European customers
