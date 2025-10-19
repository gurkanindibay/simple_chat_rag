# Fixing Authentication Error

## Problem
Your application is showing this error:
```
openai.AuthenticationError: Error code: 401 - Incorrect API key provided
```

## Cause
Your OpenAI API key in the `.env` file is either:
1. **Expired** - OpenAI keys can expire
2. **Invalid** - The key format is incorrect or has line breaks
3. **Revoked** - The key was deleted from your OpenAI account

## Solution

### Step 1: Get a New OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in with your OpenAI account
3. Click **"+ Create new secret key"**
4. Give it a name (e.g., "RAG Chat App")
5. Copy the key **immediately** (you won't be able to see it again!)

### Step 2: Update Your `.env` File

Open `/Users/gurkan_indibay/source/ai_tryouts/.env` and update:

```bash
# IMPORTANT: The API key must be on ONE line with NO line breaks!
OPENAI_API_KEY=sk-proj-YOUR_NEW_KEY_HERE

# Rest of your configuration
PDF_PATH=./citus-doc-readthedocs-io-en-latest.pdf
PORT=8000
EMBEDDING_PROVIDER=OPENAI
LLM_PROVIDER=OPENAI
```

**⚠️ CRITICAL**: Make sure the API key is:
- On a **single line** (no line breaks in the middle!)
- Has no extra spaces before or after
- Starts with `sk-proj-` or `sk-`

### Step 3: Restart Your Application

After updating the `.env` file:

```bash
# Stop the current server (Ctrl+C in the terminal)

# Restart it
cd /Users/gurkan_indibay/source/ai_tryouts
source .venv/bin/activate
PYTHONPATH=/Users/gurkan_indibay/source/ai_tryouts uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Step 4: Test It

```bash
# Test that the config loads correctly
curl http://localhost:8000/config -H "Authorization: Bearer YOUR_TOKEN"

# Test a chat (after ingesting a PDF)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"question": "Test question"}'
```

## Alternative: Use Azure OpenAI Instead

If you have issues with OpenAI, you can switch to Azure OpenAI:

1. Follow the setup guide in `AZURE_OPENAI_QUICKSTART.md`
2. Update your `.env`:
   ```bash
   LLM_PROVIDER=AZURE_OPENAI
   EMBEDDING_PROVIDER=AZURE_OPENAI
   
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-azure-key
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
   ```

## Common Mistakes

### Mistake 1: Line Break in API Key
❌ **Wrong:**
```bash
OPENAI_API_KEY=sk-proj-abcd1234
efgh5678
```

✅ **Correct:**
```bash
OPENAI_API_KEY=sk-proj-abcd1234efgh5678
```

### Mistake 2: Spaces Around the Key
❌ **Wrong:**
```bash
OPENAI_API_KEY= sk-proj-abcd1234efgh5678 
```

✅ **Correct:**
```bash
OPENAI_API_KEY=sk-proj-abcd1234efgh5678
```

### Mistake 3: Using an Old/Expired Key
- OpenAI keys don't have an expiration date, but can be revoked
- Check your [API keys page](https://platform.openai.com/api-keys) to see if the key still exists
- If you see a key with a ⚠️ warning, it's been revoked

## How to Check If Your Key Works

Use this simple test:

```bash
# Replace YOUR_KEY with your actual key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_KEY"
```

**If it works**, you'll see a list of models.
**If it fails**, you'll see an authentication error.

## Summary

1. ✅ Get new OpenAI API key from platform.openai.com
2. ✅ Update `.env` file (single line, no spaces)
3. ✅ Restart application
4. ✅ Test with curl

The deprecation warnings for `get_relevant_documents` have been fixed in the code, so you won't see those anymore.
