# Using Local Models

This document explains how to configure and use local LLM and embedding models instead of OpenAI.

## Configuration

You can change between OpenAI and Local models in two ways:

### 1. Via the UI (Recommended)

1. Open the application at `http://localhost:5173`
2. Look at the **Configuration** card in the sidebar
3. Use the dropdown selects to change:
   - **LLM Provider**: Switch between OpenAI and Local
   - **Embeddings Provider**: Switch between OpenAI and Local
4. Changes are saved to the database automatically

### 2. Via Environment Variables

Edit the `.env` file in the project root:

```bash
LLM_PROVIDER=LOCAL              # or OPENAI
EMBEDDING_PROVIDER=LOCAL        # or OPENAI
```

## Local LLM Models

### Supported Model Types

The application supports two types of local LLM models:

1. **HuggingFace Models** (recommended for most users)
2. **Local Model Files** (for advanced users with downloaded models)

### Option 1: HuggingFace Models (Easier)

Set the model using a HuggingFace model identifier in `.env`:

```bash
LOCAL_LLM_MODEL=google/flan-t5-small    # Small, fast (~250MB)
# or
LOCAL_LLM_MODEL=google/flan-t5-base     # Medium quality (~1GB)
# or
LOCAL_LLM_MODEL=google/flan-t5-large    # Better quality (~3GB)
```

**Recommended models:**
- `google/flan-t5-small` - Best for testing, low memory usage
- `google/flan-t5-base` - Good balance of speed and quality
- `google/flan-t5-large` - Best quality, requires more RAM

The model will be automatically downloaded on first use and cached.

### Option 2: Local Model Files (Advanced)

If you have models downloaded locally, use an absolute path:

```bash
LOCAL_LLM_MODEL=/path/to/your/model/directory
```

**Requirements:**
- The directory must contain valid model files (config.json, model weights, tokenizer files)
- The model must be compatible with HuggingFace's `AutoModelForSeq2SeqLM`

## Local Embedding Models

For embeddings, set in `.env`:

```bash
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Recommended models:**
- `sentence-transformers/all-MiniLM-L6-v2` - Fast, lightweight (default)
- `sentence-transformers/all-mpnet-base-v2` - Better quality, slower
- `BAAI/bge-small-en-v1.5` - Good for English text

## Dependencies

To use local models, you need additional Python packages:

```bash
# Required for local LLMs
pip install transformers torch

# Required for local embeddings
pip install sentence-transformers
```

## Troubleshooting

### Error: "Repo id must be in the form..."

**Problem:** The `LOCAL_LLM_MODEL` is set to a path that doesn't exist.

**Solution:** 
- Use a HuggingFace model ID like `google/flan-t5-small`
- OR ensure the local path exists and contains valid model files

**Example fix in `.env`:**
```bash
# Change this:
LOCAL_LLM_MODEL=/models/flan-t5-large

# To this:
LOCAL_LLM_MODEL=google/flan-t5-small
```

### Error: "transformers is not installed"

**Solution:**
```bash
source .venv/bin/activate
pip install transformers torch
```

### Error: "sentence-transformers is not installed"

**Solution:**
```bash
source .venv/bin/activate
pip install sentence-transformers
```

### Model downloads are slow

**Note:** On first use, models will be downloaded from HuggingFace. This can take several minutes depending on model size and internet speed.

Model sizes:
- `flan-t5-small`: ~250MB
- `flan-t5-base`: ~1GB
- `flan-t5-large`: ~3GB
- `all-MiniLM-L6-v2`: ~90MB

### Running out of memory

If you get memory errors, try a smaller model:
- Switch from `flan-t5-large` to `flan-t5-small`
- Reduce `LOCAL_LLM_MAX_NEW_TOKENS` in `.env` (default: 256)

## Performance Comparison

| Provider | Speed | Quality | Cost | Internet Required |
|----------|-------|---------|------|-------------------|
| OpenAI   | Fast  | Excellent | $$ | Yes |
| Local (flan-t5-small) | Medium | Good | Free | No (after download) |
| Local (flan-t5-base) | Slow | Very Good | Free | No (after download) |
| Local (flan-t5-large) | Very Slow | Excellent | Free | No (after download) |

## Current Configuration

To check your current configuration:

```bash
curl http://localhost:8000/config
```

This will return:
```json
{
  "LLM_PROVIDER": "OPENAI",
  "EMBEDDING_PROVIDER": "OPENAI"
}
```
