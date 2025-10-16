"""Download a Hugging Face model into ./models using huggingface_hub.

Usage:
  . .venv/bin/activate
  python scripts/download_model.py --model google/flan-t5-large --dest models/flan-t5-large

This uses huggingface_hub.snapshot_download which will download the files needed
for Transformers to load the model from a local path.

Note: Large models may be many GB. Ensure you have enough disk space.
"""
from argparse import ArgumentParser
from huggingface_hub import snapshot_download

parser = ArgumentParser()
parser.add_argument('--model', '-m', required=True, help='Hugging Face model id (e.g. google/flan-t5-large)')
parser.add_argument('--dest', '-d', default='models', help='Destination folder to place model files')
parser.add_argument('--repo-type', default='model', help='Repo type (model/dataset). Default: model')
args = parser.parse_args()

print(f"Downloading model {args.model} to {args.dest} ...")
path = snapshot_download(repo_id=args.model, repo_type=args.repo_type, local_dir=args.dest)
print('Downloaded to:', path)
print('\nYou can now set LOCAL_LLM_MODEL to a local path or use the repo id directly in the .env')
