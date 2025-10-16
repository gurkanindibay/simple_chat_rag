"""Quick test runner that loads the local Flan-T5 model and generates an answer.

Usage:
  . .venv/bin/activate
  python scripts/test_local_llm.py --model google/flan-t5-large --prompt "Say hello"

This test uses Transformers to load the model and tokenizer and run generate().
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='google/flan-t5-large')
parser.add_argument('--prompt', '-p', default='Reply with the word pong.')
parser.add_argument('--max-tokens', type=int, default=128)
parser.add_argument('--temperature', type=float, default=0.2)
args = parser.parse_args()

print('Loading model:', args.model)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

inputs = tokenizer(args.prompt, return_tensors='pt', truncation=True, max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}

out = model.generate(
    **inputs,
    max_new_tokens=args.max_tokens,
    do_sample=True,
    temperature=args.temperature,
    top_p=0.95,
    no_repeat_ngram_size=3,
)

generated = tokenizer.decode(out[0], skip_special_tokens=True)
print('Generated:', generated)
