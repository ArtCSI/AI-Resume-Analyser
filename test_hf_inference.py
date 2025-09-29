from dotenv import load_dotenv
load_dotenv()

import os, json, sys
from huggingface_hub import HfApi
import requests

token = os.getenv("HF_API_KEY", "").strip()
model = os.getenv("HF_MODEL", "google/flan-t5-base").strip()

print("HF_MODEL:", repr(model))
print("HF_API_KEY present:", bool(token), "len=", len(token) if token else 0)

if not token:
    print("ERROR: HF_API_KEY not set. Export it or ensure .env is loaded.")
    sys.exit(1)

api = HfApi()
try:
    info = api.model_info(model, token=token)
    print("model_info OK:", getattr(info, "modelId", "unknown"))
except Exception as e:
    print("model_info error:", repr(e))

# Direct HTTP inference call (explicit task param for text-to-text)
url = f"https://api-inference.huggingface.co/models/{model}?task=text-to-text-generation"
headers = {"Authorization": f"Bearer {token}"}
payload = {"inputs": "Summarize: Experienced Python developer who built ETL pipelines.", "parameters": {"max_new_tokens": 40}, "options": {"wait_for_model": True}}

print("POST", url)
try:
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    print("HTTP status:", r.status_code)
    try:
        print("Response JSON:", json.dumps(r.json(), indent=2)[:2000])
    except Exception:
        print("Response text:", r.text[:2000])
except Exception as e:
    print("HTTP request failed:", repr(e))