"""Run the PII-guarded LLM gateway:  python -m task3_pii_guardrail  (env: LLM_PROVIDER_BASE_URL, LLM_PROVIDER_API_KEY, PORT)."""

import os

import uvicorn

from .gateway import app_from_env

uvicorn.run(app_from_env(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8081")))
