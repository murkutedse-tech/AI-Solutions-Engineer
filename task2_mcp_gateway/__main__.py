"""Run the gateway:  python -m task2_mcp_gateway  (env: DOWNSTREAM_MCP_URL, GATEWAY_JWT_SECRET, PORT)."""

import os

import uvicorn

from .gateway import app_from_env

uvicorn.run(app_from_env(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8080")))
