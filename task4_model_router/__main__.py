"""Run the routing gateway:  python -m task4_model_router  (see app_from_env for configuration)."""

import os

import uvicorn

from .app import app_from_env

uvicorn.run(app_from_env(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8082")))
