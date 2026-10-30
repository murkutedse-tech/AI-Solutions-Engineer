"""Mock OpenAI-compatible providers with switchable failure modes, for demos.

    python -m task4_model_router.mock_providers   # primary on :9101, secondary on :9102

Switch a provider's behaviour at runtime:
    curl -X POST localhost:9101/_mock/mode -d '{"mode": "rate_limited"}'
Modes: ok | rate_limited | slow | error | garbage
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

MODES = {"ok", "rate_limited", "slow", "error", "garbage"}


def create_mock_llm(name: str, mode: str = "ok", *, slow_s: float = 5.0) -> Starlette:
    state = {"mode": mode, "calls": 0}

    async def completions(request: Request):
        state["calls"] += 1
        body = await request.json()
        current = state["mode"]
        if current == "rate_limited":
            return JSONResponse({"error": {"message": f"{name}: org quota exceeded", "type": "rate_limit"}}, status_code=429, headers={"retry-after": "20"})
        if current == "error":
            return PlainTextResponse(f'Traceback (most recent call last):\n  File "/srv/{name}/inference.py", line 88\nRuntimeError: CUDA OOM on node gpu-17', status_code=500)
        if current == "garbage":
            return PlainTextResponse("<html>502 bad gateway</html>", status_code=200)
        if current == "slow":
            await asyncio.sleep(slow_s)
        prompt = " ".join(m.get("content") or "" for m in body.get("messages", []) if isinstance(m.get("content"), str))
        text = f"[{name}:{body.get('model')}] answer to: {prompt[:60]}"
        prompt_tokens, completion_tokens = max(1, len(prompt) // 4), max(1, len(text) // 4)
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
        })

    async def set_mode(request: Request):
        new_mode = json.loads(await request.body()).get("mode")
        if new_mode not in MODES:
            return JSONResponse({"error": f"mode must be one of {sorted(MODES)}"}, status_code=400)
        state["mode"] = new_mode
        return JSONResponse(state)

    app = Starlette(routes=[Route("/v1/chat/completions", completions, methods=["POST"]), Route("/_mock/mode", set_mode, methods=["POST"])])
    app.state.mock = state
    return app


if __name__ == "__main__":
    import uvicorn

    async def main() -> None:
        servers = [
            uvicorn.Server(uvicorn.Config(create_mock_llm("primary"), host="127.0.0.1", port=9101, log_level="warning")),
            uvicorn.Server(uvicorn.Config(create_mock_llm("secondary"), host="127.0.0.1", port=9102, log_level="warning")),
        ]
        await asyncio.gather(*(s.serve() for s in servers))

    asyncio.run(main())
