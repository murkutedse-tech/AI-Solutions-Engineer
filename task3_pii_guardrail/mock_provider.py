"""Mock OpenAI-compatible provider that echoes the last user message as a token stream.

Chunks are random 1-6 character slices, so PII regularly straddles chunk boundaries.

Run:  python -m task3_pii_guardrail.mock_provider   (listens on :9002; MOCK_TOKEN_DELAY_MS=20)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

DEFAULT_REPLY = (
    "Sure! The customer is Jane Roe, email jane.roe@example.com, SSN 123-45-6789, "
    "card 4111 1111 1111 1111. Let me know if you need anything else."
)


def _reply_text(body: dict) -> str:
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    return users[-1]["content"] if users and users[-1].get("content") else DEFAULT_REPLY


def create_mock_provider(*, delay_s: float = 0.02, seed: int | None = None) -> Starlette:
    rng = random.Random(seed)

    async def completions(request: Request):
        body = await request.json()
        text, model = _reply_text(body), body.get("model", "mock-1")
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        if not body.get("stream"):
            return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": model,
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})

        def chunk(delta: dict, finish: str | None = None) -> bytes:
            payload = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(payload)}\n\n".encode()

        async def events():
            yield chunk({"role": "assistant", "content": ""})
            i = 0
            while i < len(text):
                n = rng.randint(1, 6)
                yield chunk({"content": text[i:i + n]})
                i += n
                if delay_s:
                    await asyncio.sleep(delay_s)
            yield chunk({}, "stop")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return Starlette(routes=[Route("/v1/chat/completions", completions, methods=["POST"])])


if __name__ == "__main__":
    import uvicorn

    delay = float(os.environ.get("MOCK_TOKEN_DELAY_MS", "20")) / 1000
    uvicorn.run(create_mock_provider(delay_s=delay), host="127.0.0.1", port=int(os.environ.get("PORT", "9002")))
