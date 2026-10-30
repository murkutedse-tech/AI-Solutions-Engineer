"""Mock downstream MCP server (Streamable HTTP, JSON or SSE responses).

Records every request it receives in ``app.state.received`` so tests can prove
that denied calls never reached it.

Run:  python -m task2_mcp_gateway.mock_downstream   (listens on :9001)
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

TOOLS = [
    {"name": "get_weather", "description": "Current weather for a city.", "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "search_docs", "description": "Search internal documentation.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "admin_reset_key", "description": "Rotate a tenant API key.", "inputSchema": {"type": "object", "properties": {"tenant": {"type": "string"}}, "required": ["tenant"]}},
    {"name": "admin_delete_user", "description": "Delete a user account.", "inputSchema": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}},
]
TOOL_NAMES = {t["name"] for t in TOOLS}


def _result(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def handle(msg: dict[str, Any]) -> dict[str, Any]:
    method, msg_id, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if method == "initialize":
        return _result(msg_id, {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "mock-downstream", "version": "1.0.0"},
        })
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        return _result(msg_id, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        if name not in TOOL_NAMES:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32602, "message": f"Unknown tool: {name}"}}
        text = f"executed {name} with {json.dumps(params.get('arguments') or {}, sort_keys=True)}"
        return _result(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "Method not found"}}


def create_mock_app(*, respond_with_sse: bool = False) -> Starlette:
    async def mcp(request: Request) -> Response:
        raw = await request.body()
        request.app.state.received.append({"method": request.method, "headers": dict(request.headers), "body": raw})
        if request.method == "DELETE":
            return Response(status_code=200)
        if request.method == "GET":
            return Response(status_code=405)

        msg = json.loads(raw)
        if "method" not in msg or "id" not in msg:  # notification or client response
            return Response(status_code=202)
        reply = handle(msg)
        headers = {"mcp-session-id": request.app.state.session_id} if msg["method"] == "initialize" else {}
        if respond_with_sse:
            async def events():
                yield b": keep-alive\n\n"
                yield f"event: message\ndata: {json.dumps(reply)}\n\n".encode()
            return StreamingResponse(events(), media_type="text/event-stream", headers=headers)
        return JSONResponse(reply, headers=headers)

    app = Starlette(routes=[Route("/mcp", mcp, methods=["GET", "POST", "DELETE"])])
    app.state.received = []
    app.state.session_id = uuid.uuid4().hex
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_mock_app(respond_with_sse=os.environ.get("MOCK_SSE") == "1"), host="127.0.0.1", port=int(os.environ.get("PORT", "9001")))
