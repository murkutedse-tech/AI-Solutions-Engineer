"""Task 2: MCP gateway authentication, JSON-RPC parsing, admin_* tool authorization and forwarding."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from task2_mcp_gateway.auth import JWTVerifier
from task2_mcp_gateway.gateway import GatewayConfig, create_app, is_admin_tool
from task2_mcp_gateway.mock_downstream import create_mock_app

SECRET = "test-secret"
DOWNSTREAM = "http://downstream.internal/mcp"
verifier = JWTVerifier(SECRET)
ADMIN = {"Authorization": f"Bearer {verifier.mint('root', 'admin')}"}
VIEWER = {"Authorization": f"Bearer {verifier.mint('alice', 'viewer')}"}


def rpc(method: str, params=None, msg_id=1) -> dict:
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def tool_call(name: str, arguments=None, msg_id=1) -> dict:
    return rpc("tools/call", {"name": name, "arguments": arguments or {}}, msg_id)


def make_env(*, sse: bool = False, max_body_bytes: int = 1_048_576):
    downstream = create_mock_app(respond_with_sse=sse)
    upstream_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=downstream))
    gateway = create_app(GatewayConfig(DOWNSTREAM, max_body_bytes=max_body_bytes), verifier, http_client=upstream_client)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway")
    return client, downstream.state.received


@pytest.fixture
async def env():
    client, received = make_env()
    async with client:
        yield client, received


# --------------------------------------------------------------------------- authentication


@pytest.mark.anyio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer not.a.jwt"},
        {"Authorization": f"Bearer {JWTVerifier('other-secret').mint('mallory', 'admin')}"},
        {"Authorization": f"Bearer {verifier.mint('bob', 'admin', ttl_s=-3600)}"},
    ],
    ids=["missing", "basic", "empty", "malformed", "wrong-signature", "expired"],
)
async def test_rejects_bad_credentials_before_touching_downstream(env, headers):
    client, received = env
    resp = await client.post("/mcp", json=rpc("tools/list"), headers=headers)
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"].startswith("Bearer")
    assert resp.json()["error"]["code"] == -32000
    assert received == []


@pytest.mark.anyio
async def test_rejects_alg_none_and_unknown_role(env):
    client, received = env

    def b64(obj) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    alg_none = f"{b64({'alg': 'none'})}.{b64({'sub': 'x', 'role': 'admin', 'exp': 9999999999})}."
    for token in (alg_none, verifier.mint("eve", "superuser")):  # type: ignore[arg-type]
        resp = await client.post("/mcp", json=rpc("tools/list"), headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401
    assert received == []


# --------------------------------------------------------------------------- authorization


@pytest.mark.anyio
async def test_tools_list_is_forwarded_transparently(env):
    client, received = env
    resp = await client.post("/mcp", json=rpc("tools/list", msg_id="abc"), headers=VIEWER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "abc"
    assert "admin_reset_key" in {t["name"] for t in body["result"]["tools"]}
    assert len(received) == 1 and json.loads(received[0]["body"]) == rpc("tools/list", msg_id="abc")


@pytest.mark.anyio
async def test_viewer_can_call_regular_tool(env):
    client, received = env
    resp = await client.post("/mcp", json=tool_call("get_weather", {"city": "Pune"}, msg_id=7), headers=VIEWER)
    assert resp.json()["result"]["content"][0]["text"].startswith("executed get_weather")
    assert len(received) == 1


@pytest.mark.anyio
async def test_viewer_admin_tool_call_is_intercepted(env):
    client, received = env
    resp = await client.post("/mcp", json=tool_call("admin_reset_key", {"tenant": "acme"}, msg_id=42), headers=VIEWER)
    assert resp.status_code == 200
    assert resp.json() == {
        "jsonrpc": "2.0",
        "id": 42,
        "error": {
            "code": -32001,
            "message": "Unauthorized Tool Call",
            "data": {"tool": "admin_reset_key", "required_role": "admin", "role": "viewer"},
        },
    }
    assert received == [], "denied call must never reach the downstream server"


@pytest.mark.anyio
async def test_admin_can_call_admin_tool(env):
    client, received = env
    resp = await client.post("/mcp", json=tool_call("admin_reset_key", {"tenant": "acme"}), headers=ADMIN)
    assert resp.json()["result"]["content"][0]["text"] == 'executed admin_reset_key with {"tenant": "acme"}'
    assert received[0]["headers"]["x-authenticated-role"] == "admin"


@pytest.mark.parametrize(
    "name",
    ["admin_reset_key", "ADMIN_reset_key", "Admin_Delete_User", " admin_reset_key", "​admin_reset_key", "ａｄｍｉｎ_reset_key"],
)
def test_admin_tool_detection_resists_lookalikes(name):
    assert is_admin_tool(name)


@pytest.mark.parametrize("name", ["get_weather", "administer_survey", "adminreset", "my_admin_tool", "search_docs"])
def test_non_admin_tools(name):
    assert not is_admin_tool(name)


@pytest.mark.anyio
async def test_duplicate_key_smuggling_is_rejected(env):
    client, received = env
    # Python keeps the last key, many other parsers keep the first: classic parser-differential bypass.
    smuggled = b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key","name":"get_weather"}}'
    resp = await client.post("/mcp", content=smuggled, headers={**VIEWER, "content-type": "application/json"})
    assert resp.status_code == 400 and resp.json()["error"]["code"] == -32600
    assert received == []


@pytest.mark.anyio
async def test_batch_cannot_hide_admin_call_behind_allowed_one(env):
    client, received = env
    batch = [tool_call("get_weather", msg_id=1), tool_call("admin_delete_user", msg_id=2)]
    resp = await client.post("/mcp", json=batch, headers=VIEWER)
    assert resp.status_code == 400 and resp.json()["error"]["code"] == -32600
    assert received == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("payload", "code", "status"),
    [
        (b"{not json", -32700, 400),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","n":NaN}}', -32700, 400),
        (b'"just a string"', -32600, 400),
        (b'{"jsonrpc":"1.0","id":1,"method":"tools/list"}', -32600, 400),
        (b'{"jsonrpc":"2.0","id":true,"method":"tools/list"}', -32600, 400),
        (b'{"jsonrpc":"2.0","id":1,"method":42}', -32600, 400),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":"admin_reset_key"}', -32600, 400),
        (b'{"jsonrpc":"2.0","method":"tools/call","params":{"name":"admin_reset_key"}}', -32600, 400),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{}}', -32602, 200),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":["admin_reset_key"]}}', -32602, 200),
        (b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":["admin_reset_key"]}', -32602, 200),
    ],
)
async def test_malformed_messages_fail_closed(env, payload, code, status):
    client, received = env
    resp = await client.post("/mcp", content=payload, headers={**VIEWER, "content-type": "application/json"})
    assert resp.status_code == status
    assert resp.json()["error"]["code"] == code
    assert received == []


@pytest.mark.anyio
async def test_body_size_limit():
    client, received = make_env(max_body_bytes=256)
    async with client:
        resp = await client.post("/mcp", json=tool_call("search_docs", {"query": "x" * 1000}), headers=ADMIN)
    assert resp.status_code == 413
    assert received == []


# --------------------------------------------------------------------------- forwarding


@pytest.mark.anyio
async def test_forwarding_headers_and_no_token_passthrough(env):
    client, received = env
    resp = await client.post(
        "/mcp",
        json=rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}),
        headers={**VIEWER, "accept": "application/json, text/event-stream", "mcp-protocol-version": "2025-06-18", "x-internal": "secret"},
    )
    assert resp.status_code == 200
    assert resp.headers["mcp-session-id"]
    forwarded = received[0]["headers"]
    assert "authorization" not in forwarded
    assert "x-internal" not in forwarded
    assert forwarded["mcp-protocol-version"] == "2025-06-18"
    assert forwarded["x-authenticated-subject"] == "alice"


@pytest.mark.anyio
async def test_notifications_and_client_responses_pass_through(env):
    client, received = env
    note = await client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=VIEWER)
    assert note.status_code == 202
    reply = await client.post("/mcp", json={"jsonrpc": "2.0", "id": "srv-1", "result": {"roots": []}}, headers=VIEWER)
    assert reply.status_code == 202
    assert len(received) == 2


@pytest.mark.anyio
async def test_sse_response_streams_through():
    client, received = make_env(sse=True)
    async with client:
        resp = await client.post("/mcp", json=tool_call("search_docs", {"query": "mcp"}, msg_id=5), headers=VIEWER)
    assert resp.headers["content-type"].startswith("text/event-stream")
    data_line = next(line for line in resp.text.splitlines() if line.startswith("data: "))
    assert json.loads(data_line[6:])["id"] == 5


@pytest.mark.anyio
async def test_downstream_errors_pass_through_unchanged(env):
    client, _ = env
    resp = await client.post("/mcp", json=tool_call("no_such_tool", msg_id=3), headers=VIEWER)
    assert resp.json()["error"] == {"code": -32602, "message": "Unknown tool: no_such_tool"}


@pytest.mark.anyio
async def test_downstream_unavailable_returns_jsonrpc_error():
    def refuse(_request):
        raise httpx.ConnectError("connection refused")

    gateway = create_app(GatewayConfig(DOWNSTREAM), verifier, http_client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway") as client:
        resp = await client.post("/mcp", json=rpc("tools/list", msg_id=9), headers=VIEWER)
    assert resp.status_code == 502
    assert resp.json()["id"] == 9
    assert resp.json()["error"]["code"] == -32003
    assert "refused" not in resp.text  # no internal detail leaks


@pytest.mark.anyio
async def test_delete_session_requires_auth_and_is_forwarded(env):
    client, received = env
    assert (await client.delete("/mcp")).status_code == 401
    assert (await client.delete("/mcp", headers={**ADMIN, "mcp-session-id": "s1"})).status_code == 200
    assert received[-1]["method"] == "DELETE" and received[-1]["headers"]["mcp-session-id"] == "s1"
