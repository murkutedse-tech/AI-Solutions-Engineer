"""MCP security gateway: an authenticating, tool-authorizing reverse proxy for Streamable HTTP.

    agent ──Bearer JWT──▶ gateway ──(canonical JSON-RPC, no client token)──▶ downstream MCP server

Pipeline for every POST /mcp:
  1. authenticate  Bearer JWT -> Principal(subject, role)                    else HTTP 401
  2. read          bounded body (max_body_bytes)                             else HTTP 413
  3. parse         strict JSON: no duplicate keys, no NaN, no batches        else -32700 / -32600
  4. authorize     tools/call with an admin_* tool requires role=admin       else -32001, downstream never called
  5. forward       re-serialized message + allow-listed headers; response streamed back (JSON or SSE)

Security notes:
  * Step 5 forwards the *re-serialized* object the policy evaluated, never the raw bytes,
    so the downstream server cannot parse a different message than the one authorized
    (duplicate-key / parser-differential smuggling).
  * The client's bearer token is not passed through to the downstream server (MCP spec:
    no token passthrough); the verified identity is sent as X-Authenticated-* headers.
"""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .auth import AuthError, JWTVerifier, Principal, extract_bearer_token, verifier_from_env

logger = logging.getLogger("mcp_gateway")

ADMIN_TOOL_PREFIX = "admin_"

# JSON-RPC error codes. -32768..-32000 is reserved; -32099..-32000 is for implementation-defined server errors.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
UNAUTHENTICATED = -32000
UNAUTHORIZED_TOOL_CALL = -32001
UPSTREAM_UNAVAILABLE = -32003

FORWARD_REQUEST_HEADERS = ("accept", "content-type", "mcp-session-id", "mcp-protocol-version", "last-event-id")
FORWARD_RESPONSE_HEADERS = ("content-type", "mcp-session-id", "cache-control")

JsonRpcId = str | int | None


class JsonRpcFault(Exception):
    def __init__(self, code: int, message: str, *, msg_id: JsonRpcId = None, http_status: int = 200, data: Any = None):
        super().__init__(message)
        self.code, self.message, self.msg_id, self.http_status, self.data = code, message, msg_id, http_status, data

    def response(self, headers: dict[str, str] | None = None) -> JSONResponse:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return JSONResponse({"jsonrpc": "2.0", "id": self.msg_id, "error": error}, status_code=self.http_status, headers=headers)


# --------------------------------------------------------------------------- parsing


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise JsonRpcFault(INVALID_REQUEST, f"Invalid Request: duplicate key {key!r}", http_status=400)
        obj[key] = value
    return obj


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-standard JSON constant {token}")


def parse_message(body: bytes) -> dict[str, Any]:
    """Parse one JSON-RPC 2.0 message strictly enough that no downstream parser can disagree with us."""
    try:
        msg = json.loads(body, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except JsonRpcFault:
        raise
    except (ValueError, UnicodeDecodeError):
        raise JsonRpcFault(PARSE_ERROR, "Parse error", http_status=400) from None

    if isinstance(msg, list):
        # MCP 2025-06-18+ removed JSON-RPC batching. Rejecting outright also closes the classic
        # bypass where a gateway inspects only the first element of a batch.
        raise JsonRpcFault(INVALID_REQUEST, "Invalid Request: batch requests are not supported", http_status=400)
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        raise JsonRpcFault(INVALID_REQUEST, "Invalid Request", http_status=400)

    msg_id = msg.get("id")
    if "id" in msg and not (isinstance(msg_id, (str, int)) and not isinstance(msg_id, bool)):
        raise JsonRpcFault(INVALID_REQUEST, "Invalid Request: id must be a string or integer", http_status=400)

    if "method" in msg:
        if not isinstance(msg["method"], str) or not msg["method"]:
            raise JsonRpcFault(INVALID_REQUEST, "Invalid Request: method must be a non-empty string", msg_id=msg_id, http_status=400)
        if "params" in msg and not isinstance(msg["params"], (dict, list)):
            raise JsonRpcFault(INVALID_REQUEST, "Invalid Request: params must be an object or array", msg_id=msg_id, http_status=400)
    elif not ("id" in msg and (("result" in msg) != ("error" in msg))):
        # Not a request/notification and not a well-formed response to a server-initiated request.
        raise JsonRpcFault(INVALID_REQUEST, "Invalid Request", msg_id=msg_id, http_status=400)
    return msg


# --------------------------------------------------------------------------- policy


def normalize_tool_name(name: str) -> str:
    """Canonical form for policy checks: NFKC, invisible format chars removed, trimmed, casefolded.

    Comparing the canonical form means look-alike spellings ("ADMIN_reset", "ａｄｍｉｎ_reset",
    "​admin_reset") are still treated as admin tools even if the downstream server
    resolves names more leniently than an exact match.
    """
    nfkc = unicodedata.normalize("NFKC", name)
    visible = "".join(ch for ch in nfkc if unicodedata.category(ch) != "Cf")
    return visible.strip().casefold()


def is_admin_tool(name: str) -> bool:
    return normalize_tool_name(name).startswith(ADMIN_TOOL_PREFIX)


def authorize(msg: dict[str, Any], principal: Principal) -> None:
    """Raise JsonRpcFault if ``principal`` may not send ``msg``. Fails closed on malformed tools/call."""
    if msg.get("method") != "tools/call":
        return  # tools/list, initialize, ping, notifications and responses pass through untouched

    msg_id = msg.get("id")
    if "id" not in msg:
        raise JsonRpcFault(INVALID_REQUEST, "Invalid Request: tools/call must carry an id", http_status=400)
    params = msg.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if not isinstance(name, str) or not name:
        raise JsonRpcFault(INVALID_PARAMS, "Invalid params: tools/call requires a string params.name", msg_id=msg_id)
    if is_admin_tool(name) and principal.role != "admin":
        raise JsonRpcFault(
            UNAUTHORIZED_TOOL_CALL,
            "Unauthorized Tool Call",
            msg_id=msg_id,
            data={"tool": name, "required_role": "admin", "role": principal.role},
        )


# --------------------------------------------------------------------------- proxy


@dataclass(frozen=True)
class GatewayConfig:
    downstream_url: str
    max_body_bytes: int = 1_048_576
    upstream_connect_timeout_s: float = 5.0
    upstream_read_timeout_s: float = 300.0  # long tool calls / SSE responses


def _authenticate(request: Request, verifier: JWTVerifier) -> Principal:
    try:
        return verifier.verify(extract_bearer_token(request.headers.get("authorization")))
    except AuthError as exc:
        raise JsonRpcFault(UNAUTHENTICATED, f"Unauthenticated: {exc}", http_status=401) from None


async def _read_bounded(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise JsonRpcFault(INVALID_REQUEST, "Request body too large", http_status=413)
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise JsonRpcFault(INVALID_REQUEST, "Request body too large", http_status=413)
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(
    config: GatewayConfig,
    verifier: JWTVerifier,
    http_client: httpx.AsyncClient | None = None,
) -> Starlette:
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(config.upstream_read_timeout_s, connect=config.upstream_connect_timeout_s)
    )
    www_auth = {"WWW-Authenticate": 'Bearer realm="mcp-gateway"'}

    async def forward(request: Request, principal: Principal, body: bytes | None, msg_id: JsonRpcId) -> Response:
        headers = {k: v for k in FORWARD_REQUEST_HEADERS if (v := request.headers.get(k)) is not None}
        headers["x-authenticated-subject"] = principal.subject
        headers["x-authenticated-role"] = principal.role
        upstream_req = client.build_request(request.method, config.downstream_url, headers=headers, content=body)
        try:
            upstream = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            logger.warning("downstream unavailable: %s", type(exc).__name__)
            raise JsonRpcFault(UPSTREAM_UNAVAILABLE, "Downstream MCP server unavailable", msg_id=msg_id, http_status=502) from None

        response_headers = {k: v for k in FORWARD_RESPONSE_HEADERS if (v := upstream.headers.get(k)) is not None}
        # Stream through so SSE responses and long tool results are relayed incrementally.
        return StreamingResponse(
            upstream.aiter_bytes(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream.aclose),
        )

    async def mcp_post(request: Request) -> Response:
        principal: Principal | None = None
        msg: dict[str, Any] | None = None
        try:
            principal = _authenticate(request, verifier)
            msg = parse_message(await _read_bounded(request, config.max_body_bytes))
            authorize(msg, principal)
        except JsonRpcFault as fault:
            _audit(request, principal, msg, f"rejected:{fault.code}")
            return fault.response(www_auth if fault.http_status == 401 else None)

        _audit(request, principal, msg, "forwarded")
        canonical = json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode()
        try:
            return await forward(request, principal, canonical, msg.get("id"))
        except JsonRpcFault as fault:
            return fault.response()

    async def mcp_stream_or_terminate(request: Request) -> Response:
        """GET opens the server->client SSE stream; DELETE ends the session. Authenticated, not inspected."""
        try:
            principal = _authenticate(request, verifier)
            return await forward(request, principal, None, None)
        except JsonRpcFault as fault:
            return fault.response(www_auth if fault.http_status == 401 else None)

    async def healthz(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        yield
        if http_client is None:  # only close a client we created
            await client.aclose()

    return Starlette(
        routes=[
            Route("/mcp", mcp_post, methods=["POST"]),
            Route("/mcp", mcp_stream_or_terminate, methods=["GET", "DELETE"]),
            Route("/healthz", healthz, methods=["GET"]),
        ],
        lifespan=lifespan,
    )


def _audit(request: Request, principal: Principal | None, msg: dict[str, Any] | None, decision: str) -> None:
    method = msg.get("method") if msg else None
    params = msg.get("params") if msg else None
    tool = params.get("name") if method == "tools/call" and isinstance(params, dict) else None
    logger.info(
        "audit sub=%s role=%s method=%s tool=%s decision=%s client=%s",
        principal.subject if principal else "-",
        principal.role if principal else "-",
        method,
        tool,
        decision,
        request.client.host if request.client else "-",
    )


def app_from_env() -> Starlette:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = GatewayConfig(downstream_url=os.environ.get("DOWNSTREAM_MCP_URL", "http://127.0.0.1:9001/mcp"))
    return create_app(config, verifier_from_env())
