"""LLM gateway HTTP surface: tenant auth -> token rate limit -> fallback routing.

    POST /v1/chat/completions   Authorization: Bearer <tenant api key>

Success: the provider's completion body plus x-request-id, x-gateway-provider and
x-ratelimit-* headers. Failure: always the standardized payload from ``errors``.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal

import anyio
import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from . import errors
from .rate_limiter import RateLimitExceeded, RequestTooLarge, SqliteTokenRateLimiter, tenant_key
from .router import FallbackRouter, ProviderTarget
from .tokens import actual_tokens, reservation_tokens

logger = logging.getLogger("llm_gateway")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # temperature, top_p, tools, ... pass through
    model: str = "default"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=32_000)
    stream: bool = False


class RequestIdMiddleware:
    """Assigns every request an id before anything else runs, so even crashes can be correlated."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})["request_id"] = f"req_{secrets.token_hex(8)}"
        await self.app(scope, receive, send)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or f"req_{secrets.token_hex(8)}"


def create_app(
    *,
    limiter: SqliteTokenRateLimiter,
    router: FallbackRouter,
    api_keys: Mapping[str, str],  # api key -> tenant name
    purge_interval_s: float = 30.0,
) -> Starlette:
    tenants_by_hash = {tenant_key(key): name for key, name in api_keys.items()}

    def authenticate(request: Request) -> tuple[str, str]:
        scheme, _, key = (request.headers.get("authorization") or "").partition(" ")
        if scheme.lower() != "bearer" or not key.strip():
            raise errors.unauthenticated()
        hashed = tenant_key(key.strip())
        if hashed not in tenants_by_hash:  # dict lookup on a hash: no early-exit string compare on the secret
            raise errors.unauthenticated()
        return hashed, tenants_by_hash[hashed]

    async def chat_completions(request: Request) -> Response:
        request_id = _request_id(request)
        tenant, tenant_name = authenticate(request)
        try:
            parsed = ChatCompletionRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            fields = sorted({".".join(str(p) for p in e["loc"]) or "body" for e in exc.errors()})
            raise errors.invalid_request(f"Invalid request body (fields: {', '.join(fields)}).") from None
        if parsed.stream:
            raise errors.invalid_request("Streaming is not supported by this endpoint.", code="stream_not_supported")
        payload = parsed.model_dump(exclude_none=True)

        estimate = reservation_tokens(payload)
        try:
            reservation = await limiter.acquire(tenant, estimate)
        except RequestTooLarge as exc:
            raise errors.request_exceeds_limit(exc.requested, exc.limit) from None
        except RateLimitExceeded as exc:
            logger.info("rate limited tenant=%s used=%s requested=%s request_id=%s", tenant_name, exc.used, estimate, request_id)
            raise errors.tenant_rate_limited(exc.retry_after_s, exc.limit) from None

        try:
            result = await router.complete(payload)
        except BaseException:
            # No provider served it (or the client went away): give the reservation back.
            # Shielded so the release still runs while this task is being cancelled.
            with anyio.CancelScope(shield=True):
                await limiter.release(reservation)
            raise

        used = actual_tokens(result.body, payload)
        await limiter.reconcile(reservation, used)
        remaining = max(0, reservation.remaining + reservation.tokens - used)
        logger.info(
            "completed tenant=%s provider=%s reserved=%s used=%s attempts=%s request_id=%s",
            tenant_name, result.provider, reservation.tokens, used, [(a.provider, a.outcome) for a in result.attempts], request_id,
        )
        return JSONResponse(
            result.body,
            headers={
                "x-request-id": request_id,
                "x-gateway-provider": result.provider,
                "x-ratelimit-limit-tokens": str(reservation.limit),
                "x-ratelimit-remaining-tokens": str(remaining),
            },
        )

    async def healthz(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def on_gateway_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, errors.GatewayError)
        request_id = _request_id(request)
        logger.info("request failed status=%s code=%s request_id=%s", exc.status, exc.code, request_id)
        return exc.to_response(request_id)

    async def on_http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, HTTPException)
        mapped = {404: ("not_found", "Not found."), 405: ("method_not_allowed", "Method not allowed.")}
        code, message = mapped.get(exc.status_code, ("http_error", "Request could not be processed."))
        return errors.GatewayError(exc.status_code, "invalid_request_error", code, message).to_response(_request_id(request))

    async def on_unhandled(request: Request, exc: Exception) -> Response:
        request_id = _request_id(request)
        logger.exception("unhandled error request_id=%s", request_id)  # full trace stays server-side
        return errors.internal_error().to_response(request_id)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with anyio.create_task_group() as tg:
            async def purge_loop() -> None:
                while True:
                    await anyio.sleep(purge_interval_s)
                    try:
                        await limiter.purge_expired()
                    except Exception:
                        logger.exception("rate-limit purge failed")

            tg.start_soon(purge_loop)
            yield
            tg.cancel_scope.cancel()

    app = Starlette(
        routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"]), Route("/healthz", healthz)],
        exception_handlers={errors.GatewayError: on_gateway_error, HTTPException: on_http_error, Exception: on_unhandled},
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    return app


def app_from_env() -> Starlette:
    """Env: RATE_LIMIT_DB, TENANT_API_KEYS ("key1:tenantA,key2:tenantB"), PRIMARY_URL/MODEL/API_KEY, SECONDARY_URL/MODEL/API_KEY."""
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw_keys = os.environ.get("TENANT_API_KEYS", "sk-tenant-a:tenant-a,sk-tenant-b:tenant-b")
    api_keys = dict(pair.split(":", 1) for pair in raw_keys.split(",") if ":" in pair)

    # httpx timeouts deliberately exceed the router's 3 s deadline, which is the one that governs.
    def client(url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(10.0, connect=2.0))

    targets = [
        ProviderTarget("primary", client(os.environ.get("PRIMARY_URL", "http://127.0.0.1:9101")),
                       os.environ.get("PRIMARY_MODEL", "primary-large"), api_key=os.environ.get("PRIMARY_API_KEY")),
        ProviderTarget("secondary", client(os.environ.get("SECONDARY_URL", "http://127.0.0.1:9102")),
                       os.environ.get("SECONDARY_MODEL", "backup-medium"), api_key=os.environ.get("SECONDARY_API_KEY")),
    ]
    limiter = SqliteTokenRateLimiter(
        os.environ.get("RATE_LIMIT_DB", "gateway_ratelimit.sqlite3"),
        tokens_per_window=int(os.environ.get("TOKENS_PER_MINUTE", "50000")),
    )
    return create_app(limiter=limiter, router=FallbackRouter(targets, attempt_timeout_s=float(os.environ.get("ATTEMPT_TIMEOUT_S", "3.0"))), api_keys=api_keys)
