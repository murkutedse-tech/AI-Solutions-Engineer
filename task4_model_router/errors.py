"""Standardized gateway error payload.

Every failure leaves the gateway in exactly this shape, whatever happened upstream:

    {"error": {"type": "rate_limit_error", "code": "tenant_token_limit_exceeded",
               "message": "...", "request_id": "req_...", "retry_after_s": 12}}

Upstream bodies, exception messages and stack traces are never included; they go to
server logs, keyed by ``request_id`` so support can correlate a client report with them.
"""

from __future__ import annotations

import math
from typing import Any

from starlette.responses import JSONResponse


class GatewayError(Exception):
    def __init__(self, status: int, type_: str, code: str, message: str, *, retry_after_s: float | None = None):
        super().__init__(message)
        self.status, self.type, self.code, self.message, self.retry_after_s = status, type_, code, message, retry_after_s

    def to_response(self, request_id: str) -> JSONResponse:
        body: dict[str, Any] = {"type": self.type, "code": self.code, "message": self.message, "request_id": request_id}
        headers = {"x-request-id": request_id}
        if self.retry_after_s is not None:
            seconds = max(1, math.ceil(self.retry_after_s))
            body["retry_after_s"] = seconds
            headers["retry-after"] = str(seconds)
        return JSONResponse({"error": body}, status_code=self.status, headers=headers)


def unauthenticated() -> GatewayError:
    return GatewayError(401, "authentication_error", "invalid_api_key", "Missing or invalid API key.")


def invalid_request(message: str, code: str = "invalid_request") -> GatewayError:
    return GatewayError(400, "invalid_request_error", code, message)


def tenant_rate_limited(retry_after_s: float, limit: int) -> GatewayError:
    return GatewayError(
        429, "rate_limit_error", "tenant_token_limit_exceeded",
        f"Token rate limit of {limit} tokens per minute exceeded for this API key.",
        retry_after_s=retry_after_s,
    )


def request_exceeds_limit(estimated: int, limit: int) -> GatewayError:
    return invalid_request(
        f"Request needs an estimated {estimated} tokens, more than the {limit} tokens/minute limit; reduce the prompt or max_tokens.",
        code="request_exceeds_token_limit",
    )


def upstream_rejected() -> GatewayError:
    return invalid_request("The model provider rejected the request.", code="upstream_rejected_request")


def all_providers_failed(retry_after_s: float | None) -> GatewayError:
    return GatewayError(
        503, "upstream_error", "all_providers_unavailable",
        "No model provider could serve the request. Please retry later.",
        retry_after_s=retry_after_s,
    )


def internal_error() -> GatewayError:
    return GatewayError(500, "internal_error", "internal_error", "An internal error occurred.")
