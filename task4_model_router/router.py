"""Ordered model fallback with a hard per-attempt deadline.

For each provider in order (primary, then backups):
  * the whole attempt — connect, send, wait, read the full body — runs under one
    wall-clock deadline (``attempt_timeout_s``, 3000 ms by default). httpx's own
    timeouts are per-operation (a server dripping one byte every 2 s never trips a
    3 s read timeout), so they are set looser and the deadline is enforced with a
    cancel scope instead. On expiry the in-flight request is cancelled and its
    connection discarded; a late primary response can never be used or race the
    backup, because it no longer exists.
  * 429, timeouts, transport errors, 5xx and provider-side auth/config errors fail
    over to the next provider. Other 4xx mean the request itself is bad: the next
    provider would reject it too, so it is surfaced (sanitized) instead.
  * a 429 carrying Retry-After puts that provider in a short cooldown: subsequent
    requests try the backup first instead of paying a wasted round trip, and only
    fall back to the cooling provider if every other one fails.

Nothing from an upstream response other than a successful completion body leaves
this module: failures are reduced to an ``Attempt`` record for logs and metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx

from . import errors

logger = logging.getLogger("model_router")

FAILOVER_STATUSES = frozenset({401, 403, 404, 408, 409, 429, 500, 502, 503, 504})
MAX_COOLDOWN_S = 30.0


@dataclass(frozen=True)
class ProviderTarget:
    name: str
    client: httpx.AsyncClient  # base_url points at the provider
    model: str  # provider-specific model name substituted into the request
    path: str = "/v1/chat/completions"
    api_key: str | None = None


@dataclass(frozen=True)
class Attempt:
    provider: str
    outcome: str  # ok | timeout | transport_error | http_<status> | invalid_response
    elapsed_ms: float


@dataclass
class RoutedCompletion:
    provider: str
    model: str
    body: dict[str, Any]
    attempts: list[Attempt] = field(default_factory=list)


class _Failover(Exception):
    def __init__(self, outcome: str, retry_after_s: float | None = None):
        super().__init__(outcome)
        self.outcome, self.retry_after_s = outcome, retry_after_s


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None  # HTTP-date form: ignore rather than trust a clock we do not share


class FallbackRouter:
    def __init__(self, targets: list[ProviderTarget], *, attempt_timeout_s: float = 3.0, clock=time.monotonic):
        if not targets:
            raise ValueError("at least one provider target is required")
        self.targets = targets
        self.attempt_timeout_s = attempt_timeout_s
        self._clock = clock
        self._cooldown_until: dict[str, float] = {}

    def _cooling(self, target: ProviderTarget) -> bool:
        return self._cooldown_until.get(target.name, 0.0) > self._clock()

    async def complete(self, payload: dict[str, Any]) -> RoutedCompletion:
        attempts: list[Attempt] = []
        retry_hints: list[float] = []
        # Providers in cooldown are not skipped outright, only moved to the back as a last resort.
        cooling = [t for t in self.targets if self._cooling(t)]
        ordered = [t for t in self.targets if t not in cooling] + cooling

        for target in ordered:
            started = self._clock()
            try:
                body = await self._attempt(target, payload)
            except _Failover as failure:
                elapsed = (self._clock() - started) * 1000
                attempts.append(Attempt(target.name, failure.outcome, elapsed))
                if failure.retry_after_s is not None:
                    retry_hints.append(failure.retry_after_s)
                    if failure.outcome == "http_429":
                        self._cooldown_until[target.name] = self._clock() + min(failure.retry_after_s, MAX_COOLDOWN_S)
                logger.warning("provider %s failed (%s) after %.0fms; failing over", target.name, failure.outcome, elapsed)
                continue
            attempts.append(Attempt(target.name, "ok", (self._clock() - started) * 1000))
            return RoutedCompletion(target.name, target.model, body, attempts)

        logger.error("all providers failed: %s", [(a.provider, a.outcome) for a in attempts])
        raise errors.all_providers_failed(min(retry_hints) if retry_hints else None)

    async def _attempt(self, target: ProviderTarget, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"authorization": f"Bearer {target.api_key}"} if target.api_key else {}
        request_body = {**payload, "model": target.model, "stream": False}
        try:
            with anyio.fail_after(self.attempt_timeout_s):
                response = await target.client.post(target.path, json=request_body, headers=headers)
        except TimeoutError:
            raise _Failover("timeout") from None
        except httpx.TransportError as exc:
            raise _Failover("transport_error") from exc

        status = response.status_code
        if status in FAILOVER_STATUSES:
            raise _Failover(f"http_{status}", _parse_retry_after(response.headers.get("retry-after")))
        if status >= 400:
            logger.info("provider %s rejected request with %s", target.name, status)
            raise errors.upstream_rejected()
        try:
            body = response.json()
        except ValueError:
            raise _Failover("invalid_response") from None
        if not isinstance(body, dict) or not isinstance(body.get("choices"), list):
            raise _Failover("invalid_response")
        return body
