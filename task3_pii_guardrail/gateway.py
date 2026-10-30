"""LLM gateway endpoint with a real-time PII redaction guardrail on the response stream.

    client ──POST /v1/chat/completions──▶ gateway ──▶ provider (OpenAI-compatible, stream=true)
           ◀── SSE, redacted deltas ────────┘◀── SSE deltas ──┘

The response is never buffered: provider bytes are decoded incrementally, split into
SSE events, each ``choices[i].delta.content`` goes through that choice's
``StreamingRedactor``, and the rewritten event is yielded immediately. Per-stream
memory is one partial SSE event plus each redactor's bounded holdback. When the
client disconnects, the generator is cancelled and the upstream connection closed,
so the provider stops generating (and billing) tokens nobody will read.
"""

from __future__ import annotations

import codecs
import json
import logging
import os
import time
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .redactor import DEFAULT_MATCHER, PiiMatcher, StreamingRedactor, redact_text

logger = logging.getLogger("pii_gateway")


# --------------------------------------------------------------------------- SSE framing


@dataclass
class SseEvent:
    data: str
    event: str | None = None

    def encode(self) -> bytes:
        head = f"event: {self.event}\n" if self.event else ""
        body = "".join(f"data: {line}\n" for line in self.data.split("\n"))
        return f"{head}{body}\n".encode()


class SseParser:
    """Incremental text/event-stream parser (WHATWG rules: CRLF/CR/LF, multi-line data, comments).

    Bytes may be split anywhere — mid-line, mid-event or mid UTF-8 code point.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buf = ""
        self._data: list[str] = []
        self._event: str | None = None

    def feed(self, chunk: bytes) -> Iterator[SseEvent]:
        self._buf += self._decoder.decode(chunk)
        yield from self._drain()

    def close(self) -> Iterator[SseEvent]:
        self._buf += self._decoder.decode(b"", final=True)
        if self._buf:
            self._buf += "\n"
        yield from self._drain()
        if self._data:
            yield self._dispatch()

    def _drain(self) -> Iterator[SseEvent]:
        while True:
            idx = min((i for i in (self._buf.find("\r"), self._buf.find("\n")) if i != -1), default=-1)
            if idx == -1 or (self._buf[idx] == "\r" and idx == len(self._buf) - 1):
                return  # no complete line yet (a trailing CR may be the first half of CRLF)
            line = self._buf[:idx]
            skip = 2 if self._buf.startswith("\r\n", idx) else 1
            self._buf = self._buf[idx + skip :]
            if line == "":
                if self._data:
                    yield self._dispatch()
                self._event = None
                continue
            if line.startswith(":"):
                continue  # comment / keep-alive
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "data":
                self._data.append(value)
            elif field == "event":
                self._event = value

    def _dispatch(self) -> SseEvent:
        event = SseEvent("\n".join(self._data), self._event)
        self._data, self._event = [], None
        return event


# --------------------------------------------------------------------------- guardrail


class StreamGuardrail:
    """Rewrites OpenAI-style chat.completion.chunk events, one redactor per choice index."""

    def __init__(self, matcher: PiiMatcher = DEFAULT_MATCHER):
        self._matcher = matcher
        self._redactors: dict[int, StreamingRedactor] = {}
        self._template: dict[str, Any] | None = None

    def _redactor(self, index: int) -> StreamingRedactor:
        if index not in self._redactors:
            self._redactors[index] = StreamingRedactor(self._matcher)
        return self._redactors[index]

    @property
    def counts(self) -> Counter[str]:
        total: Counter[str] = Counter()
        for r in self._redactors.values():
            total.update(r.counts)
        return total

    def process(self, event: SseEvent) -> list[SseEvent]:
        if event.data.strip() == "[DONE]":
            return [*self._flush_all(), event]
        try:
            chunk = json.loads(event.data)
        except ValueError:
            return [SseEvent(redact_text(event.data, self._matcher), event.event)]  # unknown payload: fail safe
        if not isinstance(chunk, dict) or not isinstance(chunk.get("choices"), list):
            return [event]  # e.g. provider error object; no completion text to inspect

        self._template = {k: chunk[k] for k in ("id", "object", "created", "model") if k in chunk}
        meaningful = bool(chunk.get("usage")) or not chunk["choices"]
        for choice in chunk["choices"]:
            if not isinstance(choice, dict):
                continue
            redactor = self._redactor(choice.get("index", 0))
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            text = delta.get("content")
            safe = redactor.feed(text) if isinstance(text, str) else ""
            if choice.get("finish_reason") is not None:
                safe += redactor.flush()  # the choice is over: release (redacted) holdback before finishing
                meaningful = True
            if isinstance(text, str) or safe:
                delta["content"] = safe
                choice["delta"] = delta
            if safe or any(k != "content" for k in delta):
                meaningful = True
        if not meaningful:
            return []  # everything in this chunk is held back; nothing to send yet
        return [SseEvent(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")), event.event)]

    def _flush_all(self) -> list[SseEvent]:
        out = []
        for index, redactor in self._redactors.items():
            tail = redactor.flush()
            if tail:
                chunk = {**(self._template or {"object": "chat.completion.chunk"}),
                         "choices": [{"index": index, "delta": {"content": tail}, "finish_reason": None}]}
                out.append(SseEvent(json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))))
        return out

    def finish(self) -> list[SseEvent]:
        """Upstream ended without [DONE]: still release whatever is held, redacted."""
        return self._flush_all()


async def guarded_stream(upstream_bytes: AsyncIterator[bytes], guardrail: StreamGuardrail) -> AsyncIterator[bytes]:
    parser = SseParser()
    async for raw in upstream_bytes:
        for event in parser.feed(raw):
            for out in guardrail.process(event):
                yield out.encode()
    for event in parser.close():
        for out in guardrail.process(event):
            yield out.encode()
    for out in guardrail.finish():
        yield out.encode()


# --------------------------------------------------------------------------- HTTP app


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key: str | None = None
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 60.0  # max silence between chunks, not total duration


def _error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status)


def create_app(provider: ProviderConfig, http_client: httpx.AsyncClient | None = None, matcher: PiiMatcher = DEFAULT_MATCHER) -> Starlette:
    client = http_client or httpx.AsyncClient(
        base_url=provider.base_url,
        timeout=httpx.Timeout(provider.read_timeout_s, connect=provider.connect_timeout_s),
    )
    upstream_headers = {"content-type": "application/json"}
    if provider.api_key:
        upstream_headers["authorization"] = f"Bearer {provider.api_key}"

    async def chat_completions(request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "invalid_request_error", "Request body must be valid JSON.")
        if not isinstance(body, dict):
            return _error(400, "invalid_request_error", "Request body must be a JSON object.")

        upstream_req = client.build_request("POST", "/v1/chat/completions", json=body, headers=upstream_headers)
        try:
            upstream = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            logger.warning("provider unreachable: %s", type(exc).__name__)
            return _error(502, "upstream_error", "The model provider is unavailable.")

        if upstream.status_code != 200:
            await upstream.aclose()
            logger.warning("provider returned %s", upstream.status_code)
            return _error(502 if upstream.status_code >= 500 else upstream.status_code, "upstream_error",
                          f"The model provider rejected the request (status {upstream.status_code}).")

        if not body.get("stream"):
            try:
                data = json.loads(await upstream.aread())
            finally:
                await upstream.aclose()
            for choice in data.get("choices", []):
                message = choice.get("message") or {}
                if isinstance(message.get("content"), str):
                    message["content"] = redact_text(message["content"], matcher)
            return JSONResponse(data)

        guardrail = StreamGuardrail(matcher)
        started = time.perf_counter()

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                async for out in guarded_stream(upstream.aiter_bytes(), guardrail):
                    yield out
            finally:
                await upstream.aclose()  # also runs on client disconnect (generator cancelled)
                logger.info("stream done in %.0fms redactions=%s", (time.perf_counter() - started) * 1000, dict(guardrail.counts))

        return StreamingResponse(
            body_iter(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},  # stop intermediaries buffering SSE
        )

    async def healthz(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        yield
        if http_client is None:
            await client.aclose()

    return Starlette(
        routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"]), Route("/healthz", healthz)],
        lifespan=lifespan,
    )


def app_from_env() -> Starlette:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app(ProviderConfig(
        base_url=os.environ.get("LLM_PROVIDER_BASE_URL", "http://127.0.0.1:9002"),
        api_key=os.environ.get("LLM_PROVIDER_API_KEY"),
    ))
