"""Task 3: streaming PII redaction — correctness across chunk boundaries, bounded holdback, low latency."""

from __future__ import annotations

import itertools
import json
import random

import anyio
import httpx
import pytest

from task3_pii_guardrail.gateway import ProviderConfig, SseParser, StreamGuardrail, create_app, guarded_stream
from task3_pii_guardrail.mock_provider import create_mock_provider
from task3_pii_guardrail.redactor import DEFAULT_MATCHER, StreamingRedactor, luhn_valid, redact_text

PII = ["jane.roe@example.com", "a.b-c+tag@mail.co.uk", "123-45-6789", "123 45 6789", "4111 1111 1111 1111", "4111111111111111", "3782 822463 10005"]
SAMPLE = (
    "Contact jane.roe@example.com (alt a.b-c+tag@mail.co.uk). SSN 123-45-6789 / 123 45 6789. "
    "Cards: 4111 1111 1111 1111, 4111111111111111, 3782 822463 10005. Thanks!"
)


def stream_redact(text: str, sizes) -> str:
    r = StreamingRedactor()
    out, i = [], 0
    for n in sizes:
        out.append(r.feed(text[i : i + n]))
        i += n
    out.append(r.feed(text[i:]))
    out.append(r.flush())
    return "".join(out)


# --------------------------------------------------------------------------- patterns


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail jane.roe@example.com now", "mail [REDACTED] now"),
        ("<x@y.io>", "<[REDACTED]>"),
        ("SSN: 123-45-6789.", "SSN: [REDACTED]."),
        ("ssn 123 45 6789", "ssn [REDACTED]"),
        ("card 4111-1111-1111-1111!", "card [REDACTED]!"),
        ("amex 378282246310005", "amex [REDACTED]"),
        ("two: a@b.co,c@d.co", "two: [REDACTED],[REDACTED]"),
        # Negatives: not PII, must be left intact.
        ("order 4111 1111 1111 1112", "order 4111 1111 1111 1112"),  # fails Luhn
        ("id 000-12-3456 or 666-12-3456 or 900-12-3456", "id 000-12-3456 or 666-12-3456 or 900-12-3456"),
        ("ref 123-45-67890", "ref 123-45-67890"),  # too many digits
        ("mixed 123-45 6789", "mixed 123-45 6789"),  # inconsistent separators
        ("user@localhost and @handle", "user@localhost and @handle"),
        ("version 1.2.3 at 10:45", "version 1.2.3 at 10:45"),
    ],
)
def test_redact_text(text, expected):
    assert redact_text(text) == expected


def test_luhn_rejection_does_not_hide_overlapping_ssn():
    assert redact_text("x 4111 1111 1111 1112 123-45-6789 y") == "x 4111 1111 1111 1112 [REDACTED] y"


def test_luhn():
    assert luhn_valid("4111 1111 1111 1111") and luhn_valid("378282246310005")
    assert not luhn_valid("4111 1111 1111 1112") and not luhn_valid("1234")


# --------------------------------------------------------------------------- streaming invariants


def test_every_two_way_split_matches_one_shot_and_never_leaks():
    expected = redact_text(SAMPLE)
    for cut in range(len(SAMPLE) + 1):
        got = stream_redact(SAMPLE, [cut])
        assert got == expected, f"split at {cut}"


def test_every_three_way_split_of_each_pii_value():
    for value in PII:
        text = f"x {value} y"
        expected = redact_text(text)
        assert value not in expected
        for a, b in itertools.combinations(range(len(text) + 1), 2):
            got = stream_redact(text, [a, b - a])
            assert got == expected, (value, a, b)


def test_random_chunkings_including_single_characters():
    rng = random.Random(1234)
    expected = redact_text(SAMPLE)
    assert stream_redact(SAMPLE, [1] * len(SAMPLE)) == expected
    for _ in range(500):
        sizes = [rng.randint(1, 12) for _ in range(len(SAMPLE))]
        assert stream_redact(SAMPLE, sizes) == expected


FUZZ_FRAGMENTS = ["a@b.co", ".uk", "@", "x.y-z+q@mail.example.org", "123-45-6789", "123 45 6789", "4111 1111 1111 1111",
                  "4111111111111111", "3782 822463 10005", " ", ".", "-", "0", "1234", "abc", "\n", ",", "+", "co.", "666-12-3456", "é"]


def test_fuzz_streaming_equals_one_shot():
    """Adversarial concatenations (PII glued to PII, near-misses, separators) under random chunking."""
    rng = random.Random(2026)
    for _ in range(3000):
        text = "".join(rng.choice(FUZZ_FRAGMENTS) for _ in range(rng.randint(1, 14)))
        assert stream_redact(text, [rng.randint(1, 6) for _ in range(len(text))]) == redact_text(text), text


@pytest.mark.parametrize(
    "text",
    [
        "co.123-45-6789a@b.co",  # complete SSN inside a longer email: the email must win, as in one-shot
        "a.b-c+tag@mail.co" + ".uk",  # complete-looking email that keeps growing
        "x 123-45-6789.y@z.io",
    ],
)
def test_regressions_for_matches_that_keep_growing(text):
    expected = redact_text(text)
    for cut in range(len(text) + 1):
        assert stream_redact(text, [cut]) == expected


def test_safe_text_is_released_without_waiting_for_end_of_stream():
    r = StreamingRedactor()
    assert r.feed("Hello") == ""  # might be the local part of an email
    assert r.feed(" world") == "Hello "  # boundary seen: released immediately
    assert r.feed(". The") == "world. "
    assert r.feed(" answer is 42") == "The answer is "  # "42" may still grow into a card/SSN
    assert r.feed(".\n") == "42.\n"
    assert r.held == 0


def test_partial_pii_is_never_emitted():
    r = StreamingRedactor()
    emitted = "".join(r.feed(t) for t in ["Card ", "4111", " 1111", " 1111", " 111", "1 ok"])
    assert "4111" not in emitted and "1111" not in emitted
    assert emitted + r.flush() == "Card [REDACTED] ok"


def test_holdback_is_bounded_on_adversarial_input():
    rng = random.Random(0)
    r = StreamingRedactor()
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789.-_+@"
    peak = 0
    for _ in range(20_000):
        r.feed("".join(rng.choice(alphabet) for _ in range(5)))
        peak = max(peak, r.held)
    assert peak <= DEFAULT_MATCHER.max_match_len


def test_flush_at_end_redacts_trailing_pii():
    r = StreamingRedactor()
    assert r.feed("reach me at bob@corp.io") == "reach me at "
    assert r.flush() == "[REDACTED]"
    assert r.counts["EMAIL"] == 1


# --------------------------------------------------------------------------- SSE framing


def sse_chunk(content=None, *, index=0, finish=None, role=None) -> dict:
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": index, "delta": delta, "finish_reason": finish}]}


def sse_bytes(chunks: list[dict], done: bool = True, newline: str = "\n") -> bytes:
    body = "".join(f"data: {json.dumps(c, ensure_ascii=False)}{newline}{newline}" for c in chunks)
    if done:
        body += f"data: [DONE]{newline}{newline}"
    return body.encode()


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_sse_parser_handles_any_byte_split(newline):
    raw = (f": keep-alive{newline}event: message{newline}data: {{\"a\":\"héllo €\"}}{newline}{newline}"
           f"data: line1{newline}data: line2{newline}{newline}").encode()
    for cut in range(len(raw) + 1):
        p = SseParser()
        events = [*p.feed(raw[:cut]), *p.feed(raw[cut:]), *p.close()]
        assert [(e.event, e.data) for e in events] == [("message", '{"a":"héllo €"}'), (None, "line1\nline2")], cut


def collect_content(raw: bytes) -> dict[int, str]:
    text: dict[int, str] = {}
    p = SseParser()
    for event in [*p.feed(raw), *p.close()]:
        if event.data == "[DONE]":
            continue
        for choice in json.loads(event.data)["choices"]:
            text[choice["index"]] = text.get(choice["index"], "") + (choice["delta"].get("content") or "")
    return text


async def run_guarded(raw: bytes, split: int) -> bytes:
    async def source():
        for i in range(0, len(raw), split):
            yield raw[i : i + split]

    return b"".join([out async for out in guarded_stream(source(), StreamGuardrail())])


@pytest.mark.anyio
@pytest.mark.parametrize("split", [1, 3, 7, 64, 10_000])
async def test_guarded_stream_redacts_regardless_of_network_framing(split):
    pieces = ["Email ", "jane.r", "oe@exa", "mple.c", "om", ", SSN 12", "3-45-", "6789", ". Done"]
    raw = sse_bytes([sse_chunk("", role="assistant"), *[sse_chunk(p) for p in pieces], sse_chunk(finish="stop")])
    out = await run_guarded(raw, split)
    assert collect_content(out) == {0: "Email [REDACTED], SSN [REDACTED]. Done"}
    assert out.rstrip().endswith(b"data: [DONE]")
    finish = [json.loads(e.data) for e in SseParser().feed(out) if e.data != "[DONE]"][-1]
    assert finish["choices"][0]["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_multiple_choices_are_redacted_independently():
    chunks = [sse_chunk("mail a@", index=0), sse_chunk("SSN 123-", index=1), sse_chunk("b.io ok", index=0),
              sse_chunk("45-6789 ok", index=1), sse_chunk(finish="stop", index=0), sse_chunk(finish="stop", index=1)]
    out = await run_guarded(sse_bytes(chunks), 5)
    assert collect_content(out) == {0: "mail [REDACTED] ok", 1: "SSN [REDACTED] ok"}


@pytest.mark.anyio
async def test_upstream_ending_without_done_still_flushes_redacted_tail():
    out = await run_guarded(sse_bytes([sse_chunk("ping x@y.io")], done=False), 4)
    assert collect_content(out) == {0: "ping [REDACTED]"}


@pytest.mark.anyio
async def test_output_is_incremental_not_buffered():
    """Each safe delta must reach the client before the provider sends the next one."""
    send, receive = anyio.create_memory_object_stream[bytes](8)
    guard = StreamGuardrail()

    async def source():
        async with receive:
            async for item in receive:
                yield item

    gen = guarded_stream(source(), guard)
    async with send:
        await send.send(sse_bytes([sse_chunk("Hello there. ")], done=False))
        with anyio.fail_after(1):
            first = await gen.__anext__()
        assert collect_content(first) == {0: "Hello there. "}
        await send.send(sse_bytes([sse_chunk("Bye now. ")], done=False))
        with anyio.fail_after(1):
            second = await gen.__anext__()
        assert collect_content(second) == {0: "Bye now. "}
    await gen.aclose()


# --------------------------------------------------------------------------- HTTP app


def gateway_client(provider_app) -> httpx.AsyncClient:
    upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=provider_app), base_url="http://provider")
    app = create_app(ProviderConfig(base_url="http://provider"), http_client=upstream)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")


@pytest.mark.anyio
async def test_gateway_streaming_end_to_end():
    message = "Customer bob.smith@corp.example.org, SSN 078-05-1120, card 5555 5555 5555 4444. Anything else?"
    async with gateway_client(create_mock_provider(delay_s=0, seed=3)) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "m", "stream": True, "messages": [{"role": "user", "content": message}]})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert collect_content(resp.content) == {0: "Customer [REDACTED], SSN [REDACTED], card [REDACTED]. Anything else?"}


@pytest.mark.anyio
async def test_gateway_non_streaming():
    async with gateway_client(create_mock_provider(delay_s=0)) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "ssn 123-45-6789"}]})
    assert resp.json()["choices"][0]["message"]["content"] == "ssn [REDACTED]"


@pytest.mark.anyio
async def test_gateway_hides_provider_errors():
    def failing(_request):
        return httpx.Response(500, json={"error": "Traceback (most recent call last): secret internals"})

    app = create_app(ProviderConfig(base_url="http://p"), http_client=httpx.AsyncClient(transport=httpx.MockTransport(failing), base_url="http://p"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://g") as client:
        resp = await client.post("/v1/chat/completions", json={"stream": True, "messages": []})
    assert resp.status_code == 502
    assert "Traceback" not in resp.text


@pytest.mark.anyio
async def test_client_disconnect_closes_upstream_stream():
    state = {"closed": False, "sent": 0}

    async def endless():
        try:
            while True:
                state["sent"] += 1
                yield sse_bytes([sse_chunk("token ")], done=False)
                await anyio.sleep(0.005)
        finally:
            state["closed"] = True

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=endless())), base_url="http://p")
    app = create_app(ProviderConfig(base_url="http://p"), http_client=upstream)

    body = json.dumps({"stream": True, "messages": []}).encode()
    first_chunk = anyio.Event()
    messages = iter([{"type": "http.request", "body": body, "more_body": False}])

    async def receive():
        try:
            return next(messages)
        except StopIteration:
            await first_chunk.wait()
            return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk.set()

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": [(b"content-type", b"application/json")],
             "query_string": b"", "http_version": "1.1", "scheme": "http", "server": ("gw", 80), "client": ("c", 1), "root_path": ""}
    with anyio.fail_after(5):
        await app(scope, receive, send)
    with anyio.fail_after(2):
        while not state["closed"]:
            await anyio.sleep(0.01)
    assert state["sent"] < 100
