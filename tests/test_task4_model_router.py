"""Task 4: SQLite sliding-window token limiter, timeout/429 failover and sanitized gateway errors."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import anyio
import httpx
import pytest

from task4_model_router.app import create_app
from task4_model_router.rate_limiter import RateLimitExceeded, RequestTooLarge, SqliteTokenRateLimiter, tenant_key
from task4_model_router.router import FallbackRouter, ProviderTarget
from task4_model_router.tokens import reservation_tokens

ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def limiter(tmp_path, clock):
    lim = SqliteTokenRateLimiter(tmp_path / "rl.sqlite3", tokens_per_window=50_000, window_s=60, clock=clock)
    yield lim
    lim.close()


def rows(db: Path) -> list[tuple]:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT tenant, ts, tokens FROM token_usage ORDER BY id").fetchall()


# --------------------------------------------------------------------------- rate limiter


@pytest.mark.anyio
async def test_admits_up_to_limit_then_rejects_with_retry_after(limiter, clock):
    await limiter.acquire("t1", 30_000)
    clock.advance(10)
    await limiter.acquire("t1", 20_000)  # exactly at the limit is allowed
    clock.advance(5)
    with pytest.raises(RateLimitExceeded) as exc:
        await limiter.acquire("t1", 1)
    # The oldest 30k entry (t=0) leaves the window at t=60; we are at t=15.
    assert exc.value.retry_after_s == pytest.approx(45)
    assert exc.value.used == 50_000


@pytest.mark.anyio
async def test_retry_after_waits_for_enough_tokens_to_expire(limiter, clock):
    for _ in range(5):
        await limiter.acquire("t1", 10_000)
        clock.advance(10)  # entries at t=0,10,20,30,40; now t=50
    with pytest.raises(RateLimitExceeded) as exc:
        await limiter.acquire("t1", 25_000)  # needs the t=0, t=10 and t=20 entries gone
    assert exc.value.retry_after_s == pytest.approx(20 + 60 - 50)


@pytest.mark.anyio
async def test_window_slides_and_expired_rows_are_evicted(limiter, clock, tmp_path):
    await limiter.acquire("t1", 40_000)
    clock.advance(59.999)
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire("t1", 20_000)
    clock.advance(0.001)  # exactly window_s later: the first entry has expired
    await limiter.acquire("t1", 20_000)
    assert [r[2] for r in rows(tmp_path / "rl.sqlite3")] == [20_000]  # old row physically deleted
    assert await limiter.usage("t1") == 20_000


@pytest.mark.anyio
async def test_no_fixed_window_boundary_burst(limiter, clock):
    clock.advance(59)
    await limiter.acquire("t1", 50_000)
    clock.advance(2)  # a fixed per-minute window would reset here and admit another 50k
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire("t1", 1)


@pytest.mark.anyio
async def test_tenants_are_isolated(limiter):
    await limiter.acquire("a", 50_000)
    await limiter.acquire("b", 50_000)
    with pytest.raises(RateLimitExceeded):
        await limiter.acquire("a", 1)


@pytest.mark.anyio
async def test_reconcile_and_release_adjust_usage(limiter):
    r1 = await limiter.acquire("t1", 5_000)
    await limiter.reconcile(r1, 1_234)
    assert await limiter.usage("t1") == 1_234
    r2 = await limiter.acquire("t1", 2_000)
    await limiter.release(r2)
    assert await limiter.usage("t1") == 1_234


@pytest.mark.anyio
async def test_single_request_larger_than_limit(limiter):
    with pytest.raises(RequestTooLarge):
        await limiter.acquire("t1", 50_001)


@pytest.mark.anyio
async def test_state_persists_on_disk_across_instances(tmp_path, clock):
    db = tmp_path / "persist.sqlite3"
    first = SqliteTokenRateLimiter(db, clock=clock)
    await first.acquire("t1", 45_000)
    first.close()
    second = SqliteTokenRateLimiter(db, clock=clock)
    with pytest.raises(RateLimitExceeded):
        await second.acquire("t1", 10_000)
    second.close()


@pytest.mark.anyio
async def test_purge_expired_removes_idle_tenants(limiter, clock, tmp_path):
    await limiter.acquire("idle", 100)
    clock.advance(61)
    assert await limiter.purge_expired() == 1
    assert rows(tmp_path / "rl.sqlite3") == []


@pytest.mark.anyio
async def test_concurrent_acquires_never_oversubscribe(limiter):
    admitted, rejected = [], []

    async def attempt():
        try:
            admitted.append(await limiter.acquire("t1", 2_000))
        except RateLimitExceeded:
            rejected.append(1)

    async with anyio.create_task_group() as tg:
        for _ in range(60):
            tg.start_soon(attempt)
    assert len(admitted) == 25 and len(rejected) == 35
    assert await limiter.usage("t1") == 50_000


def test_multiple_processes_share_one_budget(tmp_path):
    """Several gateway processes on one SQLite file must still enforce a single limit."""
    db = tmp_path / "shared.sqlite3"
    SqliteTokenRateLimiter(db).close()
    worker = (
        "import sys\n"
        "from task4_model_router.rate_limiter import SqliteTokenRateLimiter, RateLimitExceeded\n"
        "lim = SqliteTokenRateLimiter(sys.argv[1], tokens_per_window=50_000)\n"
        "ok = 0\n"
        "for _ in range(40):\n"
        "    try:\n"
        "        lim._acquire_sync('shared', 1_000); ok += 1\n"
        "    except RateLimitExceeded:\n"
        "        pass\n"
        "print(ok)\n"
    )
    procs = [subprocess.Popen([sys.executable, "-c", worker, str(db)], cwd=ROOT, stdout=subprocess.PIPE, text=True) for _ in range(4)]
    admitted = sum(int(p.communicate(timeout=60)[0]) for p in procs)
    assert admitted == 50  # 160 attempts x 1k tokens against a 50k budget


# --------------------------------------------------------------------------- router


def provider(name: str, handler) -> ProviderTarget:
    return ProviderTarget(name, httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=f"http://{name}"), model=f"{name}-model")


def ok_handler(name: str, calls: list | None = None):
    async def handle(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": f"from {name}"}}],
                                         "usage": {"total_tokens": 42}})
    return handle


def status_handler(status: int, body: str = "Traceback (most recent call last): internal detail", headers=None):
    async def handle(_request):
        return httpx.Response(status, text=body, headers=headers)
    return handle


PAYLOAD = {"model": "default", "messages": [{"role": "user", "content": "hi"}]}


def test_default_attempt_timeout_is_3000ms():
    assert FallbackRouter([provider("p", ok_handler("p"))]).attempt_timeout_s == 3.0


@pytest.mark.anyio
async def test_primary_success_uses_primary_with_its_model():
    calls: list = []
    router = FallbackRouter([provider("primary", ok_handler("primary", calls)), provider("secondary", ok_handler("secondary"))])
    result = await router.complete(PAYLOAD)
    assert result.provider == "primary"
    assert calls[0]["model"] == "primary-model" and calls[0]["stream"] is False


@pytest.mark.anyio
async def test_429_fails_over_to_secondary():
    router = FallbackRouter([provider("primary", status_handler(429)), provider("secondary", ok_handler("secondary"))])
    result = await router.complete(PAYLOAD)
    assert result.provider == "secondary"
    assert [(a.provider, a.outcome) for a in result.attempts] == [("primary", "http_429"), ("secondary", "ok")]


@pytest.mark.anyio
async def test_timeout_fails_over_and_cancels_the_slow_request():
    primary_state = {"finished": False, "cancelled": False}

    async def hang(_request):
        try:
            await anyio.sleep(10)
            primary_state["finished"] = True
            return httpx.Response(200, json={"choices": [{"message": {"content": "late"}}]})
        except BaseException:
            primary_state["cancelled"] = True
            raise

    router = FallbackRouter([provider("primary", hang), provider("secondary", ok_handler("secondary"))], attempt_timeout_s=0.3)
    started = time.monotonic()
    result = await router.complete(PAYLOAD)
    elapsed = time.monotonic() - started
    assert result.provider == "secondary"
    assert 0.3 <= elapsed < 1.0
    assert primary_state == {"finished": False, "cancelled": True}  # the late response can never be used
    assert result.attempts[0].outcome == "timeout"


@pytest.mark.anyio
async def test_deadline_covers_slow_drip_body_not_just_headers():
    async def drip(_request):
        async def body():
            for _ in range(20):
                yield b" "
                await anyio.sleep(0.1)  # every individual read is fast; the total is 2 s
        return httpx.Response(200, content=body())

    router = FallbackRouter([provider("primary", drip), provider("secondary", ok_handler("secondary"))], attempt_timeout_s=0.3)
    started = time.monotonic()
    assert (await router.complete(PAYLOAD)).provider == "secondary"
    assert time.monotonic() - started < 1.0


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["connect", 500, 503, "garbage"])
async def test_other_upstream_failures_fail_over(failure):
    async def handler(request):
        if failure == "connect":
            raise httpx.ConnectError("refused", request=request)
        if failure == "garbage":
            return httpx.Response(200, text="<html>oops</html>")
        return httpx.Response(failure, text="boom")

    router = FallbackRouter([provider("primary", handler), provider("secondary", ok_handler("secondary"))])
    assert (await router.complete(PAYLOAD)).provider == "secondary"


@pytest.mark.anyio
async def test_client_error_is_not_retried_on_backup():
    secondary_calls: list = []
    router = FallbackRouter([provider("primary", status_handler(400)), provider("secondary", ok_handler("secondary", secondary_calls))])
    with pytest.raises(Exception) as exc:
        await router.complete(PAYLOAD)
    assert exc.value.status == 400 and exc.value.code == "upstream_rejected_request"
    assert secondary_calls == []


@pytest.mark.anyio
async def test_rate_limited_primary_is_cooled_down():
    primary_calls: list = []

    async def limited(request):
        primary_calls.append(1)
        return httpx.Response(429, headers={"retry-after": "20"})

    router = FallbackRouter([provider("primary", limited), provider("secondary", ok_handler("secondary"))])
    for _ in range(3):
        assert (await router.complete(PAYLOAD)).provider == "secondary"
    assert len(primary_calls) == 1  # later requests skip the known-throttled provider


@pytest.mark.anyio
async def test_cooling_provider_is_still_a_last_resort():
    state = {"primary_mode": 429}

    async def primary(request):
        if state["primary_mode"] == 429:
            return httpx.Response(429, headers={"retry-after": "20"})
        return await ok_handler("primary")(request)

    router = FallbackRouter([provider("primary", primary), provider("secondary", status_handler(503))])
    with pytest.raises(Exception):
        await router.complete(PAYLOAD)  # primary 429 (now cooling), secondary 503
    state["primary_mode"] = "ok"
    assert (await router.complete(PAYLOAD)).provider == "primary"  # secondary tried first, then primary


@pytest.mark.anyio
async def test_all_providers_failing_raises_sanitized_error():
    router = FallbackRouter([provider("primary", status_handler(429, headers={"retry-after": "7"})), provider("secondary", status_handler(500))])
    with pytest.raises(Exception) as exc:
        await router.complete(PAYLOAD)
    err = exc.value
    assert (err.status, err.code, err.retry_after_s) == (503, "all_providers_unavailable", 7.0)
    assert "Traceback" not in err.message


# --------------------------------------------------------------------------- HTTP app


API_KEY = "sk-test-tenant"


def make_app(tmp_path, clock, targets, tokens_per_window=50_000):
    limiter = SqliteTokenRateLimiter(tmp_path / "app.sqlite3", tokens_per_window=tokens_per_window, clock=clock)
    app = create_app(limiter=limiter, router=FallbackRouter(targets, attempt_timeout_s=0.3), api_keys={API_KEY: "acme"})
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
    return client, limiter


AUTH = {"Authorization": f"Bearer {API_KEY}"}
BODY = {"model": "default", "messages": [{"role": "user", "content": "Hello there"}], "max_tokens": 100}


def assert_standard_error(resp: httpx.Response, status: int, code: str) -> dict:
    assert resp.status_code == status
    err = resp.json()["error"]
    assert set(err) >= {"type", "code", "message", "request_id"}
    assert err["code"] == code
    assert err["request_id"] == resp.headers["x-request-id"]
    assert "Traceback" not in resp.text and "internal detail" not in resp.text
    return err


@pytest.mark.anyio
async def test_app_success_reconciles_reserved_tokens_to_actual_usage(tmp_path, clock):
    client, limiter = make_app(tmp_path, clock, [provider("primary", ok_handler("primary")), provider("secondary", ok_handler("secondary"))])
    async with client:
        resp = await client.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert resp.status_code == 200
    assert resp.headers["x-gateway-provider"] == "primary"
    assert reservation_tokens(BODY) > 42
    assert await limiter.usage(tenant_key(API_KEY)) == 42  # reservation replaced by provider-reported usage
    assert resp.headers["x-ratelimit-remaining-tokens"] == str(50_000 - 42)
    stored = rows(tmp_path / "app.sqlite3")[0][0]
    assert stored == tenant_key(API_KEY) and API_KEY not in stored  # raw API key never persisted


@pytest.mark.anyio
async def test_app_failover_on_429(tmp_path, clock):
    client, _ = make_app(tmp_path, clock, [provider("primary", status_handler(429)), provider("secondary", ok_handler("secondary"))])
    async with client:
        resp = await client.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert resp.headers["x-gateway-provider"] == "secondary"
    assert resp.json()["choices"][0]["message"]["content"] == "from secondary"


@pytest.mark.anyio
async def test_app_tenant_rate_limit_returns_429_with_retry_after(tmp_path, clock):
    client, _ = make_app(tmp_path, clock, [provider("primary", ok_handler("primary"))], tokens_per_window=300)
    async with client:
        body = {**BODY, "max_tokens": 200}  # reservation ~210 tokens; actual usage reconciles to 42
        assert (await client.post("/v1/chat/completions", json=body, headers=AUTH)).status_code == 200
        assert (await client.post("/v1/chat/completions", json=body, headers=AUTH)).status_code == 200
        clock.advance(1)
        assert (await client.post("/v1/chat/completions", json=body, headers=AUTH)).status_code == 200
        resp = await client.post("/v1/chat/completions", json=body, headers=AUTH)
    err = assert_standard_error(resp, 429, "tenant_token_limit_exceeded")
    assert err["type"] == "rate_limit_error"
    # used 3x42 at t=0,0,1; ~210 more needs 36 freed -> first t=0 entry expires at t=60; now t=1.
    assert resp.headers["retry-after"] == str(err["retry_after_s"]) == "59"


@pytest.mark.anyio
async def test_app_all_providers_down_releases_reservation(tmp_path, clock):
    client, limiter = make_app(tmp_path, clock, [provider("primary", status_handler(500)), provider("secondary", status_handler(502))])
    async with client:
        resp = await client.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert_standard_error(resp, 503, "all_providers_unavailable")
    assert await limiter.usage(tenant_key(API_KEY)) == 0  # gateway failure does not burn tenant budget


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("headers", "body", "status", "code"),
    [
        ({}, BODY, 401, "invalid_api_key"),
        ({"Authorization": "Bearer sk-unknown"}, BODY, 401, "invalid_api_key"),
        (AUTH, {"messages": []}, 400, "invalid_request"),
        (AUTH, {"messages": [{"role": "wizard", "content": "x"}]}, 400, "invalid_request"),
        (AUTH, {**BODY, "max_tokens": -5}, 400, "invalid_request"),
        (AUTH, {**BODY, "stream": True}, 400, "stream_not_supported"),
        (AUTH, {**BODY, "max_tokens": 32_000, "messages": [{"role": "user", "content": "x" * 100_000}]}, 400, "request_exceeds_token_limit"),
    ],
)
async def test_app_rejections_use_standard_payload(tmp_path, clock, headers, body, status, code):
    client, _ = make_app(tmp_path, clock, [provider("primary", ok_handler("primary"))])
    async with client:
        resp = await client.post("/v1/chat/completions", json=body, headers=headers)
    assert_standard_error(resp, status, code)


@pytest.mark.anyio
async def test_app_unhandled_exception_is_sanitized(tmp_path, clock):
    class ExplodingRouter:
        async def complete(self, payload):
            raise RuntimeError("Traceback (most recent call last): internal detail at /srv/secret.py")

    limiter = SqliteTokenRateLimiter(tmp_path / "x.sqlite3", clock=clock)
    app = create_app(limiter=limiter, router=ExplodingRouter(), api_keys={API_KEY: "acme"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://gw") as client:
        resp = await client.post("/v1/chat/completions", json=BODY, headers=AUTH)
    err = assert_standard_error(resp, 500, "internal_error")
    assert err["message"] == "An internal error occurred."
    assert await limiter.usage(tenant_key(API_KEY)) == 0


@pytest.mark.anyio
async def test_app_unknown_route_uses_standard_payload(tmp_path, clock):
    client, _ = make_app(tmp_path, clock, [provider("primary", ok_handler("primary"))])
    async with client:
        resp = await client.get("/v1/nope")
    assert_standard_error(resp, 404, "not_found")
