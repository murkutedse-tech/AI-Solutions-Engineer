# AI Field Developer / Solutions Engineer Assessment

Four runnable Python services covering the assessment tasks, each with a test suite:

| # | Task | Package | Tests |
|---|------|---------|-------|
| 1 | Custom MCP server: strict validation, stdio transport | [`task1_mcp_server/`](task1_mcp_server) | 32 |
| 2 | MCP security gateway: bearer auth, `admin_*` tool authorization | [`task2_mcp_gateway/`](task2_mcp_gateway) | 42 |
| 3 | LLM gateway streaming guardrail: real-time PII redaction | [`task3_pii_guardrail/`](task3_pii_guardrail) | 41 |
| 4 | Token-aware rate limiter (on-disk SQLite) and model fallback router | [`task4_model_router/`](task4_model_router) | 37 |

```bash
pip install -r requirements.txt     # Python 3.11+; the official MCP SDK (mcp>=2.3)
python -m pytest                    # 152 tests, ~25 s
```

---

## Task 1: Customer MCP server (`task1_mcp_server`)

```bash
python -m task1_mcp_server          # speaks MCP on stdin/stdout; logs on stderr
```

Example host config (Claude Desktop and similar):

```json
{ "mcpServers": { "customers": { "command": "python", "args": ["-m", "task1_mcp_server"], "cwd": "/path/to/repo" } } }
```

**Tools.** `get_customer_record(customer_id)` and `trigger_refund(customer_id, amount, reason)`. Seed customers: `CUST-00001`, `CUST-00042`, and `CUST-31337` (suspended).

**Stdio isolation.** `claim_stdout()` runs first in `main()`. It duplicates fd 1 into a private descriptor that only the transport writes to. It then points fd 1 at stderr and rebinds `sys.stdout`. After that, a stray `print()`, a C extension writing to fd 1, or a child process inheriting fd 1 all land on stderr. Logging is configured for stderr only, and warnings are captured into logging. The SDK diverts fd 1 itself only while `stdio_server` is running; claiming at process start also covers import-time noise and older SDK versions. Tests check every stdout byte against the JSON-RPC schema. They also show `print`, `os.write(1, ...)` and a subprocess's output all ending up on stderr.

**Validation.** One set of Pydantic models (`schemas.py`) is both advertised as each tool's `inputSchema`/`outputSchema` in `tools/list` and enforced on every `tools/call`:
- `strict=True`: no coercion, so `"10.00"` and `true` are rejected as amounts.
- `extra="forbid"`: unknown fields are rejected.
- Customer ID matches `^CUST-[0-9]{5}$`. It uses `[0-9]` because `\d` in the Rust regex engine accepts non-ASCII digits such as `CUST-١٢٣٤٥`.
- Amount must be `> 0`, finite, at most 2 decimal places, and `<= 10,000`.
- Reason is whitespace-stripped, then must be 10–500 characters.

Error details list each failing field with a message and an error type. They never echo the submitted values.

**Error mapping**

| Situation | Response |
|---|---|
| Unknown tool, missing or invalid arguments | JSON-RPC `-32602` Invalid params, with `data.errors[]` |
| Invalid JSON line / not a JSON-RPC message | `-32700` / `-32600` with `id: null`. The SDK drops these silently; a filter stage adds the required reply |
| Unknown method | `-32601` |
| Unexpected exception | `-32603 Internal error`; the traceback goes to stderr only |
| Business rule (not found, over balance, suspended) | `CallToolResult` with `isError: true`, so the model can read the reason and recover |

> Newer MCP spec revisions recommend reporting *argument* validation failures as tool execution errors (`isError: true`) so the model can self-correct. This server follows the brief and uses `-32602`. Switching means catching the `MCPError` from `validate_arguments` in `call_tool` and returning `_tool_error(...)`.

Refunds use an async lock around check-then-debit, so concurrent refunds cannot overdraw a balance (tested).

---

## Task 2: MCP security gateway (`task2_mcp_gateway`)

```bash
python -m task2_mcp_gateway.mock_downstream                  # downstream MCP server on :9001
DOWNSTREAM_MCP_URL=http://127.0.0.1:9001/mcp python -m task2_mcp_gateway    # gateway on :8080
TOKEN=$(python -m task2_mcp_gateway.auth --role viewer --sub alice)
curl -s localhost:8080/mcp -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{}}}'
# {"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unauthorized Tool Call","data":{...}}}
```

**Pipeline for `POST /mcp`**

1. **Authenticate.** Read the `Bearer` HS256 JWT and verify it with the standard library (constant-time compare, algorithm pinned so `alg: none` is rejected, `exp`/`nbf`/`iss`/`aud` checked). Take `role` (`admin` | `viewer`) and `sub` from the claims. Failure returns HTTP 401 with `WWW-Authenticate` and a JSON-RPC error `-32000`. `JWTVerifier.verify()` is the only interface the gateway uses, so an RS256/JWKS verifier for a real IdP can replace it.
2. **Bounded read.** Bodies over 1 MiB get HTTP 413.
3. **Strict parse.** Duplicate keys, `NaN`, batches, a bad `jsonrpc`/`id`/`method`/`params` shape, or a tools/call with no id all get `-32700`/`-32600`. Client responses to server-initiated requests (sampling, elicitation) are recognized and forwarded.
4. **Authorize.** `tools/list` and every other method pass through unchanged. For `tools/call`, `params.name` must be a non-empty string (else `-32602`, failing closed). If it is an `admin_*` tool and the role isn't `admin`, the gateway returns `-32001 Unauthorized Tool Call` with the original `id`, and **the downstream server is never contacted**. Tests verify this with a recording mock.
5. **Forward.** The gateway sends its **re-serialized** message plus allow-listed headers (`Accept`, `Mcp-Session-Id`, `MCP-Protocol-Version`, `Last-Event-ID`) and streams the response back as-is, whether JSON or SSE. `GET` (SSE stream) and `DELETE` (end session) are authenticated and forwarded. If the downstream is unreachable, the client gets HTTP 502 with `-32003` and no internal detail.

**Bypasses closed (each has a test)**
- **Duplicate-key smuggling.** `{"name":"admin_x","name":"get_weather"}` is read differently by different JSON parsers. Rejecting duplicates *and* forwarding the parsed object means the downstream server runs exactly the message that was authorized.
- **Batch smuggling.** An allowed call followed by an admin call, sent as an array. Batches were removed in MCP 2025-06-18, so they are rejected outright.
- **Look-alike names.** `ADMIN_x`, ` admin_x`, `​admin_x` and full-width `ａｄｍｉｎ_x` are all treated as admin. The check applies NFKC, strips invisible format characters, trims and casefolds before comparing the prefix. That protects against a downstream that resolves names loosely.
- **No token passthrough.** The client's bearer token is *not* forwarded (this follows the MCP spec). The verified identity goes downstream as `X-Authenticated-Subject` and `X-Authenticated-Role`. Those headers are only trustworthy if the gateway is the downstream server's only network path.

A tool-level denial returns HTTP 200 carrying a JSON-RPC error. That is an application-level outcome, and MCP clients surface it as a failed tool call; many clients treat a non-2xx status as a transport failure. Every allow/deny decision is audit-logged (subject, role, method, tool, decision).

Possible extensions: hide `admin_*` tools from viewers' `tools/list` results, a default-deny method allow-list, and per-tool rate limits.

---

## Task 3: Streaming PII guardrail (`task3_pii_guardrail`)

```bash
python -m task3_pii_guardrail.mock_provider        # OpenAI-compatible mock on :9002 (echoes your message)
python -m task3_pii_guardrail                      # gateway on :8081 (LLM_PROVIDER_BASE_URL, LLM_PROVIDER_API_KEY)
curl -N localhost:8081/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"m","stream":true,"messages":[{"role":"user","content":"mail jane@corp.com, SSN 123-45-6789"}]}'
python -m task3_pii_guardrail.bench                # throughput / latency / holdback numbers
```

**The problem.** PII straddles chunk boundaries (`"jane.roe@"` + `"example.com"`). Redacting each chunk on its own leaks it. Buffering the whole response fixes that but ruins TTFT and memory.

**Algorithm (`redactor.py`)**
- One compiled alternation covers EMAIL, SSN (`AAA-GG-SSSS` or space-separated, with invalid area/group/serial numbers excluded) and CREDIT_CARD (13–19 digits, grouped or contiguous, **Luhn-validated**).
- After each delta, a second regex, `(?:ALT)\Z.`, is searched in partial mode. It can never match completely, so the search returns the **leftmost position where some match path runs into the end of the buffer**. That is the earliest point more input could still change. A plain partial search isn't enough here: it prefers a *later* complete match over an *earlier* partial one, for example the SSN inside `co.123-45-6789…`, which becomes part of an email once `a@b.co` arrives.
- Everything before that point is settled, redacted and emitted immediately. Only the tail is held. Some cases this handles:
  - `a@mail.co` + `.uk` is held until the email can't grow.
  - `123-45-6789` + `0` stays unredacted, because a 10-digit run isn't an SSN.
  - A Luhn-failing digit run is re-scanned from `start+1`, so an SSN inside it is still caught.
- Every quantifier is bounded, so **the holdback can never exceed the longest possible match**, whatever the stream length. In prose the held tail is the current word, since it might be the local part of an email, so output trails input by about one token. A few characters of raw context are kept so lookbehind assertions behave as they would on the full text.
- **Exactness.** For *every* chunking, the streamed output equals one-shot redaction of the full response. Tests check every 2-way split of a sample and every 3-way split of each PII value, plus 3,000 seeded fuzz cases; 90,000 more were run during development.

**Gateway (`gateway.py`)**
- An incremental SSE parser handles CRLF, CR and LF line endings, comments, multi-line `data:` and UTF-8 code points split across reads.
- Each choice index gets its own redactor (correct for `n > 1`). Events whose text is entirely held back are dropped, not sent empty. When a choice reports `finish_reason`, and at `[DONE]` or upstream EOF, the redacted tail is flushed *before* the finishing event.
- Nothing is accumulated. Per-stream memory is one partial SSE event plus the bounded holdback.
- When the client disconnects, the generator is cancelled and the upstream connection closed, so the provider stops generating tokens nobody reads (tested).
- Response headers `Cache-Control: no-cache` and `X-Accel-Buffering: no` stop intermediaries from buffering the SSE stream. Non-streaming requests are redacted in one pass, and provider errors are sanitized.

**Benchmark** (one core, random 1–8 character deltas): about 0.56M chars/s (~140k tokens/s), with ~7 µs p50 and ~28 µs p99 added per delta. Peak holdback was 20 characters over 2M characters of input. A single stream runs at roughly 100 tokens/s, so redaction CPU is negligible next to model latency.

**Limitations.** Regexes catch structured PII only; names and addresses need an NER model, which can be added as another stage. SSNs with mixed separators (`123-45 6789`) and bare 9-digit numbers are deliberately not matched, to avoid redacting every order number. `tool_calls[].function.arguments` deltas are not yet redacted; they would use the same per-choice redactor.

---

## Task 4: Rate limiter and fallback router (`task4_model_router`)

```bash
python -m task4_model_router.mock_providers        # primary :9101, secondary :9102
python -m task4_model_router                       # gateway :8082, state in ./gateway_ratelimit.sqlite3
curl -s localhost:8082/v1/chat/completions -H 'Authorization: Bearer sk-tenant-a' -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
curl -s -X POST localhost:9101/_mock/mode -d '{"mode":"slow"}'           # or rate_limited | error | garbage | ok
```

Configuration (env): `TENANT_API_KEYS="key:tenant,..."`, `TOKENS_PER_MINUTE` (default 50000), `RATE_LIMIT_DB`, `ATTEMPT_TIMEOUT_S` (default 3.0), and `PRIMARY_*` / `SECONDARY_*` for the `URL`, `MODEL` and `API_KEY` of each provider.

**Rate limiter (`rate_limiter.py`): sliding log on on-disk SQLite**
- Each admitted request writes a row `(tenant, ts, tokens)`. Usage is the sum over `(now − 60 s, now]`. This is exact: a fixed window would let 2× the limit through around the window flip (tested).
- **Token-aware.** Admission *reserves* `prompt estimate + max_tokens`. After the response the row is *reconciled* to the provider's `usage.total_tokens`. If no provider served the request, or the client went away, the reservation is *released* in a shielded scope, so a gateway failure never burns the tenant's budget.
- **Atomic.** Evict, sum, check and insert run in one `BEGIN IMMEDIATE` transaction, which takes SQLite's write lock up front. WAL mode and a busy timeout keep readers unblocked. Calls run in worker threads with one connection per thread, so the event loop never blocks. Tests show 60 concurrent tasks admitting exactly 25 × 2k tokens, and **4 separate processes** sharing one file admitting exactly 50 × 1k tokens.
- **Eviction.** Expired rows are deleted on each acquire for that tenant, and a background sweep clears tenants that have gone quiet. `Retry-After` is the moment the oldest entries free enough room, not a guess.
- API keys are stored as SHA-256 hashes, never raw. Timestamps use wall-clock time because the state is shared between processes. A single request bigger than the whole limit is rejected with 400, since it could never succeed.

**Router (`router.py`)**
- **The 3000 ms deadline is wall-clock for the whole attempt** (connect, send, wait, read body), enforced with `anyio.fail_after`. httpx timeouts are per operation, so a server sending one byte every 2 s never trips a 3 s read timeout (tested). The httpx timeouts are set looser so the deadline is what governs.
- **Timeout race.** On expiry the in-flight request is cancelled and its connection dropped. A late primary response can't be used or race the backup, because it no longer exists (tested).
- **Failover triggers:** 429, timeouts, transport errors, 5xx, a malformed 200, and provider-side 401/403/404 (a misconfigured primary shouldn't take the service down). Other 4xx mean the request itself is bad, so the backup isn't called; the client gets a sanitized 400.
- **Cooldown.** A 429 with `Retry-After` moves that provider to the back of the order for up to 30 s, so later requests skip a wasted round trip. It remains a last resort if the other providers fail.

**Standardized errors (`errors.py`).** Every failure, including unknown routes and unhandled exceptions, leaves in one shape:

```json
{"error": {"type": "rate_limit_error", "code": "tenant_token_limit_exceeded",
           "message": "Token rate limit of 50000 tokens per minute exceeded for this API key.",
           "request_id": "req_9f2c…", "retry_after_s": 12}}
```

Upstream bodies, exception text and stack traces never reach the client. Tests use mock providers that return Python tracebacks and verify nothing leaks. Details are logged server-side under the same `request_id`, which the client also gets in `X-Request-ID`.

---

## Layout

```
task1_mcp_server/     schemas.py (Pydantic contracts), server.py (MCP server, stdout claim, error mapping)
task2_mcp_gateway/    auth.py (JWT), gateway.py (proxy + policy), mock_downstream.py
task3_pii_guardrail/  redactor.py (streaming redactor), gateway.py (SSE proxy), mock_provider.py, bench.py
task4_model_router/   rate_limiter.py (SQLite), router.py (fallback), tokens.py, errors.py, app.py, mock_providers.py
tests/                one pytest module per task
```
