"""Task 1: stdio isolation, JSON-RPC compliance and strict validation of the customer MCP server."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anyio
import mcp_types as types
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError

from task1_mcp_server.schemas import TriggerRefundInput
from task1_mcp_server.server import CustomerStore, ToolExecutionError, validate_arguments

ROOT = Path(__file__).resolve().parents[1]
VALID_REASON = "Item arrived damaged"


# --------------------------------------------------------------------------- raw wire


class RawStdioPeer:
    """Drives the server process line by line, the way an MCP host does."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "task1_mcp_server"],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.raw_stdout: list[str] = []

    def send(self, payload: dict | str) -> None:
        line = payload if isinstance(payload, str) else json.dumps(payload)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def recv(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, "server closed stdout"
        self.raw_stdout.append(line)
        return json.loads(line)

    def request(self, payload: dict | str) -> dict:
        self.send(payload)
        return self.recv()

    def close(self) -> str:
        self.proc.stdin.close()
        rest = self.proc.stdout.read()
        self.raw_stdout.extend(rest.splitlines(keepends=True))
        stderr = self.proc.stderr.read()
        self.proc.wait(timeout=10)
        return stderr


@pytest.fixture
def peer():
    p = RawStdioPeer()
    init = p.request(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}},
        }
    )
    assert init["result"]["serverInfo"]["name"] == "customer-support-mcp"
    p.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    yield p
    if p.proc.poll() is None:
        p.proc.kill()


def call(peer: RawStdioPeer, req_id: int, name: str, arguments=None) -> dict:
    params = {"name": name}
    if arguments is not None:
        params["arguments"] = arguments
    return peer.request({"jsonrpc": "2.0", "id": req_id, "method": "tools/call", "params": params})


def assert_jsonrpc_response(msg: dict) -> None:
    assert msg["jsonrpc"] == "2.0"
    assert "id" in msg
    assert ("result" in msg) != ("error" in msg), "exactly one of result/error"
    if "error" in msg:
        assert isinstance(msg["error"]["code"], int)
        assert isinstance(msg["error"]["message"], str)


def test_stdout_is_pure_jsonrpc_and_logs_go_to_stderr(peer):
    peer.request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    call(peer, 2, "get_customer_record", {"customer_id": "CUST-00042"})
    call(peer, 3, "trigger_refund", {"customer_id": "CUST-00042", "amount": 10.0, "reason": VALID_REASON})
    call(peer, 4, "get_customer_record", {"customer_id": "bad"})
    peer.request("{not json")
    stderr = peer.close()

    assert peer.raw_stdout, "expected responses on stdout"
    for line in peer.raw_stdout:
        assert line.endswith("\n")
        assert_jsonrpc_response(json.loads(line))  # any non-JSON-RPC byte on stdout fails here
    assert "ready on stdio" in stderr
    assert "refund issued" in stderr


def test_claim_stdout_diverts_print_and_child_processes_to_stderr():
    script = (
        "import os, subprocess, sys\n"
        "from task1_mcp_server.server import claim_stdout\n"
        "wire = claim_stdout()\n"
        "print('stray print')\n"
        "os.write(1, b'raw fd write\\n')\n"
        "subprocess.run([sys.executable, '-c', 'print(\"child output\")'])\n"
        "wire.write(b'{\"jsonrpc\":\"2.0\"}\\n'); wire.flush()\n"
    )
    out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert out.stdout == '{"jsonrpc":"2.0"}\n'
    for noise in ("stray print", "raw fd write", "child output"):
        assert noise in out.stderr


def test_tools_list_advertises_strict_schemas(peer):
    tools = {t["name"]: t for t in peer.request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}
    refund = tools["trigger_refund"]["inputSchema"]
    assert refund["additionalProperties"] is False
    assert set(refund["required"]) == {"customer_id", "amount", "reason"}
    assert refund["properties"]["customer_id"]["pattern"] == "^CUST-[0-9]{5}$"
    assert refund["properties"]["amount"]["exclusiveMinimum"] == 0
    assert refund["properties"]["reason"]["minLength"] == 10
    assert tools["trigger_refund"]["annotations"]["destructiveHint"] is True


def test_valid_calls_return_structured_content(peer):
    rec = call(peer, 1, "get_customer_record", {"customer_id": "CUST-00001"})["result"]
    assert rec["isError"] is False
    assert rec["structuredContent"]["name"] == "Ada Lovelace"
    assert json.loads(rec["content"][0]["text"]) == rec["structuredContent"]

    refund = call(peer, 2, "trigger_refund", {"customer_id": "CUST-00001", "amount": 50, "reason": VALID_REASON})["result"]
    assert refund["structuredContent"]["remaining_refundable_balance"] == 1200.0
    assert refund["structuredContent"]["refund_id"].startswith("RF-")


@pytest.mark.parametrize(
    ("name", "arguments", "bad_fields"),
    [
        ("get_customer_record", {"customer_id": "CUST-1234"}, {"customer_id"}),
        ("get_customer_record", {"customer_id": "cust-12345"}, {"customer_id"}),
        ("get_customer_record", {"customer_id": "CUST-123456"}, {"customer_id"}),
        ("get_customer_record", {"customer_id": "CUST-12345\n"}, {"customer_id"}),
        ("get_customer_record", {"customer_id": " CUST-12345"}, {"customer_id"}),
        ("get_customer_record", {"customer_id": "CUST-١٢٣٤٥"}, {"customer_id"}),  # non-ASCII digits
        ("get_customer_record", {"customer_id": 12345}, {"customer_id"}),
        ("get_customer_record", {"customer_id": None}, {"customer_id"}),
        ("get_customer_record", {}, {"customer_id"}),
        ("get_customer_record", None, {"customer_id"}),
        ("get_customer_record", {"customer_id": "CUST-00001", "admin": True}, {"admin"}),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 0, "reason": VALID_REASON}, {"amount"}),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": -1.5, "reason": VALID_REASON}, {"amount"}),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": "10.00", "reason": VALID_REASON}, {"amount"}),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": True, "reason": VALID_REASON}, {"amount"}),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 1.005, "reason": VALID_REASON}, {"amount"}),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 1e9, "reason": VALID_REASON}, {"amount"}),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 10, "reason": "too short"}, {"reason"}),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 10, "reason": "   padded    "}, {"reason"}),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 10, "reason": "x" * 501}, {"reason"}),
        ("trigger_refund", {"customer_id": "CUST-00001", "amount": 10, "reason": ["a list"]}, {"reason"}),
        ("trigger_refund", {"customer_id": "nope", "amount": -1, "reason": ""}, {"customer_id", "amount", "reason"}),
    ],
)
def test_invalid_arguments_map_to_invalid_params(peer, name, arguments, bad_fields):
    resp = call(peer, 7, name, arguments)
    assert_jsonrpc_response(resp)
    assert resp["id"] == 7
    err = resp["error"]
    assert err["code"] == -32602
    assert {e["field"] for e in err["data"]["errors"]} == bad_fields
    assert all(set(e) == {"field", "message", "type"} for e in err["data"]["errors"])  # raw values never echoed


def test_non_finite_amount_rejected():
    for value in (float("inf"), float("nan")):
        with pytest.raises(MCPError) as exc:
            validate_arguments("trigger_refund", {"customer_id": "CUST-00001", "amount": value, "reason": VALID_REASON})
        assert exc.value.code == types.INVALID_PARAMS


def test_unknown_tool_is_invalid_params(peer):
    resp = call(peer, 9, "drop_all_tables", {})
    assert resp["error"]["code"] == -32602
    assert resp["error"]["message"] == "Unknown tool: drop_all_tables"


def test_protocol_level_errors(peer):
    assert peer.request("{oops")["error"] == {"code": -32700, "message": "Parse error"}
    assert peer.request('{"jsonrpc":"2.0","id":1}')["error"]["code"] == -32600
    assert peer.request({"jsonrpc": "2.0", "id": 2, "method": "no/such"})["error"]["code"] == -32601
    assert peer.request({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"arguments": {}}})["error"]["code"] == -32602
    # The server keeps serving after each rejection.
    assert call(peer, 4, "get_customer_record", {"customer_id": "CUST-00042"})["result"]["isError"] is False


def test_business_failures_are_tool_errors_not_protocol_errors(peer):
    missing = call(peer, 1, "get_customer_record", {"customer_id": "CUST-99999"})["result"]
    assert missing["isError"] is True and "not found" in missing["content"][0]["text"]
    too_much = call(peer, 2, "trigger_refund", {"customer_id": "CUST-00042", "amount": 9999, "reason": VALID_REASON})["result"]
    assert too_much["isError"] is True and "exceeds refundable balance" in too_much["content"][0]["text"]
    suspended = call(peer, 3, "trigger_refund", {"customer_id": "CUST-31337", "amount": 1, "reason": VALID_REASON})["result"]
    assert suspended["isError"] is True and "suspended" in suspended["content"][0]["text"]


# --------------------------------------------------------------------------- unit


@pytest.mark.anyio
async def test_concurrent_refunds_cannot_overdraw_balance():
    store = CustomerStore()  # CUST-00042 has 300.00 refundable
    req = TriggerRefundInput(customer_id="CUST-00042", amount=200, reason=VALID_REASON)
    outcomes: list[str] = []

    async def attempt():
        try:
            await store.refund(req)
            outcomes.append("ok")
        except ToolExecutionError:
            outcomes.append("rejected")

    async with anyio.create_task_group() as tg:
        for _ in range(5):
            tg.start_soon(attempt)
    assert outcomes.count("ok") == 1
    assert store.get("CUST-00042").refundable_balance == 100.0


# --------------------------------------------------------------------------- official SDK client


@pytest.mark.anyio
async def test_official_sdk_client_end_to_end():
    params = StdioServerParameters(command=sys.executable, args=["-m", "task1_mcp_server"], cwd=str(ROOT))
    with open("/dev/null", "w") as devnull:
        async with stdio_client(params, errlog=devnull) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {"get_customer_record", "trigger_refund"}

            ok = await session.call_tool("get_customer_record", {"customer_id": "CUST-00042"})
            assert ok.structured_content["email"] == "alan@example.com"

            with pytest.raises(MCPError) as exc:
                await session.call_tool("trigger_refund", {"customer_id": "CUST-00042", "amount": -1, "reason": "no"})
            assert exc.value.code == types.INVALID_PARAMS
