"""Customer-support MCP server over stdio.

Run:  python -m task1_mcp_server

Wire discipline: stdout carries newline-delimited JSON-RPC and nothing else.
All logs, warnings and stray prints go to stderr (see ``claim_stdout``).

Error mapping:
  * Unknown tool / malformed or invalid arguments -> JSON-RPC -32602 (Invalid params)
  * Unexpected server fault                        -> JSON-RPC -32603 (Internal error), details only in stderr logs
  * Business-rule failures (customer not found,
    refund exceeds balance)                        -> CallToolResult with isError=true, so the model can
                                                      read the reason and recover (MCP "tool execution error")
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
from io import TextIOWrapper
from typing import Any, BinaryIO

import anyio
import mcp_types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from pydantic import BaseModel, ValidationError

from .schemas import (
    CustomerRecord,
    GetCustomerRecordInput,
    RefundReceipt,
    TriggerRefundInput,
)

logger = logging.getLogger("customer_mcp")


def claim_stdout() -> BinaryIO:
    """Reserve the real stdout exclusively for JSON-RPC and divert everything else to stderr.

    Duplicates fd 1 into a private descriptor for the transport, then points fd 1
    at stderr and rebinds ``sys.stdout``. After this, ``print()``, C extensions
    writing to fd 1 and child processes inheriting fd 1 all land on stderr; only
    the returned handle reaches the client. Must run before anything can write.
    """
    sys.stdout.flush()
    wire_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(wire_fd, "wb")


def configure_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.captureWarnings(True)


# --------------------------------------------------------------------------- data


class CustomerStore:
    """In-memory stand-in for a CRM / billing backend."""

    def __init__(self) -> None:
        self._customers: dict[str, dict[str, Any]] = {
            "CUST-00001": {"name": "Ada Lovelace", "email": "ada@example.com", "tier": "gold", "status": "active", "refundable_balance": 1250.00},
            "CUST-00042": {"name": "Alan Turing", "email": "alan@example.com", "tier": "silver", "status": "active", "refundable_balance": 300.00},
            "CUST-31337": {"name": "Grace Hopper", "email": "grace@example.com", "tier": "platinum", "status": "suspended", "refundable_balance": 0.00},
        }
        self._lock = anyio.Lock()

    def get(self, customer_id: str) -> CustomerRecord | None:
        row = self._customers.get(customer_id)
        return CustomerRecord(customer_id=customer_id, **row) if row else None

    async def refund(self, req: TriggerRefundInput) -> RefundReceipt:
        async with self._lock:  # check-then-debit must be atomic across concurrent calls
            row = self._customers.get(req.customer_id)
            if row is None:
                raise ToolExecutionError(f"Customer {req.customer_id} not found.")
            if row["status"] != "active":
                raise ToolExecutionError(f"Customer {req.customer_id} is {row['status']}; refunds are not permitted.")
            if req.amount > row["refundable_balance"]:
                raise ToolExecutionError(
                    f"Refund of {req.amount:.2f} exceeds refundable balance of {row['refundable_balance']:.2f}."
                )
            row["refundable_balance"] = round(row["refundable_balance"] - req.amount, 2)
            receipt = RefundReceipt(
                refund_id=f"RF-{secrets.token_hex(6).upper()}",
                customer_id=req.customer_id,
                amount=req.amount,
                reason=req.reason,
                status="submitted",
                remaining_refundable_balance=row["refundable_balance"],
            )
        logger.info("refund issued id=%s customer=%s amount=%.2f", receipt.refund_id, receipt.customer_id, receipt.amount)
        return receipt


class ToolExecutionError(Exception):
    """A business-rule failure reported to the model as ``isError: true``."""


# --------------------------------------------------------------------------- tools

TOOLS: dict[str, tuple[type[BaseModel], types.Tool]] = {
    "get_customer_record": (
        GetCustomerRecordInput,
        types.Tool(
            name="get_customer_record",
            title="Get customer record",
            description="Look up a customer's profile, status and refundable balance by customer ID.",
            input_schema=GetCustomerRecordInput.model_json_schema(),
            output_schema=CustomerRecord.model_json_schema(),
            annotations=types.ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False),
        ),
    ),
    "trigger_refund": (
        TriggerRefundInput,
        types.Tool(
            name="trigger_refund",
            title="Trigger refund",
            description="Issue a refund to an active customer. Fails if the amount exceeds the refundable balance.",
            input_schema=TriggerRefundInput.model_json_schema(),
            output_schema=RefundReceipt.model_json_schema(),
            annotations=types.ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False),
        ),
    ),
}


def validate_arguments(tool_name: str, arguments: dict[str, Any] | None) -> BaseModel:
    """Resolve the tool and validate its arguments, raising JSON-RPC -32602 on any failure."""
    entry = TOOLS.get(tool_name)
    if entry is None:
        raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {tool_name}", data={"tool": tool_name})
    model, _ = entry
    try:
        return model.model_validate(arguments if arguments is not None else {})
    except ValidationError as exc:
        # include_input=False: never echo raw (possibly sensitive) values back over the wire.
        errors = [
            {"field": ".".join(str(p) for p in err["loc"]) or "<arguments>", "message": err["msg"], "type": err["type"]}
            for err in exc.errors(include_url=False, include_input=False, include_context=False)
        ]
        raise MCPError(
            code=types.INVALID_PARAMS,
            message=f"Invalid arguments for tool '{tool_name}'",
            data={"tool": tool_name, "errors": errors},
        ) from None


def _ok(payload: BaseModel) -> types.CallToolResult:
    data = payload.model_dump(mode="json")
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data))],
        structured_content=data,
    )


def _tool_error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=message)], is_error=True)


def build_server(store: CustomerStore | None = None) -> Server:
    store = store or CustomerStore()

    async def list_tools(_ctx, _params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[tool for _, tool in TOOLS.values()])

    async def call_tool(_ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = validate_arguments(params.name, params.arguments)
        logger.debug("tools/call name=%s", params.name)
        try:
            if isinstance(args, GetCustomerRecordInput):
                record = store.get(args.customer_id)
                if record is None:
                    return _tool_error(f"Customer {args.customer_id} not found.")
                return _ok(record)
            if isinstance(args, TriggerRefundInput):
                return _ok(await store.refund(args))
        except ToolExecutionError as exc:
            return _tool_error(str(exc))
        except Exception:
            logger.exception("unhandled error in tool %s", params.name)
            raise MCPError(code=types.INTERNAL_ERROR, message="Internal error") from None
        raise MCPError(code=types.INTERNAL_ERROR, message="Internal error")  # unreachable: TOOLS and dispatch out of sync

    return Server(
        "customer-support-mcp",
        version="1.0.0",
        instructions="Customer lookup and refund tools. Customer IDs look like CUST-00042.",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def framing_error(exc: Exception) -> types.JSONRPCError:
    """JSON-RPC 2.0 §5.1: unparseable input -> -32700, valid JSON that is not a request -> -32600 (id null)."""
    invalid_json = isinstance(exc, ValidationError) and any(e["type"] == "json_invalid" for e in exc.errors())
    if invalid_json:
        error = types.ErrorData(code=types.PARSE_ERROR, message="Parse error")
    else:
        error = types.ErrorData(code=types.INVALID_REQUEST, message="Invalid Request")
    return types.JSONRPCError(jsonrpc="2.0", id=None, error=error)


async def serve(wire: BinaryIO) -> None:
    server = build_server()
    stdout = anyio.wrap_file(TextIOWrapper(wire, encoding="utf-8", newline="\n"))
    async with stdio_server(stdout=stdout) as (transport_in, write_stream):
        # The SDK silently drops lines it cannot parse; a compliant peer must answer them.
        # This stage sits between the transport and the server and replies to framing errors.
        to_server, from_transport = anyio.create_memory_object_stream[SessionMessage](0)

        async def filter_framing_errors() -> None:
            async with to_server:
                async for item in transport_in:
                    if isinstance(item, Exception):
                        logger.warning("rejected malformed JSON-RPC frame: %s", type(item).__name__)
                        await write_stream.send(SessionMessage(framing_error(item)))
                    else:
                        await to_server.send(item)

        async with anyio.create_task_group() as tg:
            tg.start_soon(filter_framing_errors)
            logger.info("customer-support-mcp ready on stdio")
            await server.run(from_transport, write_stream, server.create_initialization_options())


def main() -> None:
    wire = claim_stdout()  # first: nothing may reach fd 1 before this
    configure_logging()
    try:
        anyio.run(serve, wire)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("customer-support-mcp shutting down")


if __name__ == "__main__":
    main()
