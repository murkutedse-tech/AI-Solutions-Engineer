"""Strict Pydantic input/output contracts for the customer-support MCP tools.

The same models are used twice: their JSON Schema is advertised to clients in
``tools/list`` and they validate every ``tools/call`` before any business
logic runs. Keeping one source of truth means the advertised contract and the
enforced contract cannot drift apart.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

# `[0-9]` rather than `\d`: pydantic-core uses the Rust regex engine, where `\d`
# is Unicode-aware and would accept e.g. Arabic-Indic digits ("CUST-١٢٣٤٥").
CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"
MAX_REFUND_AMOUNT = 10_000.00

CustomerId = Annotated[
    str,
    StringConstraints(pattern=CUSTOMER_ID_PATTERN),
    Field(description="Customer identifier formatted as CUST-XXXXX (5 digits).", examples=["CUST-00042"]),
]


class StrictInput(BaseModel):
    """Base for tool inputs: no type coercion, no unknown fields, immutable."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class GetCustomerRecordInput(StrictInput):
    customer_id: CustomerId


class TriggerRefundInput(StrictInput):
    customer_id: CustomerId
    amount: Annotated[
        float,
        Field(
            gt=0,
            le=MAX_REFUND_AMOUNT,
            allow_inf_nan=False,
            description=f"Refund amount in USD. Positive, at most 2 decimal places, <= {MAX_REFUND_AMOUNT:.2f}.",
            examples=[49.99],
        ),
    ]
    reason: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=10, max_length=500),
        Field(description="Justification for the refund (10-500 characters, surrounding whitespace ignored)."),
    ]

    @field_validator("amount")
    @classmethod
    def _at_most_two_decimal_places(cls, value: float) -> float:
        # repr() of a float is the shortest string that round-trips, so this
        # reflects what the client actually sent (10.1 -> "10.1", not 10.0999...).
        exponent = Decimal(repr(value)).as_tuple().exponent
        if isinstance(exponent, int) and exponent < -2:
            raise ValueError("amount must have at most 2 decimal places")
        return value


class CustomerRecord(BaseModel):
    customer_id: str
    name: str
    email: str
    tier: str
    status: str
    refundable_balance: float


class RefundReceipt(BaseModel):
    refund_id: str
    customer_id: str
    amount: float
    reason: str
    status: str
    remaining_refundable_balance: float
