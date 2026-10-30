"""Micro-benchmark for the streaming redactor:  python -m task3_pii_guardrail.bench

Reports throughput, per-delta latency added by redaction, and peak holdback
(the only per-stream text the guardrail retains).
"""

from __future__ import annotations

import random
import statistics
import time

from .redactor import StreamingRedactor, redact_text

SENTENCES = [
    "The quarterly report shows steady growth across all regions. ",
    "Please reach out to jane.roe@example.com for the contract. ",
    "Her SSN on file is 123-45-6789 and must never be shared. ",
    "Charge card 4111 1111 1111 1111 for the renewal invoice. ",
    "Order 98765432 shipped on 2026-10-01 via ground freight. ",
]


def main(total_chars: int = 2_000_000, seed: int = 7) -> None:
    rng = random.Random(seed)
    text = "".join(rng.choice(SENTENCES) for _ in range(total_chars // 55))
    deltas, i = [], 0
    while i < len(text):
        n = rng.randint(1, 8)  # typical LLM delta sizes
        deltas.append(text[i : i + n])
        i += n

    redactor = StreamingRedactor()
    latencies, peak_hold, out = [], 0, []
    start = time.perf_counter()
    for d in deltas:
        t0 = time.perf_counter_ns()
        out.append(redactor.feed(d))
        latencies.append(time.perf_counter_ns() - t0)
        peak_hold = max(peak_hold, redactor.held)
    out.append(redactor.flush())
    elapsed = time.perf_counter() - start

    assert "".join(out) == redact_text(text), "streaming output must equal one-shot redaction"
    latencies.sort()
    print(f"input              {len(text) / 1e6:.2f} M chars in {len(deltas):,} deltas")
    print(f"throughput         {len(text) / elapsed / 1e6:.2f} M chars/s")
    print(f"per-delta latency  p50 {statistics.median(latencies) / 1000:.1f} us   p99 {latencies[int(len(latencies) * 0.99)] / 1000:.1f} us")
    print(f"peak holdback      {peak_hold} chars (bounded; independent of stream length)")
    print(f"redactions         {dict(redactor.counts)}")


if __name__ == "__main__":
    main()
