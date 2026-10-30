"""Incremental PII redaction over a text stream.

The problem: PII can straddle chunk boundaries ("john.doe@" | "example.com"), so a
redactor that scans each chunk independently leaks. Buffering the whole response
fixes that but destroys TTFT and memory bounds.

Approach: keep only a *holdback* tail. After every chunk we search the pending text
with the `regex` module's partial matching, which also reports matches that run
into the end of the buffer and could still grow. Text before the earliest such
candidate can never become part of a match, so it is redacted and released
immediately; only the candidate tail is held. The output is identical to redacting
the fully buffered response in one pass, for every possible chunking (fuzz-tested).

Bounds: every pattern uses bounded quantifiers, so a partial match is at most
``max_match_len`` characters long — the holdback never exceeds it no matter how
long the stream is. In ordinary prose the held tail is the current word (it
might be the local part of an email), i.e. output lags input by about one token.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import regex


def luhn_valid(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        checksum += d
    return checksum % 10 == 0


@dataclass(frozen=True)
class PiiPattern:
    name: str
    pattern: str
    max_len: int
    validator: Callable[[str], bool] | None = None  # reject false positives after a syntactic match


# ASCII-only classes ([0-9], not \d): the regex module's \d is Unicode-aware.
EMAIL = PiiPattern(
    "EMAIL",
    r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]{1,64}@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.){1,8}[A-Za-z]{2,24}(?![A-Za-z0-9\-])",
    max_len=64 + 1 + 8 * 64 + 24,
)
SSN = PiiPattern(
    "SSN",
    # Separator required (dash or space, used consistently) to avoid redacting every 9-digit number.
    r"(?<![0-9])(?!000|666|9[0-9]{2})[0-9]{3}(?P<ssn_sep>[\- ])(?!00)[0-9]{2}(?P=ssn_sep)(?!0000)[0-9]{4}(?![0-9])",
    max_len=11,
)
CREDIT_CARD = PiiPattern(
    "CREDIT_CARD",
    # 13-19 contiguous digits, 4-4-4-4(-3) groups, or Amex/Diners 4-6-4/5 groups; Luhn-checked.
    r"(?<![0-9])(?:[0-9]{13,19}|[0-9]{4}(?:[\- ][0-9]{4}){3}(?:[\- ][0-9]{3})?|[0-9]{4}[\- ][0-9]{6}[\- ][0-9]{4,5})(?![0-9])",
    max_len=23,
    validator=luhn_valid,
)
DEFAULT_PATTERNS: tuple[PiiPattern, ...] = (EMAIL, SSN, CREDIT_CARD)
REDACTED = "[REDACTED]"


class PiiMatcher:
    """Compiled alternation of all patterns; shared, immutable, safe to reuse across streams."""

    def __init__(self, patterns: Iterable[PiiPattern] = DEFAULT_PATTERNS):
        self.patterns = {p.name: p for p in patterns}
        alternation = "|".join(f"(?P<{p.name}>{p.pattern})" for p in self.patterns.values())
        self.regex = regex.compile(alternation, regex.V0)
        # Can never match completely (nothing follows the end of the string), so a partial search
        # returns the leftmost position where *some* match path runs into the end of the buffer —
        # i.e. the leftmost start that more input could still turn into, or change, a match.
        # (A plain partial search is not enough: it prefers a later complete match over an
        # earlier partial one, e.g. the SSN inside "co.123-45-6789" over the email prefix "co.1…".)
        self._open_tail = regex.compile(f"(?:{alternation})\\Z.", regex.V0)
        self.max_match_len = max(p.max_len for p in self.patterns.values())

    def _hold_point(self, text: str, pos: int) -> int:
        m = self._open_tail.search(text, pos, partial=True)
        return m.start() if m is not None else len(text)

    def scan(self, text: str, *, begin: int = 0, final: bool, replacement: str, counts: Counter[str]) -> tuple[str, int]:
        """Redact ``text[begin:]``; ``text[:begin]`` is context for lookbehinds only and is not emitted.

        Returns ``(safe_output, consumed)``: ``text[begin:consumed]`` is settled and redacted
        exactly as a one-shot scan of the whole stream would redact it; ``text[consumed:]``
        may still be part of a match and must wait for more input (always empty when final).
        """
        n = len(text)
        out: list[str] = []
        cursor = pos = begin
        hold = n if final else self._hold_point(text, pos)
        while True:
            m = self.regex.search(text, pos) if pos < hold else None
            if m is None or m.start() >= hold:
                # Nothing settled left before the hold point. A match starting before it can
                # never be affected by future input: no match path from there reaches the end.
                out.append(text[cursor:hold])
                return "".join(out), hold
            start, end = m.span()
            kind = next(name for name in self.patterns if m.group(name) is not None)
            validator = self.patterns[kind].validator
            if validator is not None and not validator(m.group()):
                pos = start + 1  # not PII; rescan inside it so a real match it overlaps is not skipped
                continue
            out.append(text[cursor:start])
            out.append(replacement)
            counts[kind] += 1
            cursor = pos = end
            if end > hold:  # this match swallowed the old hold point; find the next one
                hold = n if final else self._hold_point(text, pos)


DEFAULT_MATCHER = PiiMatcher()


class StreamingRedactor:
    """Stateful per-stream redactor. ``feed()`` each delta, ``flush()`` once at end of stream."""

    CONTEXT_CHARS = 8  # raw text kept from before the held tail so lookbehinds see real context

    def __init__(self, matcher: PiiMatcher = DEFAULT_MATCHER, *, replacement: str = REDACTED, max_holdback: int | None = None):
        self._matcher = matcher
        self._replacement = replacement
        self._context = ""
        self._pending = ""
        # Safety valve for custom patterns; with the bounded defaults this is never reached.
        self._max_holdback = max_holdback or 2 * matcher.max_match_len
        self.counts: Counter[str] = Counter()

    @property
    def held(self) -> int:
        return len(self._pending)

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        safe = self._scan(self._pending + chunk, final=False)
        if len(self._pending) > self._max_holdback:
            overflow = len(self._pending) - self._max_holdback
            rest = self._pending[overflow:]
            safe += self._scan(self._pending[:overflow], final=True)
            self._pending = rest
        return safe

    def flush(self) -> str:
        text, self._pending = self._pending, ""
        return self._scan(text, final=True) if text else ""

    def _scan(self, text: str, *, final: bool) -> str:
        begin = len(self._context)
        full = self._context + text
        safe, consumed = self._matcher.scan(full, begin=begin, final=final, replacement=self._replacement, counts=self.counts)
        self._pending = full[consumed:]
        self._context = full[max(0, consumed - self.CONTEXT_CHARS) : consumed]
        return safe


def redact_text(text: str, matcher: PiiMatcher = DEFAULT_MATCHER, replacement: str = REDACTED) -> str:
    """One-shot redaction (non-streaming responses, and the oracle for streaming tests)."""
    return matcher.scan(text, final=True, replacement=replacement, counts=Counter())[0]
