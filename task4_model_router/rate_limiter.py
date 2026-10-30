"""Token-aware sliding-window rate limiter persisted in an on-disk SQLite database.

Algorithm: sliding *log*. Every admitted request writes one row (tenant, ts, tokens).
A tenant's usage is the sum of its rows with ts in (now - window, now]; rows at or
beyond the window edge are evicted. This is exact — unlike fixed windows it has no
boundary burst where 2x the limit fits around a window flip.

Token-awareness: the true cost of a completion is unknown until it finishes, so
admission *reserves* an estimate (prompt estimate + max_tokens) and the router later
*reconciles* the row to the provider-reported usage, or *releases* it if no provider
served the request (a gateway failure must not burn the tenant's budget).

Concurrency: check-and-reserve runs in one ``BEGIN IMMEDIATE`` transaction, which takes
SQLite's write lock up front, so concurrent requests — from threads in this process or
from other gateway processes sharing the file — cannot both see the same headroom.
SQLite calls are blocking, so they run in worker threads (one connection per thread).
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import anyio.to_thread

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant  TEXT    NOT NULL,                        -- sha256(api key); raw keys are never stored
    ts      REAL    NOT NULL,                        -- unix epoch seconds (wall clock: shared across processes)
    tokens  INTEGER NOT NULL CHECK (tokens >= 0)
);
CREATE INDEX IF NOT EXISTS ix_token_usage_tenant_ts ON token_usage (tenant, ts);
"""


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_s: float, limit: int, used: int):
        super().__init__(f"rate limited; retry in {retry_after_s:.2f}s")
        self.retry_after_s, self.limit, self.used = retry_after_s, limit, used


class RequestTooLarge(Exception):
    def __init__(self, requested: int, limit: int):
        super().__init__(f"request needs {requested} tokens; limit is {limit}")
        self.requested, self.limit = requested, limit


@dataclass(frozen=True)
class Reservation:
    row_id: int
    tenant: str
    tokens: int
    limit: int
    remaining: int  # headroom left in the window right after this reservation


def tenant_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class SqliteTokenRateLimiter:
    def __init__(
        self,
        db_path: str | Path,
        *,
        tokens_per_window: int = 50_000,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.time,
        busy_timeout_ms: int = 5_000,
    ):
        self.db_path = str(db_path)
        self.tokens_per_window = tokens_per_window
        self.window_s = window_s
        self._clock = clock
        self._busy_timeout_ms = busy_timeout_ms
        self._local = threading.local()
        self._all_connections: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    # ------------------------------------------------------------------ connections

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False, timeout=self._busy_timeout_ms / 1000)
        conn.execute("PRAGMA journal_mode=WAL")  # readers never block the writer
        conn.execute("PRAGMA synchronous=NORMAL")  # durable at checkpoints; safe with WAL
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        return conn

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = self._connect()
            with self._lock:
                self._all_connections.append(conn)
        return conn

    def close(self) -> None:
        with self._lock:
            for conn in self._all_connections:
                conn.close()
            self._all_connections.clear()

    # ------------------------------------------------------------------ sync core (runs in worker threads)

    def _acquire_sync(self, tenant: str, tokens: int) -> Reservation:
        limit = self.tokens_per_window
        if tokens > limit:
            raise RequestTooLarge(tokens, limit)
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            now = self._clock()
            cutoff = now - self.window_s
            conn.execute("DELETE FROM token_usage WHERE tenant = ? AND ts <= ?", (tenant, cutoff))
            (used,) = conn.execute("SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ?", (tenant,)).fetchone()
            if used + tokens > limit:
                retry_after = self._retry_after(conn, tenant, need=used + tokens - limit, now=now)
                conn.execute("COMMIT")  # keep the eviction
                raise RateLimitExceeded(retry_after, limit, used)
            cur = conn.execute("INSERT INTO token_usage (tenant, ts, tokens) VALUES (?, ?, ?)", (tenant, now, tokens))
            conn.execute("COMMIT")
            return Reservation(cur.lastrowid, tenant, tokens, limit, limit - used - tokens)
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def _retry_after(self, conn: sqlite3.Connection, tenant: str, *, need: int, now: float) -> float:
        """Seconds until enough of the oldest entries age out to free ``need`` tokens."""
        freed = 0
        for ts, tokens in conn.execute("SELECT ts, tokens FROM token_usage WHERE tenant = ? ORDER BY ts", (tenant,)):
            freed += tokens
            if freed >= need:
                return max(0.0, ts + self.window_s - now)
        return self.window_s

    def _reconcile_sync(self, reservation: Reservation, actual_tokens: int) -> None:
        self._conn().execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (max(0, actual_tokens), reservation.row_id))

    def _release_sync(self, reservation: Reservation) -> None:
        self._conn().execute("DELETE FROM token_usage WHERE id = ?", (reservation.row_id,))

    def _usage_sync(self, tenant: str) -> int:
        cutoff = self._clock() - self.window_s
        (used,) = self._conn().execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts > ?", (tenant, cutoff)
        ).fetchone()
        return used

    def _purge_expired_sync(self) -> int:
        """Global eviction for tenants that stopped sending (per-tenant eviction happens on acquire)."""
        cutoff = self._clock() - self.window_s
        return self._conn().execute("DELETE FROM token_usage WHERE ts <= ?", (cutoff,)).rowcount

    # ------------------------------------------------------------------ async API

    async def acquire(self, tenant: str, tokens: int) -> Reservation:
        return await anyio.to_thread.run_sync(self._acquire_sync, tenant, tokens)

    async def reconcile(self, reservation: Reservation, actual_tokens: int) -> None:
        await anyio.to_thread.run_sync(self._reconcile_sync, reservation, actual_tokens)

    async def release(self, reservation: Reservation) -> None:
        await anyio.to_thread.run_sync(self._release_sync, reservation)

    async def usage(self, tenant: str) -> int:
        return await anyio.to_thread.run_sync(self._usage_sync, tenant)

    async def purge_expired(self) -> int:
        return await anyio.to_thread.run_sync(self._purge_expired_sync)
