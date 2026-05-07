"""Async token-bucket rate limiter for the Home Connect HTTP request path.

Why this exists
---------------
The Home Connect cloud enforces both a daily call quota and a short-window
burst cap per client_id. When either is exceeded the API returns HTTP 429
with a ``Retry-After`` header; the client then has to sit out the window
(seconds-to-hours, depending on which limit was hit). Reactive 429
handling is already implemented in :mod:`api.py`, but reactive handling
alone leaves the integration *guaranteed* to fall off the cliff once a
day if the call rate averages above the quota.

A small token bucket in front of the request loop turns "fall off the
cliff" into "ride the curve" — calls beyond the sustained quota wait a
few seconds rather than triggering a multi-hour BLOCKED state.

Defaults are sized for the publicly-documented BSH free / developer tier
(1000 calls per 24 hours per client_id, with a short-window burst cap):

* ``capacity = 30`` tokens — sized to cover an initial appliance
  discovery sweep (≈ 5 calls per appliance × ~5 appliances + a handful
  for auth / list) without waiting, while staying well under the
  observed ~50/5-minute short-window cap. Smaller capacity also caps
  the per-restart overshoot: each integration restart instantiates a
  fresh full bucket, and ``capacity`` extra tokens above the sustained
  daily allowance are the unavoidable cost of that.
* ``refill_rate = (1000 - 30) / 86400`` tokens per second. Subtracting
  ``capacity`` from the daily allowance is intentional: the bucket's
  total output across a 24 h period is ``capacity + refill_rate * 86400``
  (one full burst plus continuous refill), so a refill rate of strictly
  ``1000 / 86400`` would emit 1030 calls/day — 30 over the quota.
  Subtracting the burst keeps the daily ceiling at exactly 1000 with a
  single restart per day. Integrations that restart multiple times per
  day will need a more conservative refill rate to stay under quota.

Callers on a paid / accredited tier can pass a ``TokenBucket`` instance
with higher numbers via ``HomeConnect.async_create(rate_limiter=...)``.
"""
from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Simple asyncio-friendly token bucket.

    The bucket holds at most ``capacity`` tokens and refills continuously
    at ``refill_rate`` tokens per second. Each call to :meth:`acquire`
    waits until at least one token is available and then consumes it.
    """

    DEFAULT_CAPACITY: float = 30.0
    # (daily quota - capacity) / 86400 — see module docstring for why we
    # subtract capacity. With these numbers, one full burst plus 24 h of
    # sustained calls lands at exactly 1000 (BSH free-tier daily quota).
    DEFAULT_REFILL_RATE: float = (1000.0 - DEFAULT_CAPACITY) / 86400.0

    def __init__(
        self,
        capacity: float = DEFAULT_CAPACITY,
        refill_rate: float = DEFAULT_REFILL_RATE,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be positive")
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Suspend until a token is available, then consume one."""
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity,
                    self._tokens + (now - self._last_refill) * self.refill_rate,
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_for = (1.0 - self._tokens) / self.refill_rate
            # Lock released during sleep so other tasks racing on the same
            # bucket are scheduled fairly.
            await asyncio.sleep(wait_for)
