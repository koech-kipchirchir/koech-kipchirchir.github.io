"""
Enhanced rate limiter with per-user, per-role, and per-endpoint buckets.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Optional

from security.config import SecurityConfig
from security.models import Role

logger = logging.getLogger("aios.security.rate_limiter")


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int = 60, message: str = "Rate limit exceeded") -> None:
        self.retry_after = retry_after
        self.message = message
        super().__init__(message)


class SlidingWindowBucket:
    """Sliding window rate limit bucket using timestamp lists."""

    __slots__ = ("_max_requests", "_window_seconds", "_timestamps")

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._timestamps: list[float] = []

    def allow(self) -> tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, retry_after_seconds)."""
        now = time.time()
        cutoff = now - self._window_seconds
        # Prune expired timestamps
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        if len(self._timestamps) >= self._max_requests:
            retry_after = int(self._window_seconds - (now - self._timestamps[0]))
            return False, max(1, retry_after)

        self._timestamps.append(now)
        return True, 0


class RateLimiter:
    """Multi-level rate limiter with IP, user, role, and endpoint granularity."""

    def __init__(self, config: SecurityConfig) -> None:
        self._config = config
        self._enabled = config.rate_limit_enabled

        # Bucket hierarchy
        self._global_bucket = SlidingWindowBucket(
            config.rate_limit_default * 10,
            config.rate_limit_window_seconds,
        )
        self._ip_buckets: dict[str, SlidingWindowBucket] = {}
        self._user_buckets: dict[str, SlidingWindowBucket] = {}
        self._endpoint_buckets: dict[str, SlidingWindowBucket] = {}

    def check(
        self,
        ip: str = "",
        user_id: str = "",
        role: Role | str | None = None,
        endpoint: str = "",
    ) -> None:
        """Check rate limits at all applicable levels. Raises RateLimitExceeded on failure."""
        if not self._enabled:
            return

        window = self._config.rate_limit_window_seconds

        # 1. Global
        allowed, retry = self._global_bucket.allow()
        if not allowed:
            raise RateLimitExceeded(retry_after=retry, message="Global rate limit exceeded")

        # 2. Per-IP
        if ip:
            bucket = self._ip_buckets.get(ip)
            if bucket is None:
                bucket = SlidingWindowBucket(self._config.rate_limit_default, window)
                self._ip_buckets[ip] = bucket
            allowed, retry = bucket.allow()
            if not allowed:
                raise RateLimitExceeded(retry_after=retry, message="IP rate limit exceeded")

        # 3. Per-User
        if user_id:
            bucket = self._user_buckets.get(user_id)
            if bucket is None:
                user_limit = self._config.rate_limit_per_role.get("user", self._config.rate_limit_default)
                if role:
                    if isinstance(role, Role):
                        role_val = role.value
                    else:
                        role_val = role
                    user_limit = self._config.rate_limit_per_role.get(role_val, user_limit)
                bucket = SlidingWindowBucket(user_limit, window)
                self._user_buckets[user_id] = bucket
            allowed, retry = bucket.allow()
            if not allowed:
                raise RateLimitExceeded(retry_after=retry, message="User rate limit exceeded")

        # 4. Per-Endpoint
        if endpoint:
            bucket = self._endpoint_buckets.get(endpoint)
            if bucket is None:
                bucket = SlidingWindowBucket(self._config.rate_limit_default, window)
                self._endpoint_buckets[endpoint] = bucket
            allowed, retry = bucket.allow()
            if not allowed:
                raise RateLimitExceeded(retry_after=retry, message="Endpoint rate limit exceeded")

    def get_rate_limit(self, role: Role | str | None = None) -> int:
        """Get the rate limit count for a given role."""
        if isinstance(role, Role):
            role = role.value
        return self._config.rate_limit_per_role.get(role or "user", self._config.rate_limit_default)

    def cleanup(self) -> None:
        """Prune expired buckets to free memory."""
        now = time.time()
        cutoff = now - self._config.rate_limit_window_seconds * 2
        for store in (self._ip_buckets, self._user_buckets, self._endpoint_buckets):
            expired = [k for k, v in store.items() if v._timestamps and v._timestamps[-1] < cutoff]
            for k in expired:
                del store[k]
