"""
FastAPI middleware: authentication, audit logging, and security headers.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from security.audit import AuditEvent, get_audit_store, record_audit_event
from security.auth import get_user_store
from security.config import SecurityConfig
from security.jwt import decode_token, JWTError
from security.rate_limiter import RateLimiter, RateLimitExceeded
from security.api_keys import APIKeyStore

logger = logging.getLogger("aios.security.middleware")


class AuthMiddleware(BaseHTTPMiddleware):
    """Extract and validate JWT or API key from requests.

    Sets ``request.state.user`` with authenticated user claims
    and ``request.state.auth_method`` (jwt, api_key, or None).
    """

    def __init__(
        self,
        app: FastAPI,
        config: SecurityConfig,
        api_key_store: APIKeyStore | None = None,
    ) -> None:
        super().__init__(app)
        self._config = config
        self._api_key_store = api_key_store or APIKeyStore()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request.state.user = None
        request.state.auth_method = None
        request.state.auth_error = None

        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get(self._config.api_key_header, "")

        # 1. Try API key (X-API-Key header)
        if api_key_header:
            api_key = self._api_key_store.validate(api_key_header)
            if api_key:
                user = get_user_store().get_user(api_key.user_id)
                if user:
                    request.state.user = user.to_claims()
                    request.state.auth_method = "api_key"
                    request.state.user["auth_method"] = "api_key"
                    request.state.user["token_type"] = "api_key"

        # 2. Try JWT Bearer token
        if request.state.user is None and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                claims = decode_token(token, self._config, verify_type="access")
                request.state.user = claims
                request.state.auth_method = "jwt"
                request.state.user["auth_method"] = "jwt"
                request.state.user["token_type"] = "access"
            except JWTError as e:
                request.state.auth_error = str(e)

        # 3. Set defaults for unauthenticated requests
        if request.state.user is None:
            request.state.user = {
                "sub": "",
                "username": "anonymous",
                "role": "anonymous",
                "is_verified": False,
                "auth_method": None,
            }

        response = await call_next(request)
        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Record audit events for every request."""

    def __init__(self, app: FastAPI, config: SecurityConfig) -> None:
        super().__init__(app)
        self._config = config

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not self._config.audit_enabled:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        user = request.state.user or {}
        status = "success" if response.status_code < 400 else "error"

        # Skip health checks to reduce noise
        path = request.url.path
        if path in ("/v1/live", "/v1/ready", "/v1/health", "/metrics"):
            return response

        record_audit_event(
            action=request.method.lower(),
            resource=path,
            user_id=user.get("sub", ""),
            username=user.get("username", "anonymous"),
            role=user.get("role", "anonymous"),
            ip_address=request.client.host if request.client else "",
            status=status,
            details={
                "method": request.method,
                "path": path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 1),
                "user_agent": request.headers.get("user-agent", ""),
            },
            request_id=getattr(request.state, "request_id", ""),
        )

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""

    def __init__(self, app: FastAPI, config: SecurityConfig) -> None:
        super().__init__(app)
        self._config = config

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)

        if self._config.enable_security_headers:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self._config.hsts_max_age}; includeSubDomains"
            )
            response.headers["Content-Security-Policy"] = self._config.content_security_policy
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = (
                "camera=(), microphone=(), geolocation=()"
            )
            response.headers["Cache-Control"] = "no-store"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply rate limiting to all requests."""

    def __init__(
        self,
        app: FastAPI,
        config: SecurityConfig,
        rate_limiter: RateLimiter,
    ) -> None:
        super().__init__(app)
        self._config = config
        self._limiter = rate_limiter

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not self._config.rate_limit_enabled:
            return await call_next(request)

        user = request.state.user or {}
        user_id = user.get("sub", "")
        role = user.get("role", "anonymous")
        ip = request.client.host if request.client else ""
        path = request.url.path

        # Skip rate limiting for health endpoints
        if path in ("/v1/live", "/v1/ready", "/v1/health", "/metrics"):
            return await call_next(request)

        try:
            self._limiter.check(ip=ip, user_id=user_id, role=role, endpoint=path)
        except RateLimitExceeded as e:
            logger.warning("Rate limit exceeded: ip=%s user=%s path=%s", ip, user_id, path)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limit_exceeded",
                        "message": e.message,
                        "retry_after": e.retry_after,
                        "status": 429,
                    }
                },
                headers={"Retry-After": str(e.retry_after)},
            )

        return await call_next(request)


def configure_security_middleware(
    app: FastAPI,
    config: SecurityConfig,
    api_key_store: APIKeyStore | None = None,
    rate_limiter: RateLimiter | None = None,
) -> None:
    """Add all security middleware to the FastAPI app.

    Order matters: AuthMiddleware runs first to populate request.state.user,
    then RateLimitMiddleware, then AuditMiddleware, then SecurityHeadersMiddleware.
    """
    limiter = rate_limiter or RateLimiter(config)

    # Security headers (outermost — runs last on response)
    app.add_middleware(SecurityHeadersMiddleware, config=config)

    # Audit logging
    if config.audit_enabled:
        app.add_middleware(AuditMiddleware, config=config)

    # Rate limiting
    if config.rate_limit_enabled:
        app.add_middleware(RateLimitMiddleware, config=config, rate_limiter=limiter)

    # Authentication (innermost — runs first on request)
    app.add_middleware(AuthMiddleware, config=config, api_key_store=api_key_store)

    logger.info(
        "Security middleware configured (auth=%s, rate_limit=%s, audit=%s, headers=%s)",
        True,
        config.rate_limit_enabled,
        config.audit_enabled,
        config.enable_security_headers,
    )
