"""
JWT token creation, validation, and refresh.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from typing import Any, Optional

from security.config import SecurityConfig

logger = logging.getLogger("aios.security.jwt")


class JWTError(Exception):
    pass


class JWTExpiredError(JWTError):
    pass


class JWTInvalidError(JWTError):
    pass


def _b64_encode(data: bytes) -> str:
    return urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64_decode(data: str) -> bytes:
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return urlsafe_b64decode(data)


def _hmac_sha256(key: str, message: str) -> str:
    return hmac.new(key.encode(), message.encode(), hashlib.sha256).hexdigest()


def create_access_token(
    claims: dict[str, Any],
    config: SecurityConfig,
) -> str:
    """Create a signed JWT access token."""
    header = {"alg": config.jwt_algorithm, "typ": "JWT"}
    payload = {
        **claims,
        "iss": config.jwt_issuer,
        "iat": int(time.time()),
        "exp": int(time.time()) + config.jwt_access_token_ttl_minutes * 60,
        "jti": uuid.uuid4().hex[:16],
        "type": "access",
    }
    return _encode(header, payload, config.jwt_secret_key)


def create_refresh_token(
    user_id: str,
    config: SecurityConfig,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a signed JWT refresh token (longer-lived)."""
    header = {"alg": config.jwt_algorithm, "typ": "JWT"}
    payload = {
        "sub": user_id,
        "iss": config.jwt_issuer,
        "iat": int(time.time()),
        "exp": int(time.time()) + config.jwt_refresh_token_ttl_days * 86400,
        "jti": uuid.uuid4().hex[:16],
        "type": "refresh",
        **(extra_claims or {}),
    }
    return _encode(header, payload, config.jwt_secret_key)


def decode_token(token: str, config: SecurityConfig, verify_type: str | None = "access") -> dict[str, Any]:
    """Decode and verify a JWT token.

    Raises JWTExpiredError if expired, JWTInvalidError for other failures.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise JWTInvalidError("Invalid token format")

    header_b64, payload_b64, sig = parts

    # Verify signature
    expected_sig = _hmac_sha256(config.jwt_secret_key, f"{header_b64}.{payload_b64}")
    if not hmac.compare_digest(sig, expected_sig):
        raise JWTInvalidError("Invalid token signature")

    # Decode payload
    try:
        payload = json.loads(_b64_decode(payload_b64))
    except (ValueError, json.JSONDecodeError) as e:
        raise JWTInvalidError(f"Invalid token payload: {e}")

    # Check expiry
    exp = payload.get("exp", 0)
    if time.time() > exp:
        raise JWTExpiredError("Token has expired")

    # Check issuer
    if payload.get("iss") != config.jwt_issuer:
        raise JWTInvalidError("Invalid token issuer")

    # Check token type
    if verify_type and payload.get("type") != verify_type:
        raise JWTInvalidError(f"Invalid token type (expected {verify_type})")

    return payload


def refresh_access_token(refresh_token: str, config: SecurityConfig) -> tuple[str, str, dict[str, Any]]:
    """Exchange a refresh token for a new access + refresh token pair.

    Returns (new_access_token, new_refresh_token, claims).
    """
    payload = decode_token(refresh_token, config, verify_type="refresh")
    user_id = payload.get("sub", "")
    if not user_id:
        raise JWTInvalidError("Refresh token missing subject")

    claims = {k: v for k, v in payload.items() if k in ("sub", "username", "email", "role")}
    new_access = create_access_token(claims, config)
    new_refresh = create_refresh_token(user_id, config)

    return new_access, new_refresh, claims


def revoke_token(jti: str) -> None:
    """Mark a token as revoked (stub — integrate with Redis/DB for production)."""
    pass


# ---------------------------------------------------------------------------
# Internal: encode a JWT
# ---------------------------------------------------------------------------

def _encode(header: dict, payload: dict, secret: str) -> str:
    header_b64 = _b64_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64_encode(json.dumps(payload, separators=(",", ":")).encode())
    sig = _hmac_sha256(secret, f"{header_b64}.{payload_b64}")
    return f"{header_b64}.{payload_b64}.{sig}"
