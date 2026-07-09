"""
API key generation, hashing, and validation.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from security.models import APIKey

logger = logging.getLogger("aios.security.api_keys")


def generate_api_key(prefix: str = "aios", key_length: int = 32) -> tuple[str, str, str]:
    """Generate a new API key.

    Returns (full_key, key_prefix, key_hash).
    The full key should be returned to the caller ONCE and never stored.
    Store only the prefix (for identification) and hash (for validation).
    """
    raw = secrets.token_hex(key_length // 2)
    full_key = f"{prefix}_{raw}"
    key_prefix = full_key[:12]
    key_hash = _hash_api_key(full_key)
    return full_key, key_prefix, key_hash


def _hash_api_key(key: str) -> str:
    """Hash an API key using SHA-256 with a static secret for HMAC."""
    return hashlib.sha256(key.encode()).hexdigest()


def validate_api_key(key: str, stored_hash: str) -> bool:
    """Compare a raw API key against a stored hash."""
    return hmac.compare_digest(_hash_api_key(key), stored_hash)


class APIKeyStore:
    """In-memory API key store. Replace with a database-backed store for production."""

    def __init__(self) -> None:
        self._keys: dict[str, APIKey] = {}

    def create_key(
        self,
        user_id: str,
        name: str = "",
        role_override: str | None = None,
        permissions: list[str] | None = None,
        expires_in_days: int | None = None,
    ) -> tuple[APIKey, str]:
        """Create a new API key. Returns (key_record, raw_key)."""
        raw_key, prefix, key_hash = generate_api_key()

        from security.models import Role
        role = Role(role_override) if role_override else None

        key = APIKey(
            id=uuid.uuid4().hex[:16],
            key_prefix=prefix,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            role_override=role,
            permissions=permissions or [],
            is_active=True,
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=(
                datetime.now(timezone.utc).isoformat() if expires_in_days else None
            ),
        )
        self._keys[key.id] = key
        logger.info("Created API key %s for user %s", key.id, user_id)
        return key, raw_key

    def get_key(self, key_id: str) -> APIKey | None:
        return self._keys.get(key_id)

    def get_key_by_prefix(self, prefix: str) -> APIKey | None:
        for k in self._keys.values():
            if k.key_prefix == prefix:
                return k
        return None

    def validate(self, raw_key: str) -> APIKey | None:
        """Validate a raw (full) API key. Returns the key record or None."""
        parts = raw_key.split("_", 1)
        prefix = raw_key[:12] if not raw_key.startswith("_") else raw_key[:12]
        key = self.get_key_by_prefix(prefix)
        if key is None:
            return None
        if not key.is_active:
            return None
        if key.expires_at:
            try:
                exp = datetime.fromisoformat(key.expires_at)
                if exp < datetime.now(timezone.utc):
                    return None
            except (ValueError, TypeError):
                pass
        if not validate_api_key(raw_key, key.key_hash):
            return None
        key.last_used_at = datetime.now(timezone.utc).isoformat()
        return key

    def revoke(self, key_id: str) -> bool:
        key = self._keys.get(key_id)
        if key:
            key.is_active = False
            logger.info("Revoked API key %s", key_id)
            return True
        return False

    def list_keys(self, user_id: str | None = None) -> list[APIKey]:
        if user_id:
            return [k for k in self._keys.values() if k.user_id == user_id]
        return list(self._keys.values())
