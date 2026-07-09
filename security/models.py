"""
Data models for users, roles, sessions, and API keys.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    return uuid.uuid4().hex[:16]


class Role(Enum):
    ANONYMOUS = "anonymous"
    USER = "user"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"


@dataclass
class User:
    id: str = ""
    username: str = ""
    email: str = ""
    password_hash: str = ""
    role: Role = Role.USER
    is_active: bool = True
    is_verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_claims(self) -> dict[str, Any]:
        return {
            "sub": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "is_verified": self.is_verified,
        }


@dataclass
class Session:
    id: str = ""
    user_id: str = ""
    refresh_token_hash: str = ""
    ip_address: str = ""
    user_agent: str = ""
    is_active: bool = True
    expires_at: str = ""
    created_at: str = ""


@dataclass
class APIKey:
    id: str = ""
    key_prefix: str = ""
    key_hash: str = ""
    name: str = ""
    user_id: str = ""
    role_override: Role | None = None
    permissions: list[str] = field(default_factory=list)
    is_active: bool = True
    last_used_at: str = ""
    expires_at: str | None = None
    created_at: str = ""
