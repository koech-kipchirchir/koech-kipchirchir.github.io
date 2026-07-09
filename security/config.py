"""
Security configuration with sensitive-parameter validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SecurityConfig:
    """Security subsystem configuration.

    Loaded from AIOS_SEC_* environment variables by default.
    """

    # --- JWT ---
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_access_token_ttl_minutes: int = 30
    jwt_refresh_token_ttl_days: int = 7
    jwt_issuer: str = "aios"

    # --- API Keys ---
    api_key_enabled: bool = True
    api_key_length: int = 32
    api_key_header: str = "X-API-Key"

    # --- Encryption ---
    encryption_key: str = ""
    encryption_algorithm: str = "AES-256-GCM"

    # --- Rate Limiting ---
    rate_limit_enabled: bool = True
    rate_limit_default: int = 60
    rate_limit_window_seconds: int = 60
    rate_limit_per_user: int = 120
    rate_limit_per_role: dict[str, int] = field(default_factory=lambda: {
        "admin": 300,
        "user": 120,
        "anonymous": 20,
    })

    # --- Password ---
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digit: bool = True
    password_require_special: bool = False
    bcrypt_rounds: int = 12

    # --- Session ---
    session_ttl_hours: int = 24
    max_sessions_per_user: int = 5

    # --- Audit ---
    audit_enabled: bool = True
    audit_log_file: str = ""
    audit_retention_days: int = 90

    # --- CORS ---
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])

    # --- Security Headers ---
    enable_security_headers: bool = True
    hsts_max_age: int = 31536000
    content_security_policy: str = "default-src 'self'"

    # --- Paths ---
    secrets_dir: str = ""
    encryption_key_file: str = ""

    @classmethod
    def from_env(cls) -> SecurityConfig:
        """Load from AIOS_SEC_* environment variables."""
        prefix = "AIOS_SEC_"
        kwargs: dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            env_name = f"{prefix}{field_name.upper()}"
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            ft = cls.__dataclass_fields__[field_name].type
            if ft is bool or ft == "bool":
                kwargs[field_name] = raw.lower() in ("1", "true", "yes", "on")
            elif ft is int or ft == "int":
                kwargs[field_name] = int(raw)
            elif "dict" in str(ft):
                import json
                try:
                    kwargs[field_name] = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    kwargs[field_name] = {}
            elif "list" in str(ft):
                kwargs[field_name] = [x.strip() for x in raw.split(",") if x.strip()]
            else:
                kwargs[field_name] = raw

        return cls(**kwargs)

    def validate(self) -> None:
        """Raise ValueError if required secrets are missing."""
        if not self.jwt_secret_key or self.jwt_secret_key == "change-me":
            raise ValueError(
                "AIOS_SEC_JWT_SECRET_KEY is not set. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        if not self.encryption_key or self.encryption_key == "change-me":
            raise ValueError(
                "AIOS_SEC_ENCRYPTION_KEY is not set. "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
