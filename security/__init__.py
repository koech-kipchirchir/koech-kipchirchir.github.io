"""
AIOS Security Framework
=======================

Production-grade security subsystem with:

- **Authentication:** JWT access/refresh tokens, API keys, password hashing (bcrypt)
- **Authorization:** RBAC with hierarchical roles, fine-grained permission system
- **Encryption:** AES-256-GCM, Fernet symmetric encryption, key derivation (PBKDF2)
- **Secrets Management:** Environment variables, JSON files, HashiCorp Vault
- **Rate Limiting:** Multi-level (IP, user, role, endpoint) sliding window
- **Audit Logging:** Structured JSON events with retention and query
- **Middleware:** Auth extraction, audit recording, security headers, rate limiting
- **FastAPI Dependencies:** `get_current_user`, `require_role`, `require_permission`
"""

from __future__ import annotations

from security.api_keys import generate_api_key, APIKeyStore
from security.audit import (
    AuditEvent, AuditStore, record_audit_event, get_audit_store, set_audit_store,
)
from security.auth import (
    AuthError, UserStore, hash_password, verify_password,
    register_user, authenticate_user, login, refresh_tokens,
    get_user_store,
)
from security.config import SecurityConfig
from security.dependencies import (
    get_current_user, optional_user, require_authenticated,
    require_role, require_permission, get_security_config,
)
from security.encryption import (
    AESGCMEncryptor, FernetEncryptor, EncryptionError,
    generate_encryption_key, derive_key,
)
from security.jwt import (
    create_access_token, create_refresh_token, decode_token,
    refresh_access_token, JWTError, JWTExpiredError, JWTInvalidError,
)
from security.middleware import (
    AuthMiddleware, AuditMiddleware, SecurityHeadersMiddleware,
    RateLimitMiddleware, configure_security_middleware,
)
from security.models import Role, User, Session, APIKey
from security.permissions import Permission, PermissionAction, PermissionSet
from security.rate_limiter import RateLimiter, RateLimitExceeded
from security.rbac import (
    role_has_at_least, resolve_role_permissions, role_permissions_dict,
)
from security.secrets import (
    SecretsProvider, EnvSecretsProvider, FileSecretsProvider,
    ChainedSecretsProvider, create_default_provider,
    get_secret, require_secret, set_global_provider,
)

__all__ = [
    # JWT
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "refresh_access_token",
    "JWTError",
    "JWTExpiredError",
    "JWTInvalidError",
    # Auth
    "AuthError",
    "UserStore",
    "hash_password",
    "verify_password",
    "register_user",
    "authenticate_user",
    "login",
    "refresh_tokens",
    "get_user_store",
    # API Keys
    "generate_api_key",
    "APIKeyStore",
    # RBAC
    "role_has_at_least",
    "resolve_role_permissions",
    "role_permissions_dict",
    # Permissions
    "Permission",
    "PermissionAction",
    "PermissionSet",
    # Encryption
    "AESGCMEncryptor",
    "FernetEncryptor",
    "EncryptionError",
    "generate_encryption_key",
    "derive_key",
    # Secrets
    "SecretsProvider",
    "EnvSecretsProvider",
    "FileSecretsProvider",
    "ChainedSecretsProvider",
    "create_default_provider",
    "get_secret",
    "require_secret",
    "set_global_provider",
    # Rate Limiter
    "RateLimiter",
    "RateLimitExceeded",
    # Audit
    "AuditEvent",
    "AuditStore",
    "record_audit_event",
    "get_audit_store",
    "set_audit_store",
    # Middleware
    "AuthMiddleware",
    "AuditMiddleware",
    "SecurityHeadersMiddleware",
    "RateLimitMiddleware",
    "configure_security_middleware",
    # Models
    "Role",
    "User",
    "Session",
    "APIKey",
    # Config
    "SecurityConfig",
    # Dependencies
    "get_current_user",
    "optional_user",
    "require_authenticated",
    "require_role",
    "require_permission",
    "get_security_config",
]

__version__ = "0.1.0"
