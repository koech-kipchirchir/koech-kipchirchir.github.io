"""
FastAPI dependencies for route protection.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader

from security.config import SecurityConfig
from security.jwt import decode_token, JWTError, JWTExpiredError
from security.models import Role
from security.permissions import Permission, PermissionAction, PermissionSet
from security.rbac import resolve_role_permissions, role_has_at_least
from security.api_keys import APIKeyStore

logger = logging.getLogger("aios.security.dependencies")

# --- Schemes ---
_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_security_config(request: Request) -> SecurityConfig:
    """Extract SecurityConfig from app state."""
    config = getattr(request.app.state, "security_config", None)
    if config is None:
        config = SecurityConfig.from_env()
    return config


async def get_current_user(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    api_key: str | None = Depends(_api_key_scheme),
) -> dict[str, Any]:
    """Extract the authenticated user from JWT or API key.

    Returns user claims dict with keys: sub, username, role, auth_method, is_verified.
    Raises 401 if authentication fails.
    """
    config = get_security_config(request)

    # Check if already authenticated by middleware
    user = request.state.user if hasattr(request.state, "user") else None
    if user and user.get("auth_method") is not None:
        return user

    # JWT Bearer token
    if bearer:
        try:
            claims = decode_token(bearer.credentials, config, verify_type="access")
            return claims
        except JWTExpiredError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )

    # API Key
    if api_key:
        api_key_store: APIKeyStore = getattr(request.app.state, "api_key_store", APIKeyStore())
        key = api_key_store.validate(api_key)
        if key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        from security.auth import get_user_store
        user_record = get_user_store().get_user(key.user_id)
        if user_record is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key user not found",
            )
        role = key.role_override.value if key.role_override else user_record.role.value
        return {
            "sub": user_record.id,
            "username": user_record.username,
            "email": user_record.email,
            "role": role,
            "auth_method": "api_key",
            "is_verified": user_record.is_verified,
        }

    # No credentials
    return {
        "sub": "",
        "username": "anonymous",
        "role": "anonymous",
        "auth_method": None,
        "is_verified": False,
    }


async def optional_user(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Like get_current_user but never raises — returns anonymous claims."""
    return user


async def require_authenticated(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Require an authenticated user (not anonymous)."""
    if not user.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user


def require_role(minimum_role: Role | str):
    """Dependency factory: require a minimum role level.

    Usage::

        @router.get("/admin")
        async def admin_endpoint(user: dict = Depends(require_role("admin"))):
            ...
    """
    if isinstance(minimum_role, str):
        minimum_role = Role(minimum_role)

    async def _dependency(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        user_role_val = user.get("role", "anonymous")
        try:
            user_role = Role(user_role_val)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Unknown role: {user_role_val}",
            )
        if not role_has_at_least(user_role, minimum_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user_role_val}' does not have permission. "
                       f"Requires at least '{minimum_role.value}'.",
            )
        return user

    return _dependency


def require_permission(action: PermissionAction | str, resource: str):
    """Dependency factory: require a specific permission.

    Usage::

        @router.post("/documents")
        async def upload(
            user: dict = Depends(require_permission("create", "documents")),
        ):
            ...
    """
    if isinstance(action, str):
        action = PermissionAction(action)

    async def _dependency(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        user_role_val = user.get("role", "anonymous")
        try:
            user_role = Role(user_role_val)
        except ValueError:
            raise HTTPException(status_code=403, detail=f"Unknown role: {user_role_val}")

        perms = resolve_role_permissions(user_role)
        ps = PermissionSet(perms)

        if not ps.is_allowed(action, resource):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {action.value} on {resource} "
                       f"(role={user_role_val})",
            )
        return user

    return _dependency
