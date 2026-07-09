"""
Role-based access control with hierarchical roles and permission resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from security.models import Role
from security.permissions import Permission, PermissionAction


# Role hierarchy: higher-index roles inherit all permissions of lower ones
ROLE_HIERARCHY: list[Role] = [
    Role.ANONYMOUS,
    Role.USER,
    Role.ADMIN,
    Role.SUPERADMIN,
]


def role_has_at_least(current: Role, minimum: Role) -> bool:
    """Check if ``current`` is at or above ``minimum`` in the hierarchy."""
    try:
        return ROLE_HIERARCHY.index(current) >= ROLE_HIERARCHY.index(minimum)
    except ValueError:
        return False


def resolve_role_permissions(role: Role) -> list[Permission]:
    """Return all permissions granted to a role, including inherited ones."""
    result: list[Permission] = []
    try:
        idx = ROLE_HIERARCHY.index(role)
        for r in ROLE_HIERARCHY[:idx + 1]:
            result.extend(_ROLE_PERMISSIONS.get(r, []))
    except ValueError:
        result = _ROLE_PERMISSIONS.get(role, [])
    return result


def role_permissions_dict(role: Role) -> dict[str, list[str]]:
    """Return {resource: [actions]} for the given role (inherited)."""
    perms: dict[str, set[str]] = {}
    for p in resolve_role_permissions(role):
        if p.resource not in perms:
            perms[p.resource] = set()
        perms[p.resource].add(p.action.value)
    return {r: sorted(acts) for r, acts in perms.items()}


# ---------------------------------------------------------------------------
# Default role-permission assignments
# ---------------------------------------------------------------------------

def _perm(action: str, resource: str = "*") -> Permission:
    return Permission(action=PermissionAction(action), resource=resource)


_ROLE_PERMISSIONS: dict[Role, list[Permission]] = {
    Role.ANONYMOUS: [
        _perm("read", "health"),
        _perm("read", "docs"),
        _perm("create", "auth:login"),
        _perm("create", "auth:register"),
    ],
    Role.USER: [
        _perm("read", "health"),
        _perm("read", "docs"),
        _perm("create", "auth:login"),
        _perm("create", "auth:register"),
        _perm("create", "chat"),
        _perm("read", "chat"),
        _perm("delete", "chat"),
        _perm("create", "memory"),
        _perm("read", "memory"),
        _perm("delete", "memory"),
        _perm("create", "documents"),
        _perm("read", "documents"),
        _perm("search", "documents"),
        _perm("read", "models"),
        _perm("read", "tools"),
        _perm("execute", "tools"),
    ],
    Role.ADMIN: [
        _perm("*", "*"),  # Wildcard — all actions on all resources
    ],
    Role.SUPERADMIN: [
        _perm("*", "*"),
    ],
}
