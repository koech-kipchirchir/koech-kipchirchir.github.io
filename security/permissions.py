"""
Fine-grained permission system with action+resource model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class PermissionAction(Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    SEARCH = "search"
    ADMIN = "admin"
    ALL = "*"


@dataclass(frozen=True)
class Permission:
    """A single permission: an action on a resource.

    Resource uses colon-separated hierarchy (e.g. ``chat:session``)
    and supports glob patterns.
    """
    action: PermissionAction = PermissionAction.READ
    resource: str = "*"

    def allows(self, action: PermissionAction, resource: str) -> bool:
        """Check if this permission grants the given action on the given resource."""
        if self.action != PermissionAction.ALL and self.action != action:
            return False
        if self.resource == "*":
            return True
        return _match_resource(self.resource, resource)

    def __repr__(self) -> str:
        return f"Permission({self.action.value}, {self.resource})"


class PermissionSet:
    """A set of permissions with efficient match checking."""

    def __init__(self, permissions: list[Permission] | None = None) -> None:
        self._permissions: list[Permission] = list(permissions or [])

    def add(self, permission: Permission) -> None:
        self._permissions.append(permission)

    def is_allowed(self, action: PermissionAction | str, resource: str) -> bool:
        if isinstance(action, str):
            action = PermissionAction(action)
        # Wildcard permission matches everything
        if any(p.action == PermissionAction.ALL and p.resource == "*" for p in self._permissions):
            return True
        return any(p.allows(action, resource) for p in self._permissions)


def _match_resource(pattern: str, resource: str) -> bool:
    """Match a resource pattern against a resource string.

    Supports ``*`` (any segment) and ``**`` (any depth).
    Examples:
        chat:*        -> chat:session, chat:history
        documents:**  -> documents:upload, documents:search, documents:index
        *:*           -> any two-segment resource
    """
    if pattern == resource:
        return True
    if pattern == "*":
        return True

    # Convert glob pattern to regex
    regex_parts = []
    for part in pattern.split(":"):
        if part == "**":
            regex_parts.append(".*")
        elif part == "*":
            regex_parts.append("[^:]+")
        else:
            regex_parts.append(re.escape(part))
    regex = "^" + ":".join(regex_parts) + "$"
    return bool(re.match(regex, resource))
