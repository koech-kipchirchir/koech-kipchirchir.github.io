"""
Audit logging: structured event recording with storage and query.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from security.config import SecurityConfig

logger = logging.getLogger("aios.security.audit")


class AuditEvent:
    """Represents a single auditable event."""

    def __init__(
        self,
        action: str,
        resource: str,
        user_id: str = "",
        username: str = "",
        role: str = "",
        ip_address: str = "",
        status: str = "success",
        details: dict[str, Any] | None = None,
        request_id: str = "",
    ) -> None:
        self.id = uuid.uuid4().hex[:16]
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.action = action
        self.resource = resource
        self.user_id = user_id
        self.username = username
        self.role = role
        self.ip_address = ip_address
        self.status = status
        self.details = details or {}
        self.request_id = request_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "action": self.action,
            "resource": self.resource,
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "ip_address": self.ip_address,
            "status": self.status,
            "details": self.details,
            "request_id": self.request_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditStore:
    """Audit event storage and query (in-memory + optional file)."""

    def __init__(self, config: SecurityConfig) -> None:
        self._config = config
        self._events: list[AuditEvent] = []
        self._file = config.audit_log_file
        if self._file:
            Path(self._file).parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: AuditEvent) -> None:
        """Store an audit event."""
        self._events.append(event)

        # Write to file if configured
        if self._file:
            try:
                with open(self._file, "a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")
            except OSError as e:
                logger.error("Failed to write audit log: %s", e)

        # Trim in-memory to retention limit
        retention = self._config.audit_retention_days
        if retention > 0:
            cutoff = time.time() - retention * 86400
            self._events = [
                e for e in self._events
                if _parse_ts(e.timestamp) > cutoff
            ]

    def query(
        self,
        user_id: str | None = None,
        action: str | None = None,
        resource: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events with filters."""
        results = self._events
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if action:
            results = [e for e in results if e.action == action]
        if resource:
            results = [e for e in results if resource in e.resource]
        if status:
            results = [e for e in results if e.status == status]
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def count_by_action(self, hours: int = 24) -> dict[str, int]:
        """Return {action: count} for the last N hours."""
        cutoff = time.time() - hours * 3600
        counts: dict[str, int] = {}
        for e in self._events:
            if _parse_ts(e.timestamp) > cutoff:
                counts[e.action] = counts.get(e.action, 0) + 1
        return counts


# Global store (set during app initialization)
_global_store: AuditStore | None = None


def get_audit_store() -> AuditStore | None:
    return _global_store


def set_audit_store(store: AuditStore) -> None:
    global _global_store
    _global_store = store


def record_audit_event(
    action: str,
    resource: str,
    user_id: str = "",
    username: str = "",
    role: str = "",
    ip_address: str = "",
    status: str = "success",
    details: dict[str, Any] | None = None,
    request_id: str = "",
) -> AuditEvent:
    """Convenience: create and store an audit event in one call."""
    event = AuditEvent(
        action=action,
        resource=resource,
        user_id=user_id,
        username=username,
        role=role,
        ip_address=ip_address,
        status=status,
        details=details,
        request_id=request_id,
    )
    store = get_audit_store()
    if store is not None:
        store.record(event)
    return event


def _parse_ts(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts).timestamp()
    except (ValueError, TypeError):
        return 0.0
