from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from memory.utils import now_utc, setup_logger, timestamp_ms

logger = setup_logger("aios.memory.storage")


class SQLiteStorage:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS memory_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT DEFAULT '',
                importance REAL DEFAULT 0.0,
                embedding BLOB,
                created_at INTEGER NOT NULL,
                accessed_at INTEGER NOT NULL,
                expires_at INTEGER,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                start_message_id INTEGER,
                end_message_id INTEGER,
                created_at INTEGER NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_memory_nodes_session ON memory_nodes(session_id);
            CREATE INDEX IF NOT EXISTS idx_memory_nodes_expires ON memory_nodes(expires_at);
            CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
        """)
        conn.commit()

    # -- Sessions ----------------------------------------------------------------

    def create_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> None:
        now = timestamp_ms()
        with self._lock:
            self._connect().execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at, metadata) VALUES (?, ?, ?, ?)",
                (session_id, now, now, json.dumps(metadata or {})),
            )
            self._connect().commit()

    def update_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> None:
        now = timestamp_ms()
        with self._lock:
            self._connect().execute(
                "UPDATE sessions SET updated_at=?, metadata=? WHERE session_id=?",
                (now, json.dumps(metadata or {}), session_id),
            )
            self._connect().commit()

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute("DELETE FROM summaries WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM memory_nodes WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._connect().execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    # -- Messages ----------------------------------------------------------------

    def add_message(
        self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> int:
        now = timestamp_ms()
        with self._lock:
            cur = self._connect().execute(
                "INSERT INTO messages (session_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, now, json.dumps(metadata or {})),
            )
            self._connect().execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (now, session_id),
            )
            self._connect().commit()
            return cur.lastrowid  # type: ignore[return-value]

    def get_messages(
        self, session_id: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        rows = self._connect().execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_messages(self, session_id: str, n: int = 50) -> list[dict[str, Any]]:
        rows = self._connect().execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
            (session_id, n),
        ).fetchall()
        rows.reverse()
        return [dict(r) for r in rows]

    def delete_old_messages(self, before_timestamp: int) -> int:
        with self._lock:
            cur = self._connect().execute(
                "DELETE FROM messages WHERE timestamp < ?", (before_timestamp,)
            )
            self._connect().commit()
            return cur.rowcount

    # -- Memory Nodes -----------------------------------------------------------

    def add_memory_node(
        self,
        session_id: str,
        content: str,
        importance: float = 0.0,
        summary: str = "",
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_days: int = 90,
    ) -> int:
        now = timestamp_ms()
        expires = now + ttl_days * 86400_000 if ttl_days > 0 else None
        emb_blob = json.dumps(embedding) if embedding else None
        with self._lock:
            cur = self._connect().execute(
                "INSERT INTO memory_nodes (session_id, content, summary, importance, embedding, created_at, accessed_at, expires_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, content, summary, importance, emb_blob, now, now, expires, json.dumps(metadata or {})),
            )
            self._connect().commit()
            return cur.lastrowid  # type: ignore[return-value]

    def get_memory_nodes(
        self, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        rows = self._connect().execute(
            "SELECT * FROM memory_nodes WHERE session_id=? AND (expires_at IS NULL OR expires_at > ?) ORDER BY importance DESC, created_at DESC LIMIT ?",
            (session_id, timestamp_ms(), limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_memory_node_access(self, node_id: int) -> None:
        with self._lock:
            self._connect().execute(
                "UPDATE memory_nodes SET accessed_at=? WHERE id=?",
                (timestamp_ms(), node_id),
            )
            self._connect().commit()

    def delete_expired_nodes(self) -> int:
        now = timestamp_ms()
        with self._lock:
            cur = self._connect().execute(
                "DELETE FROM memory_nodes WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            self._connect().commit()
            return cur.rowcount

    # -- Summaries --------------------------------------------------------------

    def add_summary(
        self,
        session_id: str,
        content: str,
        start_message_id: int = 0,
        end_message_id: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = timestamp_ms()
        with self._lock:
            cur = self._connect().execute(
                "INSERT INTO summaries (session_id, content, start_message_id, end_message_id, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, content, start_message_id, end_message_id, now, json.dumps(metadata or {})),
            )
            self._connect().commit()
            return cur.lastrowid  # type: ignore[return-value]

    def get_summaries(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._connect().execute(
            "SELECT * FROM summaries WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Maintenance ------------------------------------------------------------

    def vacuum(self) -> None:
        with self._lock:
            self._connect().execute("VACUUM")
            self._connect().commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
