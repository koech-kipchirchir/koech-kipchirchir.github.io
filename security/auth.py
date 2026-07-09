"""
Authentication endpoints: login, register, token refresh, logout.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import bcrypt

from security.config import SecurityConfig
from security.jwt import (
    create_access_token, create_refresh_token,
    decode_token, refresh_access_token,
    JWTError, JWTExpiredError,
)
from security.models import Role, Session, User

logger = logging.getLogger("aios.security.auth")


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 401) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class UserStore:
    """In-memory user store. Replace with a database adapter for production."""

    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._sessions: dict[str, Session] = {}

    # --- Users ---

    def add_user(self, user: User) -> User:
        if not user.id:
            user.id = uuid.uuid4().hex[:16]
        user.created_at = user.updated_at = datetime.now(timezone.utc).isoformat()
        self._users[user.id] = user
        return user

    def get_user(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        for u in self._users.values():
            if u.username == username:
                return u
        return None

    def get_user_by_email(self, email: str) -> User | None:
        for u in self._users.values():
            if u.email == email:
                return u
        return None

    def update_user(self, user_id: str, **updates: Any) -> User | None:
        user = self._users.get(user_id)
        if user is None:
            return None
        for key, val in updates.items():
            if hasattr(user, key):
                setattr(user, key, val)
        user.updated_at = datetime.now(timezone.utc).isoformat()
        return user

    # --- Sessions ---

    def create_session(
        self,
        user_id: str,
        refresh_token_hash: str,
        ip: str = "",
        user_agent: str = "",
        ttl_hours: int = 24,
    ) -> Session:
        session = Session(
            id=uuid.uuid4().hex[:16],
            user_id=user_id,
            refresh_token_hash=refresh_token_hash,
            ip_address=ip,
            user_agent=user_agent,
            expires_at=datetime.now(timezone.utc).isoformat(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)


# Global stores (set during init)
_user_store = UserStore()


def get_user_store() -> UserStore:
    return _user_store


# ---------------------------------------------------------------------------
# Authentication logic
# ---------------------------------------------------------------------------

def hash_password(password: str, rounds: int = 12) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=rounds)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def register_user(
    username: str,
    password: str,
    email: str = "",
    config: SecurityConfig | None = None,
) -> User:
    """Register a new user. Returns the User (password not included)."""
    cfg = config or SecurityConfig.from_env()

    if len(password) < cfg.password_min_length:
        raise AuthError(
            f"Password must be at least {cfg.password_min_length} characters",
            status_code=400,
        )

    if get_user_store().get_user_by_username(username):
        raise AuthError("Username already taken", status_code=409)

    if email and get_user_store().get_user_by_email(email):
        raise AuthError("Email already registered", status_code=409)

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password, rounds=cfg.bcrypt_rounds),
        role=Role.USER,
    )
    return get_user_store().add_user(user)


def authenticate_user(
    username: str,
    password: str,
    config: SecurityConfig | None = None,
) -> User:
    """Authenticate a user by username + password."""
    cfg = config or SecurityConfig.from_env()
    user = get_user_store().get_user_by_username(username)
    if user is None:
        raise AuthError("Invalid username or password")
    if not user.is_active:
        raise AuthError("Account is disabled", status_code=403)
    if not verify_password(password, user.password_hash):
        raise AuthError("Invalid username or password")
    return user


def login(
    username: str,
    password: str,
    ip: str = "",
    user_agent: str = "",
    config: SecurityConfig | None = None,
) -> dict[str, Any]:
    """Authenticate and return tokens + user info."""
    cfg = config or SecurityConfig.from_env()
    user = authenticate_user(username, password, cfg)

    access_token = create_access_token(user.to_claims(), cfg)
    refresh_token = create_refresh_token(user.id, cfg)

    # Create session
    get_user_store().create_session(
        user_id=user.id,
        refresh_token_hash=hash_password(refresh_token),
        ip=ip,
        user_agent=user_agent,
        ttl_hours=cfg.session_ttl_hours,
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": cfg.jwt_access_token_ttl_minutes * 60,
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
        },
    }


def refresh_tokens(refresh_token: str, config: SecurityConfig) -> dict[str, Any]:
    """Exchange a refresh token for a new token pair."""
    try:
        new_access, new_refresh, claims = refresh_access_token(refresh_token, config)
        return {
            "access_token": new_access,
            "refresh_token": new_refresh,
            "token_type": "bearer",
            "expires_in": config.jwt_access_token_ttl_minutes * 60,
        }
    except JWTExpiredError:
        raise AuthError("Refresh token expired", status_code=401)
    except JWTError as e:
        raise AuthError(str(e), status_code=401)
