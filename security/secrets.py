"""
Secrets management: load secrets from environment, files, or a vault.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from security.encryption import AESGCMEncryptor, EncryptionError

logger = logging.getLogger("aios.security.secrets")


class SecretsProvider:
    """Abstract base for secrets providers."""

    def get(self, key: str, default: str | None = None) -> str | None:
        raise NotImplementedError

    def get_or_raise(self, key: str) -> str:
        val = self.get(key)
        if val is None:
            raise ValueError(f"Required secret not found: {key}")
        return val


class EnvSecretsProvider(SecretsProvider):
    """Load secrets from environment variables."""

    def __init__(self, prefix: str = "AIOS_SECRET_") -> None:
        self._prefix = prefix

    def get(self, key: str, default: str | None = None) -> str | None:
        env_name = f"{self._prefix}{key.upper()}"
        return os.environ.get(env_name, default)


class FileSecretsProvider(SecretsProvider):
    """Load secrets from a JSON file.

    The file should be a flat dict of {key: value}.
    """

    def __init__(self, filepath: str | Path) -> None:
        self._path = Path(filepath)
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = self._path.read_text()
                self._data = json.loads(raw)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load secrets file %s: %s", self._path, e)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._data.get(key, default)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self._path.write_text(json.dumps(self._data, indent=2))

    def reload(self) -> None:
        self._load()


class VaultSecretsProvider(SecretsProvider):
    """Load secrets from HashiCorp Vault KV store.

    Requires ``hvac`` package.
    """

    def __init__(self, url: str, token: str, mount_point: str = "secret") -> None:
        self._url = url
        self._token = token
        self._mount_point = mount_point
        self._client: Any = None
        self._cache: dict[str, str] = {}

    def _connect(self) -> None:
        if self._client is not None:
            return
        try:
            import hvac
            self._client = hvac.Client(url=self._url, token=self._token)
            if not self._client.is_authenticated():
                raise ConnectionError("Vault authentication failed")
        except ImportError:
            raise ImportError("hvac package required for Vault support: pip install hvac")

    def get(self, key: str, default: str | None = None) -> str | None:
        if key in self._cache:
            return self._cache[key]
        self._connect()
        try:
            secret = self._client.secrets.kv.v1.read_secret(
                path=key, mount_point=self._mount_point,
            )
            val = secret.get("data", {}).get("data", {}).get(key)
            if val:
                self._cache[key] = str(val)
            return str(val) if val else default
        except Exception:
            return default


class ChainedSecretsProvider(SecretsProvider):
    """Try multiple providers in order, returning the first hit."""

    def __init__(self, providers: list[SecretsProvider]) -> None:
        self._providers = providers

    def get(self, key: str, default: str | None = None) -> str | None:
        for p in self._providers:
            val = p.get(key)
            if val is not None:
                return val
        return default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Global secrets provider (set during app startup)
_global_provider: SecretsProvider | None = None


def set_global_provider(provider: SecretsProvider) -> None:
    global _global_provider
    _global_provider = provider


def get_secret(key: str, default: str | None = None) -> str | None:
    if _global_provider is not None:
        return _global_provider.get(key, default)
    return os.environ.get(key, default)


def require_secret(key: str) -> str:
    val = get_secret(key)
    if val is None:
        raise ValueError(f"Required secret not found: {key}")
    return val


def create_default_provider(secrets_file: str | Path | None = None) -> ChainedSecretsProvider:
    """Create a chained provider: env vars, then optional file, then Vault."""
    providers: list[SecretsProvider] = [EnvSecretsProvider()]
    if secrets_file:
        providers.append(FileSecretsProvider(secrets_file))
    try:
        vault_addr = os.environ.get("VAULT_ADDR")
        vault_token = os.environ.get("VAULT_TOKEN")
        if vault_addr and vault_token:
            providers.append(VaultSecretsProvider(vault_addr, vault_token))
    except ImportError:
        pass
    return ChainedSecretsProvider(providers)
