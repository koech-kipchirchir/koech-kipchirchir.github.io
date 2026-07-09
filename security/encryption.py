"""
Encryption utilities: AES-256-GCM, Fernet, key derivation.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from base64 import urlsafe_b64decode, urlsafe_b64encode
from typing import Any, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger("aios.security.encryption")


class EncryptionError(Exception):
    pass


def derive_key(master_key: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Derive a 256-bit AES key from a master key using PBKDF2.

    Returns (key_bytes, salt). Salt is generated if not provided.
    """
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        master_key.encode(),
        salt,
        600_000,  # OWASP recommended iterations
        dklen=32,
    )
    return key, salt


class AESGCMEncryptor:
    """AES-256-GCM encryption/decryption.

    Produces ciphertext with the nonce prepended (12 bytes).
    Output is base64-url-safe encoded.
    """

    def __init__(self, key_hex: str) -> None:
        key_bytes = bytes.fromhex(key_hex) if len(key_hex) == 64 else key_hex.encode()
        if len(key_bytes) != 32:
            key_bytes, _ = derive_key(key_hex)
        self._aesgcm = AESGCM(key_bytes)

    def encrypt(self, plaintext: str, associated_data: bytes | None = None) -> str:
        """Encrypt plaintext. Returns base64-url-safe string."""
        nonce = os.urandom(12)
        ct = self._aesgcm.encrypt(nonce, plaintext.encode(), associated_data or b"")
        return urlsafe_b64encode(nonce + ct).decode()

    def decrypt(self, ciphertext_b64: str, associated_data: bytes | None = None) -> str:
        """Decrypt base64-url-safe ciphertext. Returns plaintext string."""
        try:
            raw = urlsafe_b64decode(ciphertext_b64)
        except Exception as e:
            raise EncryptionError(f"Invalid base64: {e}")
        if len(raw) < 12:
            raise EncryptionError("Ciphertext too short")
        nonce = raw[:12]
        ct = raw[12:]
        try:
            plaintext = self._aesgcm.decrypt(nonce, ct, associated_data or b"")
            return plaintext.decode()
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}")


class FernetEncryptor:
    """Symmetric encryption using Fernet (AES-128-CBC + HMAC)."""

    def __init__(self, key_hex: str) -> None:
        key_bytes = bytes.fromhex(key_hex) if len(key_hex) == 64 else key_hex.encode()
        if len(key_bytes) != 32:
            key_bytes, _ = derive_key(key_hex)
        # Fernet needs a 32-byte URL-safe base64-encoded key
        self._fernet = Fernet(urlsafe_b64encode(key_bytes))

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except Exception as e:
            raise EncryptionError(f"Fernet decryption failed: {e}")


def generate_encryption_key() -> str:
    """Generate a 256-bit hex-encoded encryption key."""
    return secrets.token_hex(32)
