from __future__ import annotations

from typing import Any


class APIError(Exception):
    status_code: int = 500
    detail: str = "Internal server error"
    error_code: str = "internal_error"

    def __init__(
        self,
        detail: str | None = None,
        error_code: str | None = None,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if detail is not None:
            self.detail = detail
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code
        self.headers = headers or {}
        super().__init__(self.detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.error_code,
                "message": self.detail,
                "status": self.status_code,
            }
        }


class ServiceUnavailableError(APIError):
    status_code = 503
    detail = "Service temporarily unavailable"
    error_code = "service_unavailable"


class RateLimitError(APIError):
    status_code = 429
    detail = "Too many requests"
    error_code = "rate_limit_exceeded"

    def __init__(
        self,
        retry_after: int = 60,
        detail: str | None = None,
    ) -> None:
        super().__init__(detail=detail)
        self.retry_after = retry_after
        self.headers["Retry-After"] = str(retry_after)


class EngineError(APIError):
    status_code = 502
    detail = "AI engine error"
    error_code = "engine_error"


class ModelNotReadyError(APIError):
    status_code = 503
    detail = "Model is not ready yet"
    error_code = "model_not_ready"


class BadRequestError(APIError):
    status_code = 400
    detail = "Bad request"
    error_code = "bad_request"


class NotFoundError(APIError):
    status_code = 404
    detail = "Resource not found"
    error_code = "not_found"


class ConflictError(APIError):
    status_code = 409
    detail = "Resource conflict"
    error_code = "conflict"


class ValidationError(APIError):
    status_code = 422
    detail = "Validation error"
    error_code = "validation_error"

    def __init__(
        self,
        errors: list[dict[str, Any]] | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(detail=detail)
        self.errors = errors or []

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["error"]["details"] = self.errors
        return base
