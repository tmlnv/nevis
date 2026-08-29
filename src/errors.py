"""Expected failures travel as typed exceptions and are mapped to HTTP once, here."""

import logging
import uuid
from dataclasses import dataclass

import orjson
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.schemas import AIProviderErrorOut, ErrorOut, ValidationErrorOut, ValidationIssueOut

log = logging.getLogger("nevis")


class ApplicationError(Exception):
    """Base for every expected failure."""


class ClientNotFoundError(ApplicationError):
    pass


class DatabaseUnavailableError(ApplicationError):
    pass


@dataclass(frozen=True, slots=True)
class AIProviderError(ApplicationError):
    """One provider failure. Never carries an upstream body, prompt, or key."""

    code: str
    message: str
    status: int
    provider: str = "openrouter"
    upstream_status: int | None = None
    is_retryable: bool = False
    retry_after_seconds: int | None = None


def _json(status: int, body: BaseModel, headers: dict[str, str] | None = None) -> Response:
    # orjson rather than ORJSONResponse: current FastAPI deprecates that response
    # class because it serialises route return values natively.
    return Response(
        content=orjson.dumps(body.model_dump()),
        status_code=status,
        media_type="application/json",
        headers=headers,
    )


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> Response:
        issues = [
            ValidationIssueOut(location=".".join(str(p) for p in e["loc"]), message=e["msg"])
            for e in exc.errors()
        ]
        return _json(422, ValidationErrorOut(message="Request validation failed.", issues=issues))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> Response:
        code = "unauthorized" if exc.status_code == 401 else "http_error"
        return _json(
            exc.status_code, ErrorOut(code=code, message=str(exc.detail)), exc.headers or None
        )

    @app.exception_handler(ClientNotFoundError)
    async def _not_found(request: Request, exc: ClientNotFoundError) -> Response:
        return _json(404, ErrorOut(code="client_not_found", message="Client not found."))

    @app.exception_handler(DatabaseUnavailableError)
    async def _db(request: Request, exc: DatabaseUnavailableError) -> Response:
        body = ErrorOut(
            code="database_unavailable",
            message="The database is unavailable.",
            request_id=_request_id(request),
        )
        return _json(503, body)

    @app.exception_handler(AIProviderError)
    async def _provider(request: Request, exc: AIProviderError) -> Response:
        request_id = _request_id(request)
        log.warning(
            "ai provider failure code=%s provider=%s upstream_status=%s request_id=%s",
            exc.code,
            exc.provider,
            exc.upstream_status,
            request_id,
        )
        body = AIProviderErrorOut(
            code=exc.code,
            message=exc.message,
            provider=exc.provider,
            upstream_status=exc.upstream_status,
            is_retryable=exc.is_retryable,
            retry_after_seconds=exc.retry_after_seconds,
            request_id=request_id,
        )
        headers = {"Retry-After": str(exc.retry_after_seconds)} if exc.retry_after_seconds else None
        return _json(exc.status, body, headers)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> Response:
        request_id = _request_id(request)
        log.exception("unhandled error request_id=%s", request_id)
        body = ErrorOut(
            code="internal_error", message="Internal server error.", request_id=request_id
        )
        return _json(500, body)
