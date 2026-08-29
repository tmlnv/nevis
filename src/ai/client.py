"""OpenRouter adapter. Everything upstream is untrusted: bodies are validated, never leaked."""

import asyncio
import math
import random
from collections.abc import Sequence

import httpx
import orjson
from pydantic import BaseModel, ValidationError

from src.ai.schemas import (
    ChatCompletionRequestOut,
    ChatCompletionResponseIn,
    ChatMessageOut,
    EmbeddingRequestOut,
    EmbeddingResponseIn,
)
from src.config import EMBEDDING_DIMENSIONS, Settings
from src.dto import EmbeddingResultDTO, SummaryResultDTO, Vector
from src.errors import AIProviderError

# Transient upstream failures worth one immediate retry.
_RETRY_STATUSES = frozenset({500, 502, 503, 504, 524, 529})
# Upstream is up but not serving us; the caller may try again later.
_UNAVAILABLE_STATUSES = _RETRY_STATUSES | {404, 408}
# We sent something upstream refused; retrying the same body cannot help.
_REJECTED_STATUSES = frozenset({400, 403, 413, 422})

_MAX_RETRY_AFTER_SECONDS = 3600
# Headroom: some free models emit reasoning tokens before any content, and a tight
# budget makes them stop at finish_reason="length" with an empty message.
_SUMMARY_MAX_TOKENS = 400

_SUMMARY_PROMPT = (
    "Summarize the following document in 2-3 sentences. Reply with the summary only.\n\n"
)


def _unavailable(upstream_status: int | None) -> AIProviderError:
    return AIProviderError(
        code="ai_provider_unavailable",
        message="The OpenRouter AI provider is temporarily unavailable.",
        status=503,
        upstream_status=upstream_status,
        is_retryable=True,
    )


def _invalid_response(upstream_status: int | None = None) -> AIProviderError:
    return AIProviderError(
        code="ai_provider_invalid_response",
        message="The OpenRouter AI provider returned an unusable response.",
        status=502,
        upstream_status=upstream_status,
    )


def _retry_after(response: httpx.Response) -> int | None:
    raw = response.headers.get("retry-after", "").strip()
    if not raw.isdigit():
        return None
    return min(int(raw), _MAX_RETRY_AFTER_SECONDS)


def _map_status(response: httpx.Response) -> AIProviderError:
    status = response.status_code
    if status == 401:
        return AIProviderError(
            code="ai_provider_configuration_error",
            message="The OpenRouter AI provider rejected our credentials.",
            status=503,
            upstream_status=status,
        )
    if status == 402:
        return AIProviderError(
            code="ai_provider_quota_exhausted",
            message="The OpenRouter AI provider quota is exhausted.",
            status=503,
            upstream_status=status,
        )
    if status == 429:
        return AIProviderError(
            code="ai_provider_rate_limited",
            message="The OpenRouter AI provider is rate limiting us.",
            status=503,
            upstream_status=status,
            is_retryable=True,
            retry_after_seconds=_retry_after(response),
        )
    if status in _UNAVAILABLE_STATUSES:
        return _unavailable(status)
    if status in _REJECTED_STATUSES:
        return AIProviderError(
            code="ai_provider_rejected_request",
            message="The OpenRouter AI provider rejected the request.",
            status=502,
            upstream_status=status,
        )
    return AIProviderError(
        code="ai_provider_error",
        message="The OpenRouter AI provider call failed.",
        status=502,
        upstream_status=status,
    )


def _parse[M: BaseModel](response: httpx.Response, model: type[M]) -> M:
    try:
        return model.model_validate(orjson.loads(response.content))
    except (orjson.JSONDecodeError, ValidationError) as exc:
        raise _invalid_response(response.status_code) from exc


class AIClient:
    """Embeddings and summaries over an OpenAI-compatible HTTP API."""

    def __init__(self, http: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http
        self._settings = settings

    async def embed(self, texts: Sequence[str]) -> EmbeddingResultDTO:
        body = EmbeddingRequestOut(model=self._settings.embedding_model, input=list(texts))
        response = await self._post(
            self._settings.embeddings_url(), self._settings.openrouter_api_key, body
        )
        parsed = _parse(response, EmbeddingResponseIn)

        rows = sorted(parsed.data, key=lambda row: row.index)
        if len(rows) != len(texts):
            raise _invalid_response(response.status_code)

        vectors: list[Vector] = []
        for row in rows:
            if len(row.embedding) != EMBEDDING_DIMENSIONS:
                raise _invalid_response(response.status_code)
            if not all(math.isfinite(value) for value in row.embedding):
                raise _invalid_response(response.status_code)
            vectors.append(tuple(row.embedding))

        return EmbeddingResultDTO(vectors=tuple(vectors), model=parsed.model)

    async def summarize(self, text: str) -> SummaryResultDTO:
        body = ChatCompletionRequestOut(
            model=self._settings.summary_model,
            messages=[ChatMessageOut(role="user", content=_SUMMARY_PROMPT + text)],
            max_tokens=_SUMMARY_MAX_TOKENS,
            temperature=0.2,
        )
        response = await self._post(
            self._settings.summary_url(), self._settings.summary_key(), body
        )
        parsed = _parse(response, ChatCompletionResponseIn)

        content = parsed.choices[0].message.content if parsed.choices else None
        if content is None or not content.strip():
            raise _invalid_response(response.status_code)

        return SummaryResultDTO(text=content.strip(), model=parsed.model)

    async def _post(self, url: str, key: str, body: BaseModel) -> httpx.Response:
        payload = orjson.dumps(body.model_dump())
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        for is_last in (False, True):
            try:
                response = await self._http.post(
                    url,
                    content=payload,
                    headers=headers,
                    timeout=self._settings.ai_timeout_seconds,
                )
            except httpx.TransportError as exc:
                if is_last:
                    raise _unavailable(None) from exc
                await self._backoff()
                continue

            if response.status_code in _RETRY_STATUSES and not is_last:
                await self._backoff()
                continue
            if response.is_success:
                return response
            raise _map_status(response)

        raise _unavailable(None)  # unreachable: the last pass always returns or raises

    @staticmethod
    async def _backoff() -> None:
        await asyncio.sleep(random.uniform(0.1, 0.4))  # noqa: S311 - jitter, not crypto
