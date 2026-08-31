"""Every request is served by httpx.MockTransport; nothing here touches the network."""

from collections.abc import Callable

import httpx
import orjson
import pytest

from src.ai.client import AIClient
from src.config import EMBEDDING_DIMENSIONS, Settings
from src.errors import AIProviderError
from tests.conftest import Fixture

VECTOR = [0.01] * EMBEDDING_DIMENSIONS
OK_EMBEDDING = {"model": "test-model", "data": [{"index": 0, "embedding": VECTOR}]}
OK_SUMMARY = {"model": "test-model", "choices": [{"message": {"content": " done "}}]}


@pytest.fixture
def settings() -> Fixture[Settings]:
    return Settings(
        _env_file=None,
        database_url="postgres://nevis:nevis@127.0.0.1:1/nevis",
        api_bearer_token="irrelevant",
        openrouter_api_key="not-a-real-key",
        ai_timeout_seconds=1.0,
    )


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> Fixture[None]:
    async def _noop(attempt: int = 0) -> None:
        return None

    monkeypatch.setattr(AIClient, "_backoff", staticmethod(_noop))


@pytest.fixture
def build(settings: Settings) -> Fixture[Callable[..., tuple[AIClient, list[httpx.Request]]]]:
    def _build(handler: Callable[[httpx.Request], httpx.Response]):
        seen: list[httpx.Request] = []

        def _record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        http = httpx.AsyncClient(transport=httpx.MockTransport(_record))
        return AIClient(http, settings), seen

    return _build


def _responder(*responses: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    queue = list(responses)

    def _handler(request: httpx.Request) -> httpx.Response:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return _handler


def _json(status: int, payload: object, **kwargs: object) -> httpx.Response:
    return httpx.Response(status, content=orjson.dumps(payload), **kwargs)


async def test_happy_path_returns_vectors_and_model(build) -> None:
    ai, seen = build(_responder(_json(200, OK_EMBEDDING)))

    result = await ai.embed(["hello"])

    assert result.model == "test-model"
    assert len(result.vectors) == 1
    assert len(result.vectors[0]) == EMBEDDING_DIMENSIONS
    assert len(seen) == 1
    assert seen[0].headers["authorization"] == "Bearer not-a-real-key"


async def test_summarize_strips_and_returns_content(build) -> None:
    ai, _ = build(_responder(_json(200, OK_SUMMARY)))

    assert (await ai.summarize("text")).text == "done"


@pytest.mark.parametrize(
    ("status", "code", "mapped", "is_retryable", "attempts"),
    [
        (401, "ai_provider_configuration_error", 503, False, 1),
        (402, "ai_provider_quota_exhausted", 503, False, 1),
        (404, "ai_provider_unavailable", 503, True, 1),
        (400, "ai_provider_rejected_request", 502, False, 1),
        (500, "ai_provider_unavailable", 503, True, 3),
    ],
)
async def test_status_codes_map_to_documented_errors(
    build, status: int, code: str, mapped: int, is_retryable: bool, attempts: int
) -> None:
    ai, seen = build(_responder(_json(status, {"error": "upstream detail"})))

    with pytest.raises(AIProviderError) as exc_info:
        await ai.embed(["hello"])

    error = exc_info.value
    assert (error.code, error.status, error.is_retryable) == (code, mapped, is_retryable)
    assert error.upstream_status == status
    assert error.provider == "openrouter"
    assert "upstream detail" not in error.message
    assert len(seen) == attempts


async def test_rate_limit_carries_retry_after_and_is_not_retried(build) -> None:
    ai, seen = build(_responder(_json(429, {"error": "slow down"}, headers={"Retry-After": "42"})))

    with pytest.raises(AIProviderError) as exc_info:
        await ai.embed(["hello"])

    error = exc_info.value
    assert error.code == "ai_provider_rate_limited"
    assert error.status == 503
    assert error.is_retryable is True
    assert error.retry_after_seconds == 42
    assert len(seen) == 1


async def test_non_numeric_retry_after_is_ignored(build) -> None:
    ai, _ = build(
        _responder(_json(429, {}, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}))
    )

    with pytest.raises(AIProviderError) as exc_info:
        await ai.embed(["hello"])

    assert exc_info.value.retry_after_seconds is None


async def test_absurd_retry_after_is_clamped(build) -> None:
    ai, _ = build(_responder(_json(429, {}, headers={"Retry-After": "99999999"})))

    with pytest.raises(AIProviderError) as exc_info:
        await ai.embed(["hello"])

    assert exc_info.value.retry_after_seconds == 3600


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"unexpected": True}, id="schema-mismatch"),
        pytest.param(
            {"model": "m", "data": [{"index": 0, "embedding": [0.1] * 10}]}, id="wrong-dimension"
        ),
        pytest.param({"model": "m", "data": []}, id="row-count-mismatch"),
    ],
)
async def test_unusable_success_body_maps_to_invalid_response(build, payload: dict) -> None:
    ai, _ = build(_responder(_json(200, payload)))

    with pytest.raises(AIProviderError) as exc_info:
        await ai.embed(["hello"])

    error = exc_info.value
    assert error.code == "ai_provider_invalid_response"
    assert error.status == 502
    assert error.is_retryable is False
    assert error.upstream_status == 200


async def test_non_finite_vector_maps_to_invalid_response(build) -> None:
    body = '{"model":"m","data":[{"index":0,"embedding":[' + "1e400," * 2047 + "1e400]}]}"
    ai, _ = build(_responder(httpx.Response(200, content=body.encode())))

    with pytest.raises(AIProviderError) as exc_info:
        await ai.embed(["hello"])

    assert exc_info.value.code == "ai_provider_invalid_response"


async def test_malformed_json_maps_to_invalid_response(build) -> None:
    ai, _ = build(_responder(httpx.Response(200, content=b"not json at all")))

    with pytest.raises(AIProviderError) as exc_info:
        await ai.embed(["hello"])

    assert exc_info.value.code == "ai_provider_invalid_response"


async def test_empty_summary_maps_to_invalid_response(build) -> None:
    ai, _ = build(
        _responder(_json(200, {"model": "m", "choices": [{"message": {"content": "  "}}]}))
    )

    with pytest.raises(AIProviderError) as exc_info:
        await ai.summarize("text")

    assert exc_info.value.code == "ai_provider_invalid_response"


async def test_transport_errors_retry_once_then_report_unavailable(build) -> None:
    seen: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise httpx.ConnectError("boom", request=request)

    ai, _ = build(_handler)

    with pytest.raises(AIProviderError) as exc_info:
        await ai.embed(["hello"])

    error = exc_info.value
    assert error.code == "ai_provider_unavailable"
    assert error.status == 503
    assert error.is_retryable is True
    assert error.upstream_status is None
    assert len(seen) == 3


async def test_retryable_status_succeeds_on_the_second_attempt(build) -> None:
    ai, seen = build(_responder(_json(503, {}), _json(200, OK_EMBEDDING)))

    result = await ai.embed(["hello"])

    assert len(result.vectors) == 1
    assert len(seen) == 2


async def test_ok_status_with_error_envelope_is_retried_then_mapped(build) -> None:
    """OpenRouter returns `200 {"error": {"code": 502}}` when its upstream is overloaded."""
    overloaded = _json(200, {"error": {"message": "Service temporarily overloaded", "code": 502}})
    ai, seen = build(_responder(overloaded, _json(200, OK_EMBEDDING)))

    result = await ai.embed(["hello"])

    assert len(result.vectors) == 1
    assert len(seen) == 2


async def test_persistent_error_envelope_reports_the_inner_status(build) -> None:
    ai, seen = build(_responder(_json(200, {"error": {"message": "nope", "code": 502}})))

    with pytest.raises(AIProviderError) as exc_info:
        await ai.embed(["hello"])

    error = exc_info.value
    assert error.code == "ai_provider_unavailable"
    assert error.upstream_status == 502
    assert error.is_retryable is True
    assert "nope" not in error.message
    assert len(seen) == 3
