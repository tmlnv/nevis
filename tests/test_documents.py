import asyncpg
import httpx
import pytest

from src.errors import AIProviderError
from src.schemas import MAX_CONTENT_CHARS
from src.services import MAX_PASSAGE_CHARS, PASSAGE_OVERLAP_CHARS, split_passages
from tests.conftest import FakeAIClient

_STEP = MAX_PASSAGE_CHARS - PASSAGE_OVERLAP_CHARS


def _digits(n: int) -> str:
    return "".join(str(i % 10) for i in range(n))


# --- split_passages ---------------------------------------------------------


def test_short_text_is_one_passage() -> None:
    assert split_passages("Hello there.") == ("Hello there.",)


def test_blank_paragraphs_are_dropped() -> None:
    assert split_passages("\n\n  \n\nonly\n\n \t \n\n") == ("only",)


def test_paragraphs_keep_order_and_all_text() -> None:
    a, b, c = "A" * 500, "B" * 500, "C" * 500
    passages = split_passages(f"{a}\n\n{b}\n\n{c}")

    assert len(passages) == 2
    assert passages == (f"{a}\n\n{b}", c)
    joined = "".join(passages)
    assert joined.index("A") < joined.index("B") < joined.index("C")
    for part in (a, b, c):
        assert part in joined


def test_oversized_paragraph_is_split_into_bounded_overlapping_windows() -> None:
    paragraph = _digits(3000)
    passages = split_passages(paragraph)

    assert len(passages) > 1
    assert all(len(p) <= MAX_PASSAGE_CHARS for p in passages)
    assert all(p in paragraph for p in passages)
    # consecutive windows share PASSAGE_OVERLAP_CHARS of tail/head
    assert passages[1][:PASSAGE_OVERLAP_CHARS] == passages[0][-PASSAGE_OVERLAP_CHARS:]
    assert passages[0] == paragraph[:MAX_PASSAGE_CHARS]
    assert passages[1] == paragraph[_STEP : _STEP + MAX_PASSAGE_CHARS]
    # nothing is lost: the windows minus their overlap rebuild the paragraph
    rebuilt = passages[0] + "".join(p[PASSAGE_OVERLAP_CHARS:] for p in passages[1:])
    assert rebuilt == paragraph


def test_buffered_text_is_flushed_before_an_oversized_paragraph() -> None:
    passages = split_passages(f"short intro\n\n{_digits(2000)}")

    assert passages[0] == "short intro"
    assert len(passages) == 3


# --- POST /clients/{id}/documents -------------------------------------------


async def test_create_document_stores_summary_and_one_chunk_per_passage(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    pool: asyncpg.Pool,
    make_client,
    fake_ai: FakeAIClient,
) -> None:
    owner = await make_client()
    title = "Address proof"
    content = f"{'A' * 900}\n\n{'B' * 900}\n\n{'C' * 900}"

    response = await client.post(
        f"/clients/{owner['id']}/documents",
        json={"title": title, "content": content},
        headers=auth_headers,
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["client_id"] == owner["id"]
    assert body["title"] == title
    assert body["summary"].startswith("Summary of:")

    expected = (title, *split_passages(content))
    rows = await pool.fetch(
        "SELECT position, content FROM document_chunks WHERE document_id = $1 ORDER BY position",
        body["id"],
    )
    assert [r["content"] for r in rows] == list(expected)
    assert [r["position"] for r in rows] == list(range(len(expected)))

    document = await pool.fetchrow("SELECT * FROM documents WHERE id = $1", body["id"])
    assert document["summary"] == body["summary"]
    assert document["embedding_model"] == "fake-embedding-model"
    assert document["summary_model"] == "fake-summary-model"
    assert fake_ai.embed_calls == 1
    assert fake_ai.summarize_calls == 1


async def test_unknown_client_returns_404_without_touching_the_ai_gateway(
    client: httpx.AsyncClient, auth_headers: dict[str, str], fake_ai: FakeAIClient
) -> None:
    response = await client.post(
        "/clients/00000000-0000-0000-0000-000000000000/documents",
        json={"title": "Address proof", "content": "Some content."},
        headers=auth_headers,
    )

    assert response.status_code == 404
    assert response.json()["code"] == "client_not_found"
    assert fake_ai.calls == 0


@pytest.mark.parametrize("failing_call", ["embed", "summarize"])
async def test_provider_failure_is_attributed_and_persists_nothing(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    pool: asyncpg.Pool,
    make_client,
    fake_ai: FakeAIClient,
    failing_call: str,
) -> None:
    owner = await make_client()
    setattr(
        fake_ai,
        f"{failing_call}_error",
        AIProviderError(
            code="ai_provider_unavailable",
            message="The OpenRouter AI provider is temporarily unavailable.",
            status=503,
            upstream_status=502,
            is_retryable=True,
        ),
    )

    response = await client.post(
        f"/clients/{owner['id']}/documents",
        json={"title": "Address proof", "content": "Some content."},
        headers=auth_headers,
    )

    assert response.status_code in (502, 503)
    body = response.json()
    assert body["code"] == "ai_provider_unavailable"
    assert body["provider"] == "openrouter"
    assert body["upstream_status"] == 502
    assert body["is_retryable"] is True
    assert body["request_id"]

    assert await pool.fetchval("SELECT count(*) FROM documents") == 0
    assert await pool.fetchval("SELECT count(*) FROM document_chunks") == 0


async def test_content_over_the_cap_is_rejected(
    client: httpx.AsyncClient, auth_headers: dict[str, str], make_client, fake_ai: FakeAIClient
) -> None:
    owner = await make_client()

    response = await client.post(
        f"/clients/{owner['id']}/documents",
        json={"title": "Too long", "content": "x" * (MAX_CONTENT_CHARS + 1)},
        headers=auth_headers,
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert fake_ai.calls == 0
