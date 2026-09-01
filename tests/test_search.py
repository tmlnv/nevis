import asyncpg
import httpx
import pytest

from src.config import Settings
from src.dto import SearchQueryDTO
from src.errors import AIProviderError
from src.repo import ClientRepo, DocumentRepo
from src.services import EMBEDDING_CACHE_SIZE, SearchService, split_passages
from tests.conftest import FakeAIClient, fake_vector
from tests.custom_types import Factory, Fixture

# The fake embedder scores by word overlap, so these land at roughly 1.0, 0.8 and 0.0
# against "address proof" whatever the configured thresholds are.
RELEVANT = "address proof " * 20
PARTLY_RELEVANT = "address proof context " * 20
UNRELATED = "banana bread recipe with ripe fruit sugar butter and a warm oven"

# Two paragraphs long enough that split_passages keeps them as separate chunks,
# both above the document threshold, with B the better match.
CHUNK_A = "alphamarker " + "address proof context " * 33
CHUNK_B = "betamarker " + "address proof " * 55


async def test_relevant_documents_rank_above_unrelated_ones(
    client: Fixture[httpx.AsyncClient],
    auth_headers: Fixture[dict[str, str]],
    make_client: Fixture[Factory],
    make_document: Fixture[Factory],
) -> None:
    owner = await make_client()
    strong = await make_document(owner["id"], "Address proof", RELEVANT)
    weak = await make_document(owner["id"], "Onboarding note", PARTLY_RELEVANT)
    unrelated = await make_document(owner["id"], "Kitchen notes", UNRELATED)

    response = await client.get("/search", params={"q": "address proof"}, headers=auth_headers)

    assert response.status_code == 200, response.text
    documents = [r for r in response.json() if r["result_type"] == "document"]
    ids = [r["document"]["id"] for r in documents]
    assert unrelated["id"] not in ids
    assert ids.index(strong["id"]) < ids.index(weak["id"])
    assert documents[0]["score"] > documents[1]["score"]


async def test_many_matching_chunks_collapse_to_one_result(
    client: Fixture[httpx.AsyncClient],
    auth_headers: Fixture[dict[str, str]],
    pool: Fixture[asyncpg.Pool],
    settings: Fixture[Settings],
    make_client: Fixture[Factory],
    make_document: Fixture[Factory],
) -> None:
    owner = await make_client()
    document = await make_document(owner["id"], "Client file", f"{CHUNK_A}\n\n{CHUNK_B}")
    assert len(split_passages(f"{CHUNK_A}\n\n{CHUNK_B}")) == 2

    # Both chunks must clear the threshold, or "collapse" would be tested vacuously.
    query_vector = "[" + ",".join(map(str, fake_vector("address proof"))) + "]"
    scores = await pool.fetch(
        """SELECT 1 - (embedding <=> $1::halfvec(2048)) AS score
           FROM document_chunks WHERE position > 0""",
        query_vector,
    )
    assert len(scores) == 2
    assert all(r["score"] > settings.document_score_threshold for r in scores)

    response = await client.get("/search", params={"q": "address proof"}, headers=auth_headers)

    hits = [r for r in response.json() if r["result_type"] == "document"]
    assert [h["document"]["id"] for h in hits] == [document["id"]]
    assert "betamarker" in hits[0]["matched_excerpt"]
    assert "alphamarker" not in hits[0]["matched_excerpt"]


async def test_clients_and_documents_share_one_flat_array(
    client: Fixture[httpx.AsyncClient],
    auth_headers: Fixture[dict[str, str]],
    make_client: Fixture[Factory],
    make_document: Fixture[Factory],
) -> None:
    owner = await make_client(description="Requires an address proof for onboarding.")
    await make_document(owner["id"], "Address proof", RELEVANT)

    response = await client.get("/search", params={"q": "address proof"}, headers=auth_headers)

    assert response.status_code == 200
    assert "x-search-degraded" not in response.headers
    body = response.json()
    assert isinstance(body, list)
    assert {r["result_type"] for r in body} == {"client", "document"}
    assert all(r["result_type"] in ("client", "document") for r in body)
    assert all(isinstance(r["score"], int | float) for r in body)


@pytest.mark.settings(search_result_limit=3)
async def test_total_results_are_capped_at_the_configured_limit(
    client: Fixture[httpx.AsyncClient],
    auth_headers: Fixture[dict[str, str]],
    make_client: Fixture[Factory],
    make_document: Fixture[Factory],
) -> None:
    for i in range(2):
        owner = await make_client(
            email=f"person{i}@example.com", description="Holds an address proof on file."
        )
        for j in range(3):
            await make_document(owner["id"], f"Address proof {i}-{j}", RELEVANT)

    response = await client.get("/search", params={"q": "address proof"}, headers=auth_headers)

    body = response.json()
    assert len(body) == 3
    assert sum(r["result_type"] == "client" for r in body) == 2


async def test_search_degrades_to_client_results_when_embeddings_fail(
    client: Fixture[httpx.AsyncClient],
    auth_headers: Fixture[dict[str, str]],
    make_client: Fixture[Factory],
    make_document: Fixture[Factory],
    fake_ai: Fixture[FakeAIClient],
) -> None:
    owner = await make_client(description="Requires an address proof for onboarding.")
    await make_document(owner["id"], "Address proof", RELEVANT)
    fake_ai.embed_error = AIProviderError(
        code="ai_provider_rate_limited",
        message="rate limited",
        status=503,
        upstream_status=429,
        is_retryable=True,
    )

    response = await client.get("/search", params={"q": "address proof"}, headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["x-search-degraded"] == "true"
    body = response.json()
    assert body
    assert {r["result_type"] for r in body} == {"client"}


@pytest.mark.parametrize("q", ["", "   ", "\t\n"])
async def test_blank_query_is_rejected(
    client: Fixture[httpx.AsyncClient], auth_headers: Fixture[dict[str, str]], q: str
) -> None:
    response = await client.get("/search", params={"q": q}, headers=auth_headers)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


@pytest.mark.settings(search_result_limit=2)
async def test_a_many_chunk_document_does_not_starve_other_documents(
    client: Fixture[httpx.AsyncClient],
    auth_headers: Fixture[dict[str, str]],
    make_client: Fixture[Factory],
    make_document: Fixture[Factory],
) -> None:
    owner = await make_client(email="chunky@example.org")
    # Five perfect-match chunks: with a limit-sized pool they fill it and hide `weaker`.
    crowded = await make_document(owner["id"], "Address proof", "\n\n".join([RELEVANT] * 5))
    weaker = await make_document(owner["id"], "Onboarding note", PARTLY_RELEVANT)

    response = await client.get("/search", params={"q": "address proof"}, headers=auth_headers)

    assert response.status_code == 200, response.text
    ids = [r["document"]["id"] for r in response.json() if r["result_type"] == "document"]
    assert ids == [crowded["id"], weaker["id"]]


async def test_a_repeated_query_is_embedded_once(
    client: Fixture[httpx.AsyncClient],
    auth_headers: Fixture[dict[str, str]],
    make_client: Fixture[Factory],
    fake_ai: Fixture[FakeAIClient],
) -> None:
    await make_client(description="Requires an address proof for onboarding.")

    for q in ("address proof", "  address   proof "):
        response = await client.get("/search", params={"q": q}, headers=auth_headers)
        assert response.status_code == 200, response.text

    assert fake_ai.embed_calls == 1  # the second differs only in normalised whitespace


async def test_the_embedding_cache_evicts_the_least_recently_used_query(
    pool: Fixture[asyncpg.Pool],
    settings: Fixture[Settings],
    fake_ai: Fixture[FakeAIClient],
) -> None:
    service = SearchService(ClientRepo(pool), DocumentRepo(pool), fake_ai, settings)  # type: ignore[arg-type]

    async def search(q: str) -> None:
        await service.search(SearchQueryDTO(q=q))

    await search("oldest")
    for i in range(EMBEDDING_CACHE_SIZE - 1):
        await search(f"filler {i}")
    assert fake_ai.embed_calls == EMBEDDING_CACHE_SIZE

    await search("oldest")  # cached, and now most recently used
    assert fake_ai.embed_calls == EMBEDDING_CACHE_SIZE

    await search("one too many")  # evicts "filler 0"
    await search("oldest")
    assert fake_ai.embed_calls == EMBEDDING_CACHE_SIZE + 1

    await search("filler 0")
    assert fake_ai.embed_calls == EMBEDDING_CACHE_SIZE + 2


async def test_a_title_only_match_finds_the_document(
    client: Fixture[httpx.AsyncClient],
    auth_headers: Fixture[dict[str, str]],
    make_client: Fixture[Factory],
    make_document: Fixture[Factory],
) -> None:
    owner = await make_client()
    document = await make_document(owner["id"], "Passport photocopy", UNRELATED)

    response = await client.get("/search", params={"q": "passport photocopy"}, headers=auth_headers)

    hits = [r for r in response.json() if r["result_type"] == "document"]
    assert [h["document"]["id"] for h in hits] == [document["id"]]
    assert hits[0]["matched_excerpt"] == "Passport photocopy"
