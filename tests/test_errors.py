import httpx
import pytest

from tests.custom_types import Factory, Fixture


async def test_health_reports_an_unreachable_database_without_failing(
    client_without_db: Fixture[httpx.AsyncClient],
) -> None:
    response = await client_without_db.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "is_database_ready": False}


async def test_an_unreachable_database_maps_to_503(
    client_without_db: Fixture[httpx.AsyncClient], auth_headers: Fixture[dict[str, str]]
) -> None:
    response = await client_without_db.post(
        "/clients",
        json={"first_name": "John", "last_name": "Doe", "email": "a@b.com"},
        headers=auth_headers,
    )

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "database_unavailable"
    assert "a@b.com" not in body["message"]
    assert body["request_id"]


async def test_a_supplied_request_id_is_echoed_back(
    client_without_db: Fixture[httpx.AsyncClient], auth_headers: Fixture[dict[str, str]]
) -> None:
    response = await client_without_db.get(
        "/search", params={"q": "anything"}, headers={**auth_headers, "X-Request-ID": "abc-123"}
    )

    assert response.status_code == 503
    assert response.json()["request_id"] == "abc-123"


async def test_an_unexpected_error_becomes_a_500_that_leaks_nothing(
    client: Fixture[httpx.AsyncClient],
    auth_headers: Fixture[dict[str, str]],
    make_client: Fixture[Factory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await make_client()

    def _boom(text: str) -> tuple[str, ...]:
        raise ValueError("secret internal detail")

    monkeypatch.setattr("src.services.split_passages", _boom)

    response = await client.post(
        f"/clients/{owner['id']}/documents",
        json={"title": "Address proof", "content": "Some content."},
        headers=auth_headers,
    )

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert body["message"] == "Internal server error."
    assert "secret internal detail" not in response.text
    assert body["request_id"]
