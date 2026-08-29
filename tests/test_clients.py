import asyncpg
import httpx

from tests.conftest import TOKEN


async def test_create_client_persists_normalised_data(
    client: httpx.AsyncClient, auth_headers: dict[str, str], pool: asyncpg.Pool
) -> None:
    response = await client.post(
        "/clients",
        json={
            "first_name": "  John  ",
            "last_name": "  Doe  ",
            "email": "John.Doe@NevisWealth.com",
            "description": "Wealth management client.",
            "social_links": ["https://x.com/johndoe"],
        },
        headers=auth_headers,
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["first_name"] == "John"
    assert body["last_name"] == "Doe"
    assert body["email"] == "john.doe@neviswealth.com"
    assert body["social_links"] == ["https://x.com/johndoe"]

    row = await pool.fetchrow("SELECT * FROM clients WHERE id = $1", body["id"])
    assert row["email"] == "john.doe@neviswealth.com"
    assert row["first_name"] == "John"
    assert row["last_name"] == "Doe"


async def test_invalid_email_returns_validation_error(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/clients",
        json={"first_name": "John", "last_name": "Doe", "email": "not-an-email"},
        headers=auth_headers,
    )

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["message"]
    assert any("email" in issue["location"] for issue in body["issues"])


async def test_blank_name_returns_validation_error(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/clients",
        json={"first_name": "   ", "last_name": "Doe", "email": "a@b.com"},
        headers=auth_headers,
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_missing_token_is_rejected_before_database_work(
    client_without_db: httpx.AsyncClient,
) -> None:
    response = await client_without_db.post(
        "/clients", json={"first_name": "John", "last_name": "Doe", "email": "a@b.com"}
    )

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"
    assert response.headers["www-authenticate"] == "Bearer"


async def test_wrong_token_is_rejected_before_database_work(
    client_without_db: httpx.AsyncClient,
) -> None:
    response = await client_without_db.post(
        "/clients",
        json={"first_name": "John", "last_name": "Doe", "email": "a@b.com"},
        headers={"Authorization": f"Bearer {TOKEN}-wrong"},
    )

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


async def test_search_rejects_missing_token(client_without_db: httpx.AsyncClient) -> None:
    assert (await client_without_db.get("/search", params={"q": "x"})).status_code == 401


async def test_search_finds_client_by_company_domain(
    client: httpx.AsyncClient, auth_headers: dict[str, str], make_client
) -> None:
    """Acceptance criterion: `NevisWealth` finds john.doe@neviswealth.com."""
    target = await make_client(email="john.doe@neviswealth.com")
    await make_client(first_name="Jane", last_name="Smith", email="jane.smith@example.org")

    response = await client.get("/search", params={"q": "NevisWealth"}, headers=auth_headers)

    assert response.status_code == 200, response.text
    clients = [r for r in response.json() if r["result_type"] == "client"]
    assert [c["client"]["id"] for c in clients] == [target["id"]]
    assert clients[0]["client"]["email"] == "john.doe@neviswealth.com"
