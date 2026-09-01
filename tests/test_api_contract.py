import httpx

from tests.custom_types import Fixture


async def test_openapi_exposes_every_documented_path(client: Fixture[httpx.AsyncClient]) -> None:
    schema = (await client.get("/openapi.json")).json()

    assert {"/clients", "/clients/{id}/documents", "/search", "/health"} <= set(schema["paths"])
    assert "post" in schema["paths"]["/clients"]
    assert "post" in schema["paths"]["/clients/{id}/documents"]
    assert "get" in schema["paths"]["/search"]


async def test_search_takes_q_and_exposes_no_limit_parameter(
    client: Fixture[httpx.AsyncClient],
) -> None:
    schema = (await client.get("/openapi.json")).json()
    parameters = schema["paths"]["/search"]["get"]["parameters"]
    by_name = {p["name"]: p for p in parameters}

    assert by_name["q"]["in"] == "query"
    assert by_name["q"]["required"] is True
    assert "limit" not in by_name


async def test_bearer_security_scheme_is_declared(client: Fixture[httpx.AsyncClient]) -> None:
    schema = (await client.get("/openapi.json")).json()
    schemes = schema["components"]["securitySchemes"]

    assert any(s["type"] == "http" and s["scheme"].lower() == "bearer" for s in schemes.values())
    for path in ("/clients", "/search"):
        method = "post" if path == "/clients" else "get"
        assert schema["paths"][path][method]["security"]


async def test_health_needs_no_token(client: Fixture[httpx.AsyncClient]) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "is_database_ready": True}
