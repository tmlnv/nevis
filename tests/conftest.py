"""Shared fixtures. The real AIClient is never constructed: the container gets FakeAIClient."""

import asyncio
import contextlib
import math
import os
import re
import socket
import subprocess
import time
import zlib
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import asyncpg
import httpx
import pytest
from dishka import Provider, Scope, make_async_container, provide

from src.ai.client import AIClient
from src.config import EMBEDDING_DIMENSIONS, Settings
from src.di import AppProvider, RequestProvider
from src.dto import EmbeddingResultDTO, SummaryResultDTO, Vector
from src.errors import AIProviderError
from src.main import create_app
from tests.custom_types import Factory, JsonValue

TOKEN = "test-bearer-token"
ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "db" / "migrations"
IMAGE = "pgvector/pgvector:pg17"
TABLES = ("document_chunks", "documents", "clients")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "settings(**kwargs): override Settings for this test")


# --- fake AI gateway --------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def fake_vector(text: str) -> Vector:
    """Bag-of-words hashed into 2048 buckets, L2-normalised: overlap == cosine score."""
    buckets = [0.0] * EMBEDDING_DIMENSIONS
    for word in _WORD.findall(text.lower()):
        buckets[zlib.crc32(word.encode()) % EMBEDDING_DIMENSIONS] += 1.0
    norm = math.sqrt(sum(v * v for v in buckets))
    if norm == 0.0:
        buckets[0] = 1.0
        return tuple(buckets)
    return tuple(v / norm for v in buckets)


class FakeAIClient:
    def __init__(self) -> None:
        self.embed_calls = 0
        self.summarize_calls = 0
        self.embed_error: AIProviderError | None = None
        self.summarize_error: AIProviderError | None = None

    @property
    def calls(self) -> int:
        return self.embed_calls + self.summarize_calls

    async def embed(self, texts: Sequence[str]) -> EmbeddingResultDTO:
        self.embed_calls += 1
        if self.embed_error is not None:
            raise self.embed_error
        return EmbeddingResultDTO(
            vectors=tuple(fake_vector(t) for t in texts), model="fake-embedding-model"
        )

    async def summarize(self, text: str) -> SummaryResultDTO:
        self.summarize_calls += 1
        if self.summarize_error is not None:
            raise self.summarize_error
        return SummaryResultDTO(
            text=f"Summary of: {' '.join(text.split())[:120]}", model="fake-summary-model"
        )


class OverrideProvider(Provider):
    scope = Scope.APP

    def __init__(self, settings: Settings, ai: FakeAIClient) -> None:
        super().__init__()
        self._settings = settings
        self._ai = ai

    @provide(override=True)
    def settings(self) -> Settings:
        return self._settings

    @provide(override=True)
    def ai(self) -> AIClient:
        return self._ai  # type: ignore[return-value]


# --- throwaway postgres -----------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _migration_sql() -> str:
    parts = []
    for path in sorted(MIGRATIONS.glob("*.sql")):
        up = path.read_text().split("-- migrate:down")[0]
        parts.append(up)
    return "\n".join(parts)


async def _apply_migrations(dsn: str, sql: str) -> None:
    deadline = time.monotonic() + 60
    while True:
        try:
            conn = await asyncpg.connect(dsn)
            break
        except (OSError, asyncpg.PostgresError):
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.5)
    try:
        # A reused TEST_DATABASE_URL is already migrated; re-running would fail.
        if await conn.fetchval("SELECT to_regclass('public.clients')") is None:
            await conn.execute(sql)
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def pg_dsn(request: pytest.FixtureRequest) -> str:
    if dsn := os.environ.get("TEST_DATABASE_URL"):
        asyncio.run(_apply_migrations(dsn, _migration_sql()))
        return dsn

    port = _free_port()
    container = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "-e",
            "POSTGRES_USER=nevis",
            "-e",
            "POSTGRES_PASSWORD=nevis",
            "-e",
            "POSTGRES_DB=nevis",
            "-p",
            f"{port}:5432",
            IMAGE,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    ).stdout.strip()
    request.addfinalizer(
        lambda: subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    )

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        probe = subprocess.run(
            ["docker", "exec", container, "pg_isready", "-U", "nevis", "-d", "nevis", "-q"],
            capture_output=True,
        )
        if probe.returncode == 0:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("postgres container never became ready")

    dsn = f"postgres://nevis:nevis@127.0.0.1:{port}/nevis"
    asyncio.run(_apply_migrations(dsn, _migration_sql()))
    return dsn


@pytest.fixture
async def pool(pg_dsn: str) -> AsyncIterator[asyncpg.Pool]:
    pool = await asyncpg.create_pool(pg_dsn, min_size=1, max_size=3)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture(autouse=True)
async def _clean_tables(pool: asyncpg.Pool) -> AsyncIterator[None]:
    await pool.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
    yield


# --- application ------------------------------------------------------------


@pytest.fixture
def settings_overrides(request: pytest.FixtureRequest) -> dict[str, JsonValue]:
    marker = request.node.get_closest_marker("settings")
    return dict(marker.kwargs) if marker else {}


@pytest.fixture
def settings(pg_dsn: str, settings_overrides: dict[str, JsonValue]) -> Settings:
    # _env_file=None: the suite must not inherit a developer's tuned .env.
    return Settings(
        _env_file=None,
        database_url=pg_dsn,
        api_bearer_token=TOKEN,
        openrouter_api_key="not-a-real-key",
        **settings_overrides,
    )


@pytest.fixture
def fake_ai() -> FakeAIClient:
    return FakeAIClient()


@pytest.fixture
async def client(settings: Settings, fake_ai: FakeAIClient) -> AsyncIterator[httpx.AsyncClient]:
    async for c in _client_for(settings, fake_ai):
        yield c


@pytest.fixture
async def client_without_db(fake_ai: FakeAIClient) -> AsyncIterator[httpx.AsyncClient]:
    """Same app, but the DSN points nowhere: any database work fails loudly."""
    broken = Settings(
        _env_file=None,
        database_url=f"postgres://nevis:nevis@127.0.0.1:{_free_port()}/nevis",
        api_bearer_token=TOKEN,
        openrouter_api_key="not-a-real-key",
    )
    async for c in _client_for(broken, fake_ai):
        yield c


async def _client_for(settings: Settings, ai: FakeAIClient) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    container = make_async_container(
        AppProvider(), RequestProvider(), OverrideProvider(settings, ai)
    )
    # setup_dishka() already installed the middleware; swapping the container is all
    # that is left, and it is exactly what setup_dishka's last line does.
    app.state.dishka_container = container
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        try:
            yield http
        finally:
            with contextlib.suppress(Exception):
                await container.close()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


# --- helpers ----------------------------------------------------------------


@pytest.fixture
def make_client(client: httpx.AsyncClient, auth_headers: dict[str, str]) -> Factory:
    async def _make(**overrides: JsonValue) -> dict[str, JsonValue]:
        body = {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@neviswealth.com",
            "description": None,
            "social_links": [],
        } | overrides
        response = await client.post("/clients", json=body, headers=auth_headers)
        assert response.status_code == 201, response.text
        return response.json()

    return _make


@pytest.fixture
def make_document(client: httpx.AsyncClient, auth_headers: dict[str, str]) -> Factory:
    async def _make(client_id: str, title: str, content: str) -> dict[str, JsonValue]:
        response = await client.post(
            f"/clients/{client_id}/documents",
            json={"title": title, "content": content},
            headers=auth_headers,
        )
        assert response.status_code == 201, response.text
        return response.json()

    return _make
