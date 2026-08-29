from collections.abc import AsyncIterable

import asyncpg
import httpx
from dishka import Provider, Scope, provide

from src.ai.client import AIClient
from src.config import Settings
from src.repo import ClientRepo, DocumentRepo, create_pool
from src.services import ClientService, DocumentService, SearchService
from src.use_cases import CreateClientUseCase, CreateDocumentUseCase, SearchUseCase


class AppProvider(Provider):
    scope = Scope.APP

    @provide
    def settings(self) -> Settings:
        return Settings()

    @provide
    async def pool(self, settings: Settings) -> AsyncIterable[asyncpg.Pool]:
        pool = await create_pool(settings.database_url)
        yield pool
        await pool.close()

    @provide
    async def http(self, settings: Settings) -> AsyncIterable[httpx.AsyncClient]:
        async with httpx.AsyncClient(timeout=settings.ai_timeout_seconds) as client:
            yield client

    ai = provide(AIClient)
    client_repo = provide(ClientRepo)
    document_repo = provide(DocumentRepo)

    # Services hold no per-request state, and SearchService owns the cross-request
    # query-embedding cache, so they are APP-scoped. Use cases stay per-request.
    client_service = provide(ClientService)
    document_service = provide(DocumentService)
    search_service = provide(SearchService)


class RequestProvider(Provider):
    scope = Scope.REQUEST

    create_client = provide(CreateClientUseCase)
    create_document = provide(CreateDocumentUseCase)
    search = provide(SearchUseCase)
