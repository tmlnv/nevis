from typing import Any
from uuid import UUID

import asyncpg
from dishka import FromDishka
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter, Depends, Response

from src.auth import require_bearer
from src.dto import CreateClientCommandDTO, CreateDocumentCommandDTO, SearchQueryDTO
from src.repo import check_database
from src.schemas import (
    AIProviderErrorOut,
    ClientOut,
    CreateClientIn,
    CreateDocumentIn,
    DocumentOut,
    ErrorOut,
    HealthOut,
    SearchIn,
    SearchResultsOut,
    ValidationErrorOut,
)
from src.use_cases import CreateClientUseCase, CreateDocumentUseCase, SearchUseCase

router = APIRouter(route_class=DishkaRoute)

_AUTH = [Depends(require_bearer)]
_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorOut},
    422: {"model": ValidationErrorOut},
}
_AI_ERRORS = {**_ERRORS, 502: {"model": AIProviderErrorOut}, 503: {"model": AIProviderErrorOut}}


@router.get("/health", tags=["health"])
async def health(pool: FromDishka[asyncpg.Pool]) -> HealthOut:
    """Process and database liveness. Deliberately does not call the AI provider."""
    return HealthOut(is_database_ready=await check_database(pool))


@router.post(
    "/clients",
    status_code=201,
    tags=["clients"],
    dependencies=_AUTH,
    responses=_ERRORS,
)
async def create_client(
    body: CreateClientIn,
    use_case: FromDishka[CreateClientUseCase],
) -> ClientOut:
    command = CreateClientCommandDTO(
        first_name=body.first_name,
        last_name=body.last_name,
        email=body.email,
        description=body.description,
        social_links=tuple(body.social_links),
    )
    return ClientOut.of(await use_case(command))


@router.post(
    "/clients/{id}/documents",
    status_code=201,
    tags=["documents"],
    dependencies=_AUTH,
    responses={**_AI_ERRORS, 404: {"model": ErrorOut}},
)
async def create_document(
    id: UUID,
    body: CreateDocumentIn,
    use_case: FromDishka[CreateDocumentUseCase],
) -> DocumentOut:
    command = CreateDocumentCommandDTO(client_id=id, title=body.title, content=body.content)
    return DocumentOut.of(await use_case(command))


@router.get(
    "/search",
    tags=["search"],
    dependencies=_AUTH,
    responses=_ERRORS,
)
async def search(
    response: Response,
    use_case: FromDishka[SearchUseCase],
    query: SearchIn = Depends(),
) -> SearchResultsOut:
    result = await use_case(SearchQueryDTO(q=query.q))
    if result.is_degraded:
        # Semantic search was unavailable; client matches are still returned. The
        # assignment's wire format is a bare array, so this is signalled in a header.
        response.headers["X-Search-Degraded"] = "true"
    return SearchResultsOut.of(result)
