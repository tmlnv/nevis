"""One call in, one call out. Routes depend on these, never on services directly."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.dto import (
    ClientResultDTO,
    CreateClientCommandDTO,
    CreateDocumentCommandDTO,
    DocumentResultDTO,
    SearchQueryDTO,
    SearchResultDTO,
)
from src.services import ClientService, DocumentService, SearchService

type Message = CreateClientCommandDTO | CreateDocumentCommandDTO | SearchQueryDTO
type Result = ClientResultDTO | DocumentResultDTO | SearchResultDTO

MessageT = TypeVar("MessageT", bound=Message)
ResultT = TypeVar("ResultT", bound=Result)


class UseCase(ABC, Generic[MessageT, ResultT]):  # noqa: UP046  # named TypeVars are reused below
    @abstractmethod
    async def __call__(self, message: MessageT) -> ResultT: ...


class CreateClientUseCase(UseCase[CreateClientCommandDTO, ClientResultDTO]):
    def __init__(self, clients: ClientService) -> None:
        self._clients = clients

    async def __call__(self, message: CreateClientCommandDTO) -> ClientResultDTO:
        return await self._clients.create(message)


class CreateDocumentUseCase(UseCase[CreateDocumentCommandDTO, DocumentResultDTO]):
    def __init__(self, documents: DocumentService) -> None:
        self._documents = documents

    async def __call__(self, message: CreateDocumentCommandDTO) -> DocumentResultDTO:
        return await self._documents.create(message)


class SearchUseCase(UseCase[SearchQueryDTO, SearchResultDTO]):
    def __init__(self, search: SearchService) -> None:
        self._search = search

    async def __call__(self, message: SearchQueryDTO) -> SearchResultDTO:
        return await self._search.search(message)
