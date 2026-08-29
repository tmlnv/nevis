"""HTTP trust boundary. `In` is what the API receives, `Out` is what it returns."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, RootModel, StringConstraints

from src.dto import ClientResultDTO, DocumentResultDTO, SearchResultDTO

Trimmed = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Link = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]

MAX_CONTENT_CHARS = 100_000


class CreateClientIn(BaseModel):
    first_name: Annotated[Trimmed, Field(max_length=100)]
    last_name: Annotated[Trimmed, Field(max_length=100)]
    email: EmailStr
    description: Annotated[str, Field(max_length=2_000)] | None = None
    social_links: Annotated[list[Link], Field(max_length=20)] = []


class ClientOut(BaseModel):
    id: UUID
    first_name: str
    last_name: str
    email: str
    description: str | None
    social_links: list[str]

    @classmethod
    def of(cls, dto: ClientResultDTO) -> "ClientOut":
        return cls(
            id=dto.id,
            first_name=dto.first_name,
            last_name=dto.last_name,
            email=dto.email,
            description=dto.description,
            social_links=list(dto.social_links),
        )


class CreateDocumentIn(BaseModel):
    title: Annotated[Trimmed, Field(max_length=500)]
    content: Annotated[Trimmed, Field(max_length=MAX_CONTENT_CHARS)]


class DocumentOut(BaseModel):
    id: UUID
    client_id: UUID
    title: str
    content: str
    summary: str
    created_at: datetime

    @classmethod
    def of(cls, dto: DocumentResultDTO) -> "DocumentOut":
        return cls(
            id=dto.id,
            client_id=dto.client_id,
            title=dto.title,
            content=dto.content,
            summary=dto.summary,
            created_at=dto.created_at,
        )


class SearchIn(BaseModel):
    q: Annotated[Trimmed, Field(max_length=500)]


class ClientSearchResultOut(BaseModel):
    result_type: Literal["client"] = "client"
    score: float
    client: ClientOut


class DocumentSearchResultOut(BaseModel):
    result_type: Literal["document"] = "document"
    score: float
    matched_excerpt: str
    document: DocumentOut


SearchResultOut = Annotated[
    ClientSearchResultOut | DocumentSearchResultOut, Field(discriminator="result_type")
]


class SearchResultsOut(RootModel[list[SearchResultOut]]):
    """The assignment's wire format is a bare top-level array."""

    @classmethod
    def of(cls, dto: SearchResultDTO) -> "SearchResultsOut":
        clients = [
            ClientSearchResultOut(score=round(hit.score, 4), client=ClientOut.of(hit.client))
            for hit in dto.clients
        ]
        documents = [
            DocumentSearchResultOut(
                score=round(hit.score, 4),
                matched_excerpt=hit.matched_excerpt,
                document=DocumentOut.of(hit.document),
            )
            for hit in dto.documents
        ]
        return cls(root=[*clients, *documents])


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    is_database_ready: bool


class ErrorOut(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class AIProviderErrorOut(BaseModel):
    code: str
    message: str
    provider: str
    upstream_status: int | None = None
    is_retryable: bool = False
    retry_after_seconds: int | None = None
    request_id: str | None = None


class ValidationIssueOut(BaseModel):
    location: str
    message: str


class ValidationErrorOut(BaseModel):
    code: Literal["validation_error"] = "validation_error"
    message: str
    issues: list[ValidationIssueOut]
