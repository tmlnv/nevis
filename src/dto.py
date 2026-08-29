"""Immutable inter-layer messages. No framework types cross a layer on anything else."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

type Vector = tuple[float, ...]


# --- commands and queries ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class CreateClientCommandDTO:
    first_name: str
    last_name: str
    email: str
    description: str | None
    social_links: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CreateDocumentCommandDTO:
    client_id: UUID
    title: str
    content: str


@dataclass(frozen=True, slots=True)
class SearchQueryDTO:
    q: str


# --- results ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClientResultDTO:
    id: UUID
    first_name: str
    last_name: str
    email: str
    description: str | None
    social_links: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentResultDTO:
    id: UUID
    client_id: UUID
    title: str
    content: str
    summary: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ClientSearchHitDTO:
    client: ClientResultDTO
    score: float


@dataclass(frozen=True, slots=True)
class DocumentSearchHitDTO:
    document: DocumentResultDTO
    score: float
    matched_excerpt: str


@dataclass(frozen=True, slots=True)
class SearchResultDTO:
    clients: tuple[ClientSearchHitDTO, ...]
    documents: tuple[DocumentSearchHitDTO, ...]
    is_degraded: bool


# --- AI results -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmbeddingResultDTO:
    vectors: tuple[Vector, ...]
    model: str


@dataclass(frozen=True, slots=True)
class SummaryResultDTO:
    text: str
    model: str


# --- service/repository requests --------------------------------------------


@dataclass(frozen=True, slots=True)
class ClientSearchRequestDTO:
    query: str
    limit: int


@dataclass(frozen=True, slots=True)
class DocumentSearchRequestDTO:
    embedding: Vector
    limit: int


@dataclass(frozen=True, slots=True)
class ChunkDTO:
    position: int
    content: str
    embedding: Vector


@dataclass(frozen=True, slots=True)
class PersistDocumentDTO:
    client_id: UUID
    title: str
    content: str
    summary: str
    embedding_model: str
    summary_model: str
    chunks: tuple[ChunkDTO, ...]


@dataclass(frozen=True, slots=True)
class DocumentChunkSearchHitDTO:
    document: DocumentResultDTO
    content: str
    score: float
