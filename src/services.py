"""Business layer. Repositories and the AI client are injected; nothing here talks to HTTP."""

import logging
import re
from collections import OrderedDict
from collections.abc import Iterator
from operator import attrgetter
from uuid import UUID

from src.ai.client import AIClient
from src.config import Settings
from src.dto import (
    ChunkDTO,
    ClientResultDTO,
    ClientSearchHitDTO,
    ClientSearchRequestDTO,
    CreateClientCommandDTO,
    CreateDocumentCommandDTO,
    DocumentChunkSearchHitDTO,
    DocumentResultDTO,
    DocumentSearchHitDTO,
    DocumentSearchRequestDTO,
    PersistDocumentDTO,
    SearchQueryDTO,
    SearchResultDTO,
    Vector,
)
from src.errors import AIProviderError, ClientNotFoundError
from src.repo import ClientRepo, DocumentRepo

log = logging.getLogger("nevis")

MAX_PASSAGE_CHARS = 1200
PASSAGE_OVERLAP_CHARS = 150
MAX_EXCERPT_CHARS = 300
EMBEDDING_CACHE_SIZE = 256
CHUNK_CANDIDATE_FACTOR = 5

_PARAGRAPH = re.compile(r"\n[ \t]*\n")
_BY_SCORE = attrgetter("score")


def split_passages(text: str) -> tuple[str, ...]:
    passages: list[str] = []
    buffer = ""
    for paragraph in (p for p in _PARAGRAPH.split(text) if p.strip()):
        if len(paragraph) > MAX_PASSAGE_CHARS:
            if buffer:
                passages.append(buffer)
                buffer = ""
            passages.extend(_windows(paragraph))
            continue
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) > MAX_PASSAGE_CHARS:
            passages.append(buffer)
            buffer = paragraph
        else:
            buffer = candidate
    if buffer:
        passages.append(buffer)
    return tuple(passages)


def _windows(paragraph: str) -> Iterator[str]:
    step = MAX_PASSAGE_CHARS - PASSAGE_OVERLAP_CHARS
    start = 0
    while True:
        yield paragraph[start : start + MAX_PASSAGE_CHARS]
        if start + MAX_PASSAGE_CHARS >= len(paragraph):
            return
        start += step


def _excerpt(content: str) -> str:
    if len(content) <= MAX_EXCERPT_CHARS:
        return content
    head = content[:MAX_EXCERPT_CHARS]
    word, _, _ = head.rpartition(" ")
    return f"{(word or head).rstrip()}…"


class ClientService:
    def __init__(self, clients: ClientRepo) -> None:
        self._clients = clients

    async def create(self, cmd: CreateClientCommandDTO) -> ClientResultDTO:
        return await self._clients.create(
            CreateClientCommandDTO(
                first_name=cmd.first_name.strip(),
                last_name=cmd.last_name.strip(),
                email=cmd.email.strip().lower(),
                description=cmd.description,
                social_links=cmd.social_links,
            )
        )


class DocumentService:
    def __init__(self, clients: ClientRepo, documents: DocumentRepo, ai: AIClient) -> None:
        self._clients = clients
        self._documents = documents
        self._ai = ai

    async def create(self, cmd: CreateDocumentCommandDTO) -> DocumentResultDTO:
        if await self._clients.get(cmd.client_id) is None:
            raise ClientNotFoundError

        passages = (cmd.title, *split_passages(cmd.content))
        # Both AI calls finish before the insert, so a provider failure persists nothing.
        embeddings = await self._ai.embed(passages)
        summary = await self._ai.summarize(cmd.content)

        return await self._documents.create(
            PersistDocumentDTO(
                client_id=cmd.client_id,
                title=cmd.title,
                content=cmd.content,
                summary=summary.text,
                embedding_model=embeddings.model,
                summary_model=summary.model,
                chunks=tuple(
                    ChunkDTO(position=i, content=passage, embedding=vector)
                    for i, (passage, vector) in enumerate(
                        zip(passages, embeddings.vectors, strict=True)
                    )
                ),
            )
        )


class SearchService:
    def __init__(
        self, clients: ClientRepo, documents: DocumentRepo, ai: AIClient, settings: Settings
    ) -> None:
        self._clients = clients
        self._documents = documents
        self._ai = ai
        self._settings = settings
        self._cache: OrderedDict[str, Vector] = OrderedDict()

    async def search(self, query: SearchQueryDTO) -> SearchResultDTO:
        q = " ".join(query.q.split())
        limit = self._settings.search_result_limit

        client_hits = await self._search_clients(q, limit)
        try:
            embedding = await self._embed(q)
        except AIProviderError as exc:
            # Semantic search is optional; trigram client matches still answer the query.
            log.warning("search degraded, embedding unavailable code=%s", exc.code)
            return SearchResultDTO(clients=client_hits, documents=(), is_degraded=True)

        document_hits = await self._search_documents(embedding, limit)
        return SearchResultDTO(
            clients=client_hits,
            # Client and document scores are different scales (pg_trgm vs cosine) and are
            # never merged into one ranking; the shared cap just prefers clients.
            documents=document_hits[: limit - len(client_hits)],
            is_degraded=False,
        )

    async def _search_clients(self, q: str, limit: int) -> tuple[ClientSearchHitDTO, ...]:
        hits = await self._clients.search(ClientSearchRequestDTO(query=q, limit=limit))
        kept = [h for h in hits if h.score >= self._settings.client_score_threshold]
        return tuple(sorted(kept, key=_BY_SCORE, reverse=True))[:limit]

    async def _search_documents(
        self, embedding: Vector, limit: int
    ) -> tuple[DocumentSearchHitDTO, ...]:
        # Wider than the limit: chunks collapse per document, so a many-chunk document
        # would starve the rest.
        hits = await self._documents.search_chunks(
            DocumentSearchRequestDTO(embedding=embedding, limit=limit * CHUNK_CANDIDATE_FACTOR)
        )
        best: dict[UUID, DocumentChunkSearchHitDTO] = {}
        for hit in hits:
            if hit.score < self._settings.document_score_threshold:
                continue
            current = best.get(hit.document.id)
            if current is None or hit.score > current.score:
                best[hit.document.id] = hit
        return tuple(
            DocumentSearchHitDTO(
                document=hit.document, score=hit.score, matched_excerpt=_excerpt(hit.content)
            )
            for hit in sorted(best.values(), key=_BY_SCORE, reverse=True)
        )

    async def _embed(self, q: str) -> Vector:
        # lru_cache cannot wrap a coroutine; a bounded LRU on the instance is enough.
        if (cached := self._cache.get(q)) is not None:
            self._cache.move_to_end(q)
            return cached
        vector = (await self._ai.embed((q,))).vectors[0]
        self._cache[q] = vector
        if len(self._cache) > EMBEDDING_CACHE_SIZE:
            self._cache.popitem(last=False)
        return vector
