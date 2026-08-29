"""Raw SQL persistence. asyncpg only: no ORM, no query builder, every value a $n parameter."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import asyncpg
import orjson

from src.config import EMBEDDING_DIMENSIONS
from src.dto import (
    ClientResultDTO,
    ClientSearchHitDTO,
    ClientSearchRequestDTO,
    CreateClientCommandDTO,
    DocumentChunkSearchHitDTO,
    DocumentResultDTO,
    DocumentSearchRequestDTO,
    PersistDocumentDTO,
    Vector,
)
from src.errors import DatabaseUnavailableError

# asyncpg has no halfvec codec; it sends the parameter as text, which the cast parses.
_HALFVEC = f"::halfvec({EMBEDDING_DIMENSIONS})"

_CLIENT_COLUMNS = "id, first_name, last_name, email, description, social_links, created_at"
_DOCUMENT_COLUMNS = "id AS document_id, client_id, title, content, summary, created_at"
_JOINED_DOCUMENT_COLUMNS = (
    "d.id AS document_id, d.client_id, d.title, d.content, d.summary, d.created_at"
)


async def _init_connection(conn: asyncpg.Connection) -> None:
    # orjson.dumps returns bytes and jsonb's binary wire format needs a leading version
    # byte, so text format plus a decode() is the smaller correct option.
    await conn.set_type_codec(
        "jsonb",
        encoder=lambda value: orjson.dumps(value).decode(),
        decoder=orjson.loads,
        schema="pg_catalog",
    )


async def create_pool(dsn: str) -> asyncpg.Pool:
    try:
        return await asyncpg.create_pool(dsn, min_size=1, max_size=5, init=_init_connection)
    except (asyncpg.PostgresConnectionError, OSError) as exc:
        raise DatabaseUnavailableError(str(exc)) from exc


@asynccontextmanager
async def _acquire(pool: asyncpg.Pool) -> AsyncIterator[asyncpg.Connection]:
    try:
        async with pool.acquire() as conn:
            yield conn
    except (asyncpg.PostgresConnectionError, OSError) as exc:
        raise DatabaseUnavailableError(str(exc)) from exc


async def check_database(pool: asyncpg.Pool) -> bool:
    try:
        async with _acquire(pool) as conn:
            await conn.fetchval("SELECT 1")
    except DatabaseUnavailableError:
        return False
    return True


def _vector(v: Vector) -> str:
    """pgvector's text input format; avoids a pgvector client dependency."""
    return f"[{','.join(map(str, v))}]"


def _client(r: asyncpg.Record) -> ClientResultDTO:
    return ClientResultDTO(
        id=r["id"],
        first_name=r["first_name"],
        last_name=r["last_name"],
        email=r["email"],
        description=r["description"],
        social_links=tuple(r["social_links"]),
        created_at=r["created_at"],
    )


def _document(r: asyncpg.Record) -> DocumentResultDTO:
    return DocumentResultDTO(
        id=r["document_id"],
        client_id=r["client_id"],
        title=r["title"],
        content=r["content"],
        summary=r["summary"],
        created_at=r["created_at"],
    )


class ClientRepo:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create(self, cmd: CreateClientCommandDTO) -> ClientResultDTO:
        async with _acquire(self._pool) as conn:
            row = await conn.fetchrow(
                f"""INSERT INTO clients
                        (id, first_name, last_name, email, description, social_links)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING {_CLIENT_COLUMNS}""",
                uuid4(),
                cmd.first_name,
                cmd.last_name,
                cmd.email,
                cmd.description,
                list(cmd.social_links),
            )
        return _client(row)

    async def get(self, client_id: UUID) -> ClientResultDTO | None:
        async with _acquire(self._pool) as conn:
            row = await conn.fetchrow(
                f"SELECT {_CLIENT_COLUMNS} FROM clients WHERE id = $1", client_id
            )
        return _client(row) if row else None

    async def search(self, req: ClientSearchRequestDTO) -> tuple[ClientSearchHitDTO, ...]:
        async with _acquire(self._pool) as conn, conn.transaction():
            # search_text is long, so whole-string similarity never reaches the 0.3 default;
            # 0.15 lets multi-word queries hit `%`, `%>` covers single words, LIKE substrings.
            await conn.execute("SET LOCAL pg_trgm.similarity_threshold = 0.15")
            rows = await conn.fetch(
                f"""SELECT {_CLIENT_COLUMNS},
                           GREATEST(similarity(search_text, $1),
                                    word_similarity($1, search_text)) AS score
                    FROM clients
                    WHERE search_text % $1
                       OR search_text %> $1
                       OR search_text LIKE '%' || lower($1) || '%'
                    ORDER BY score DESC
                    LIMIT $2""",
                req.query,
                req.limit,
            )
        return tuple(ClientSearchHitDTO(client=_client(r), score=r["score"]) for r in rows)


class DocumentRepo:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create(self, payload: PersistDocumentDTO) -> DocumentResultDTO:
        document_id = uuid4()
        async with _acquire(self._pool) as conn, conn.transaction():
            row = await conn.fetchrow(
                f"""INSERT INTO documents
                        (id, client_id, title, content, summary, embedding_model, summary_model)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING {_DOCUMENT_COLUMNS}""",
                document_id,
                payload.client_id,
                payload.title,
                payload.content,
                payload.summary,
                payload.embedding_model,
                payload.summary_model,
            )
            await conn.executemany(
                f"""INSERT INTO document_chunks (id, document_id, position, content, embedding)
                    VALUES ($1, $2, $3, $4, $5{_HALFVEC})""",
                [
                    (uuid4(), document_id, c.position, c.content, _vector(c.embedding))
                    for c in payload.chunks
                ],
            )
        return _document(row)

    async def search_chunks(
        self, req: DocumentSearchRequestDTO
    ) -> tuple[DocumentChunkSearchHitDTO, ...]:
        async with _acquire(self._pool) as conn:
            rows = await conn.fetch(
                f"""SELECT {_JOINED_DOCUMENT_COLUMNS},
                           ch.content AS chunk_content,
                           1 - (ch.embedding <=> $1{_HALFVEC}) AS score
                    FROM document_chunks ch
                    JOIN documents d ON d.id = ch.document_id
                    -- distance ASC == score DESC, and only this form uses the hnsw index
                    ORDER BY ch.embedding <=> $1{_HALFVEC}
                    LIMIT $2""",
                _vector(req.embedding),
                req.limit,
            )
        return tuple(
            DocumentChunkSearchHitDTO(
                document=_document(r), content=r["chunk_content"], score=r["score"]
            )
            for r in rows
        )
