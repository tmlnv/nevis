# Nevis Search API Design

- **Date:** 2026-08-29
- **Status:** Approved for implementation planning
- **Source:** Nevis backend home assignment and local project requirements

## 1. Goal

Build a small REST API that stores clients and their documents, searches client identity fields using fuzzy text matching, searches document content semantically, and produces a short document summary. The complete system must run through Docker Compose, be understandable within a home-assignment review, and remain small enough to implement and verify within the stated 10-14 hour budget.

The design favors one PostgreSQL database over a separate search service. PostgreSQL provides relational storage, `pg_trgm` client search, and `pgvector` document search in the same transactional system.

## 2. Decisions

| Area | Decision |
| --- | --- |
| Language | Python 3.13 managed with `uv` |
| HTTP API | FastAPI with generated OpenAPI/Swagger documentation |
| JSON | `orjson` for API responses and OpenRouter payload encoding/decoding |
| ASGI server | Granian, as requested |
| Architecture | API -> UseCase -> Service -> Repository |
| Dependency injection | Dishka with `FromDishka[...]` at the API boundary and constructor injection below it |
| API authentication | Mandatory bearer key for every business endpoint |
| Boundary schemas | Pydantic schemas in boundary-specific `schemas/` packages, named by application direction with `In`/`Out` suffixes |
| Inter-layer transfer | Immutable, slotted dataclass DTOs only; Pydantic schemas, `asyncpg.Record`, and repository models never cross a layer boundary |
| Database access | `asyncpg` connection pool with PyPika-built PostgreSQL queries; no ORM |
| Schema migrations | `dbmate` with timestamp-versioned, transactional raw-SQL migrations and a committed generated `db/schema.sql` |
| Database | PostgreSQL with `pg_trgm` and `pgvector` |
| Client search | Trigram matching only; clients are not embedded |
| Document search | Chunk-level dense embeddings stored in PostgreSQL |
| Search result cap | Internal fixed cap of 20; the public search schema contains only `q` |
| Default AI provider | OpenRouter free tier for embeddings and summaries |
| Embedding model | `liquid/lfm-2.5-embedding-350m:free`, 1,024 dimensions |
| Summary model | `openrouter/free`, configurable without code changes |
| Alternative summary provider | Direct DeepSeek API using an OpenAI-compatible endpoint |
| Error handling | Registered FastAPI handlers over custom application exceptions, with explicit provider-attributed OpenRouter failures |
| Tests | pytest with mocked AI responses and real PostgreSQL integration tests |
| Runtime packaging | Dockerfile plus `compose.yaml` for API, one-shot dbmate migrations, and PostgreSQL |
| Recommended free deployment | Oracle Cloud Always Free VM running the same Compose project |

## 3. Scope

### Included

- `POST /clients`
- `POST /clients/{id}/documents`
- `GET /search?q=...`
- `GET /health` for container and hosting health checks
- Mandatory bearer-key protection for every business endpoint
- Email, URL, length, and empty-value validation
- Fuzzy client search over first name, last name, email, and description
- Semantic document search over all document content
- Chunking for documents longer than the embedding model context
- A stored two-to-three-sentence document summary
- Typed, ranked client and document search results
- Docker Compose setup, API documentation, examples, and core tests
- Versioned schema migration, rollback, and migration-status commands
- Explicit custom-exception mapping of external AI failures to provider-attributed HTTP errors
- A documented DeepSeek summary-provider configuration

### Deliberately excluded

- User accounts and role-based authorization: a single required bearer key protects the API and provider quota without becoming an identity subsystem.
- Update and delete endpoints: not requested.
- Background jobs, queues, retries with persistence, or an outbox: synchronous ingestion is sufficient for the assignment workload.
- Elasticsearch: PostgreSQL already meets both search requirements without data synchronization or another service.
- A local model container: external inference keeps the deployment small and free-tier friendly.
- An ORM: repositories use PyPika and asyncpg directly. Schema versioning is still mandatory and is handled by `dbmate`, not an ORM-bound migration framework.
- Caller-supplied search limits and pagination: neither appears in the requested API schema. Search returns at most 20 results internally; expose a `limit` parameter only if the contract later requests it.

## 4. Architecture

```text
Client
  |
  v
FastAPI API: bearer authentication, validation, HTTP mapping
  |  Pydantic *In -> dataclass DTO; dataclass DTO -> Pydantic *Out
  |  FromDishka[ConcreteUseCase]
  v
UseCase[CommandDTO | QueryDTO, ResultDTO]
  |
  v
Service ------------------------------> AI gateway -> OpenRouter
  |                                      DTO <-> Pydantic *Out/*In
  |
  v
Repository -> internal dataclass Model -> PyPika -> asyncpg.Pool -> PostgreSQL
                                      + pg_trgm
                                      + pgvector
```

The dependency direction is strict: API -> UseCase -> Service -> Repository. A route never receives a service, repository, database pool, or AI client directly. It authenticates and validates the request, creates a command or query, invokes its injected use case, and maps the returned result to an HTTP response.

Every concrete use case inherits one generic abstract base. `CommandDTO`, `QueryDTO`, and `ResultDTO` below are explicit union type aliases over the concrete dataclass DTOs, so the contract permits only the defined application messages and results:

```python
MessageT = TypeVar("MessageT", bound=CommandDTO | QueryDTO)
ResultT = TypeVar("ResultT", bound=ResultDTO)


class UseCase(ABC, Generic[MessageT, ResultT]):
    @abstractmethod
    async def __call__(self, message: MessageT) -> ResultT:
        raise NotImplementedError
```

The application defines `CreateClientCommandDTO`, `CreateDocumentCommandDTO`, and `SearchQueryDTO`, with corresponding result DTOs. Every DTO uses `@dataclass(frozen=True, slots=True)` and contains only standard-library value types or other DTOs. Concrete use cases perform one application workflow and delegate business decisions to services.

The boundary rule is mechanical:

- The API validates an external `*In` Pydantic schema and maps it to a command/query DTO before invoking a use case.
- A use case, service, and repository exchange dataclass DTOs only.
- A repository may map an `asyncpg.Record` into an internal persistence `*Model`, but converts that model to a result DTO before returning. Neither records nor repository models leave the repository layer.
- The OpenRouter adapter serializes a provider `*Out` Pydantic schema and validates a provider `*In` Pydantic schema, then maps the validated data to/from AI DTOs before returning across the infrastructure/service boundary.
- Plain dictionaries are confined to Pydantic serialization internals; no layer uses untyped dictionaries as messages.
- Expected failures travel through the custom application-exception hierarchy. Provider exceptions contain a `ProviderFailureDTO`; they never carry a Pydantic model, raw response body, prompt, document text, or API key.

`In` and `Out` are always named from this application's viewpoint. An HTTP request is `In` and an HTTP response is `Out`; conversely, an OpenRouter request is `Out` and its response is `In`.

Services own normalization, chunking, AI coordination, ranking, and other business behavior. Repositories own persistence and search queries, including PyPika construction, connection acquisition, and short database transactions. PostgreSQL owns referential integrity, text indexes, vector indexes, and ranking primitives. OpenRouter only generates embeddings and summaries; it is not a repository or data store.

Dishka is the composition root. `Settings`, `asyncpg.Pool`, the shared `httpx.AsyncClient`, repositories, and external AI adapter live at `Scope.APP`; services and concrete use cases live at `Scope.REQUEST`. FastAPI routers use `DishkaRoute`, and handler parameters declare concrete use cases as `FromDishka[CreateClientUseCase]`, `FromDishka[CreateDocumentUseCase]`, or `FromDishka[SearchUseCase]`. Lower layers use constructor injection and contain no Dishka imports.

```python
@router.post("/clients", status_code=201)
async def create_client(
    body: CreateClientIn,
    use_case: FromDishka[CreateClientUseCase],
) -> ClientOut:
    command = CreateClientCommandDTO(
        first_name=body.first_name,
        last_name=body.last_name,
        email=str(body.email),
        description=body.description,
        social_links=tuple(map(str, body.social_links)),
    )
    result = await use_case(command)
    return ClientOut.from_dto(result)
```

Reference: [Dishka FastAPI `FromDishka` integration](https://dishka.readthedocs.io/en/stable/integrations/fastapi.html).

## 5. Proposed Source Layout

```text
.
├── src/
│   ├── main.py                   # app creation and Dishka setup
│   ├── config.py                 # validated required settings
│   ├── providers.py              # Dishka APP/REQUEST registrations
│   ├── api/
│   │   ├── auth.py               # mandatory HTTP bearer dependency
│   │   ├── exception_handlers.py # register_exception_handlers(app)
│   │   ├── routes.py             # routes, FromDishka use cases, HTTP mapping
│   │   └── schemas/               # Pydantic schemas for the HTTP trust boundary
│   │       ├── clients.py         # CreateClientIn, ClientOut
│   │       ├── documents.py       # ClientIdIn, CreateDocumentIn, DocumentOut
│   │       ├── search.py          # SearchIn, *ResultOut, SearchResultsOut
│   │       ├── errors.py          # ErrorOut, AIProviderErrorOut, ValidationErrorOut
│   │       └── common.py          # HealthOut
│   ├── application/
│   │   ├── exceptions.py         # custom application/provider exceptions
│   │   └── use_cases.py          # UseCase ABC and concrete use cases
│   ├── dto/                       # immutable slotted dataclasses shared across layers
│   │   ├── commands.py            # CreateClientCommandDTO, CreateDocumentCommandDTO
│   │   ├── queries.py             # SearchQueryDTO
│   │   ├── results.py             # client, document, and search result DTOs
│   │   ├── ai.py                  # embedding and summary request/result DTOs
│   │   ├── errors.py              # ProviderFailureDTO
│   │   └── persistence.py         # typed service/repository request DTOs
│   ├── services/
│   │   ├── ai.py                 # embedding and summary service boundary
│   │   ├── clients.py            # client business behavior
│   │   ├── documents.py          # chunking, embedding, summarization flow
│   │   └── search.py             # result grouping and score normalization
│   ├── repositories/
│   │   ├── models.py             # internal Client/Document/DocumentChunk models
│   │   ├── clients.py            # client persistence and pg_trgm queries
│   │   └── documents.py          # document writes and pgvector queries
│   └── infrastructure/
│       ├── database.py           # asyncpg pool and custom PyPika terms
│       └── openrouter/
│           ├── client.py         # embedding and summary HTTP adapter
│           └── schemas/          # Pydantic schemas for the provider trust boundary
│               ├── embeddings.py # EmbeddingRequestOut, EmbeddingDataIn, EmbeddingResponseIn
│               ├── errors.py     # OpenRouterErrorIn, OpenRouterErrorResponseIn
│               └── summaries.py  # ChatMessageOut, ChatCompletionRequestOut, ChatCompletionResponseIn
├── db/
│   ├── migrations/
│   │   └── 20260829000100_initial_schema.sql
│   └── schema.sql                # dbmate-generated schema snapshot; never hand-edited
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_boundaries.py
│   ├── test_migrations.py
│   ├── test_use_cases.py
│   ├── test_services.py
│   └── test_repositories.py
├── .env.example
├── Dockerfile
├── compose.yaml
├── pyproject.toml
└── README.md
```

Each layer has one direction of dependency. API code depends on application use cases and DTOs; use cases depend on services and DTOs; services depend on repositories and DTOs; repositories depend on infrastructure and DTOs. Infrastructure never imports an upper layer. The source root is `src/`; there is deliberately no extra `src/nevis/` package level.

Repository implementations and their persistence models live under `src/repositories/`; specifically, `src/repositories/models.py` defines immutable, slotted dataclasses named `ClientModel`, `DocumentModel`, and `DocumentChunkModel`. They mirror persisted rows and exist only to make record mapping explicit and typed. They are not Pydantic schemas, application DTOs, ORM entities, or repository return types.

The DTO catalogue is intentionally concrete rather than hierarchical:

- Commands and queries: `CreateClientCommandDTO`, `CreateDocumentCommandDTO`, and `SearchQueryDTO`.
- Workflow results: `ClientResultDTO`, `DocumentResultDTO`, `ClientSearchHitDTO`, `DocumentSearchHitDTO`, and `SearchResultDTO`.
- AI calls: `EmbeddingRequestDTO`, `EmbeddingResultDTO`, `SummaryRequestDTO`, and `SummaryResultDTO`.
- Service/repository calls: `ClientLookupDTO`, `PersistDocumentDTO`, `ClientSearchRequestDTO`, `DocumentSearchRequestDTO`, and `DocumentChunkSearchHitDTO`.

Repository methods accept one request DTO and return a result DTO, a tuple of result DTOs, or `None` for absence. For example, client lookup returns `ClientResultDTO | None`, document insertion accepts `PersistDocumentDTO` and returns `DocumentResultDTO`, and semantic search returns `tuple[DocumentChunkSearchHitDTO, ...]`. Bare positional payloads and untyped row tuples are not layer contracts.

### Query construction contract

Repositories build all runtime `SELECT`, `INSERT`, `UPDATE`, and `DELETE` statements with PyPika's `PostgreSQLQuery`. Values are represented by explicit `Parameter("$n")` terms and passed separately to `asyncpg` in the same order:

```python
clients = Table("clients")
query = (
    PostgreSQLQuery.from_(clients)
    .select(clients.star)
    .where(clients.id == Parameter("$1"))
)
row = await connection.fetchrow(query.get_sql(), client_id)
```

Application values are never formatted into SQL strings. PostgreSQL functions absent from PyPika, such as `similarity` and `word_similarity`, use `CustomFunction`. The pgvector cosine-distance expression uses one small custom PyPika `Term` for the `<=>` operator with `Parameter("$1")` as its right operand; it does not accept or interpolate raw user text.

The extension, table, constraint, and index definitions remain explicit DDL in `db/migrations/20260829000100_initial_schema.sql`. PyPika is a runtime query builder, not a schema migration system.

`asyncpg.create_pool()` opens once during the FastAPI lifespan and closes during shutdown. The free-tier deployment starts with a small pool, such as one warm connection and at most five connections. Acquired connections are returned promptly, and the document transaction begins only after both external AI calls have completed. Direct Compose connections may use asyncpg's statement cache; if a transaction-mode PgBouncer is introduced later, disable that cache or configure the pooler deliberately.

References: [PyPika parameterized queries](https://github.com/kayak/pypika#parametrized-queries), [asyncpg native query arguments](https://github.com/MagicStack/asyncpg/blob/master/docs/usage.rst), and [pgvector asyncpg integration](https://github.com/pgvector/pgvector-python#asyncpg).

### Migration contract

`dbmate` is mandatory from the first schema version. It is a standalone migration binary, so SQLAlchemy and a second Python database driver are not added to the application. Migrations are timestamp-versioned raw SQL in `db/migrations/`, contain both `-- migrate:up` and `-- migrate:down`, and run transactionally unless a specific PostgreSQL operation requires `transaction:false`.

The initial migration enables `pg_trgm` and `vector`, then creates tables, constraints, and indexes in dependency order. Its down section reverses that order. `dbmate` owns the `schema_migrations` table. Applied migrations are immutable: correcting an already-applied schema requires a new migration, not editing its versioned file.

`db/schema.sql` is the generated current-schema snapshot. It is committed for review and fast test-database setup but never hand-edited; `dbmate dump` regenerates it. Migration files remain the authoritative history.

Compose uses a pinned `ghcr.io/amacneil/dbmate` image as a one-shot `migrate` service. It waits for healthy PostgreSQL, executes `dbmate migrate`, and exits. The API starts only after that service completes successfully. Migrations do not run inside FastAPI startup, which prevents multiple API workers from racing to alter the schema. A hosted deployment uses the same one-shot service or an equivalent release command before replacing the API container.

Required operations are:

```bash
dbmate new <migration_name>
dbmate migrate
dbmate status
dbmate rollback
dbmate dump
```

Reference: [dbmate official documentation](https://github.com/amacneil/dbmate).

## 6. Data Model

### `clients`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Generated by the API |
| `first_name` | text | Required, stripped, non-empty |
| `last_name` | text | Required, stripped, non-empty |
| `email` | text | Required, validated and stored lowercase |
| `description` | text, nullable | Optional |
| `social_links` | jsonb | Array of validated absolute URLs; defaults to `[]` |
| `search_text` | generated text | Lowercase concatenation of searchable client fields |
| `created_at` | timestamptz | Database default `now()` |

`search_text` receives a GIN index using `gin_trgm_ops`. Email is not made unique because the assignment does not state that family members or shared accounts cannot reuse an address.

### `documents`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Generated before external calls |
| `client_id` | UUID | Foreign key to `clients`; cascade on client deletion |
| `title` | text | Required, stripped, non-empty |
| `content` | text | Required, stripped, non-empty |
| `summary` | text | Generated during creation |
| `embedding_model` | text | Model provenance for future re-embedding |
| `summary_model` | text | Model provenance for debugging |
| `created_at` | timestamptz | Database default `now()` |

### `document_chunks`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Generated by the API |
| `document_id` | UUID | Foreign key to `documents`; cascade on delete |
| `position` | integer | Zero-based order in the source document |
| `content` | text | Exact passage used for the embedding |
| `embedding` | `vector(1024)` | OpenRouter embedding |

The pair `(document_id, position)` is unique. The vector column uses cosine distance and an HNSW index with `vector_cosine_ops`. HNSW is not required for the tiny demo corpus, but it makes the selected storage design valid as the corpus grows without adding application code.

## 7. AI Provider Design

### Default configuration

```dotenv
API_BEARER_TOKEN=replace-with-a-long-random-value
OPENROUTER_API_KEY=replace-me
EMBEDDING_MODEL=liquid/lfm-2.5-embedding-350m:free
SUMMARY_BASE_URL=https://openrouter.ai/api/v1
SUMMARY_API_KEY=replace-me
SUMMARY_MODEL=openrouter/free
```

`SUMMARY_API_KEY` defaults to `OPENROUTER_API_KEY` when omitted. Embeddings always use `https://openrouter.ai/api/v1/embeddings` in the initial implementation. Summaries use `${SUMMARY_BASE_URL}/chat/completions`.

The API uses `httpx.AsyncClient` directly. Two JSON endpoints do not justify an additional OpenAI or OpenRouter SDK dependency. Provider requests are encoded with `orjson.dumps()` and responses are decoded with `orjson.loads()` before Pydantic validation. The adapter sets `Content-Type: application/json` explicitly because `orjson.dumps()` returns bytes.

The provider boundary has its own Pydantic schemas and never reuses API schemas:

| Direction from application | Pydantic schema | Purpose |
| --- | --- | --- |
| Out | `EmbeddingRequestOut` | OpenRouter embeddings request body |
| In | `EmbeddingResponseIn` | Validated embeddings response, including indexed data items |
| Out | `ChatCompletionRequestOut` | OpenRouter/DeepSeek chat-completions request body |
| In | `ChatCompletionResponseIn` | Validated summary response, including choices and message content |
| In | `OpenRouterErrorResponseIn` | Best-effort validation of non-2xx OpenRouter error envelopes |

Nested provider structures follow the same rule: examples include `EmbeddingDataIn`, `UsageIn`, `ChatMessageOut`, `ChatChoiceIn`, `ChatMessageIn`, and `OpenRouterErrorIn`. Provider JSON is never represented by nested dictionaries after validation. Error parsing is permissive about unknown fields because OpenRouter may add metadata; failure to validate an error envelope never hides its HTTP status or turns it into an internal `500`.

The AI gateway accepts and returns dataclass DTOs. Its adapter alone performs DTO <-> provider-schema mapping; OpenRouter field names and response envelopes cannot leak into services.

`API_BEARER_TOKEN` is required at startup in every environment. There is no unauthenticated development mode. Every business endpoint requires `Authorization: Bearer <token>`; only `/health`, `/docs`, and `/openapi.json` remain public. A FastAPI `HTTPBearer` dependency retrieves `Settings` through `FromDishka[Settings]`, compares credentials with `hmac.compare_digest`, returns `401` with `WWW-Authenticate: Bearer` when invalid, and never logs the token. Swagger exposes the bearer security scheme so reviewers can authorize once and exercise the API.

### DeepSeek summary alternative

The same summary client can use a direct DeepSeek key:

```dotenv
SUMMARY_BASE_URL=https://api.deepseek.com
SUMMARY_API_KEY=replace-with-deepseek-key
SUMMARY_MODEL=deepseek-v4-flash
```

DeepSeek is an alternative only for summaries because its current first-party API does not expose an embeddings endpoint. No runtime fallback between providers is implemented: a configured provider either succeeds or the request returns a controlled error. Automatic multi-provider fallback would obscure failures and complicate tests.

### Privacy constraint

The selected free OpenRouter embedding endpoint states that successful requests and embeddings may be retained and used for training. The deployed demo must use synthetic data only. A real WealthTech deployment must switch to an approved zero-data-retention endpoint, a provider covered by the company's data-processing agreement, or local inference before accepting client documents.

References:

- [OpenRouter embedding model list](https://openrouter.ai/api/v1/embeddings/models)
- [OpenRouter embeddings API](https://openrouter.ai/docs/api/api-reference/embeddings/submit-an-embedding-request)
- [OpenRouter free-model limits](https://openrouter.ai/docs/faq)
- [OpenRouter zero-data-retention controls](https://openrouter.ai/docs/guides/features/zdr)
- [DeepSeek API models and OpenAI-compatible base URL](https://api-docs.deepseek.com/quick_start/pricing/)

## 8. Document Creation Flow

`POST /clients/{id}/documents` follows the layer chain synchronously:

1. The API authenticates the request, validates `ClientIdIn` and `CreateDocumentIn`, maps them to `CreateDocumentCommandDTO`, and calls `FromDishka[CreateDocumentUseCase]`.
2. `CreateDocumentUseCase` delegates the command DTO to `DocumentService` and returns `DocumentResultDTO`.
3. `DocumentService` sends `ClientLookupDTO` to `ClientRepository` and requires a non-`None` `ClientResultDTO` before spending an external API request.
4. `DocumentService` splits `title + content` into ordered passages. It prefers paragraph boundaries and splits oversized paragraphs into approximately 1,200-character windows with a small overlap. This avoids adding a local model tokenizer while remaining comfortably below 512 tokens for expected English prose.
5. The AI service submits all passages in one OpenRouter embedding request with `input_type: "search_document"` and requests a two-to-three-sentence summary with a bounded output length.
6. `DocumentService` validates the embedding count and requires every vector to contain exactly 1,024 finite numbers.
7. `DocumentService` builds `PersistDocumentDTO`; `DocumentRepository` inserts its document and chunks in one short `asyncpg` transaction and returns `DocumentResultDTO`.
8. The result DTO travels back through use case and API, where it becomes `DocumentOut`; the API returns `201 Created` only after the transaction commits.

All external calls happen before the database transaction, so no transaction remains open while waiting on the network. If either external call fails, nothing is inserted. This all-or-nothing behavior is the smallest reliable contract for the assignment.

The API accepts content up to 100,000 characters. That boundary prevents accidental memory and provider-cost abuse while still covering substantial text documents. A larger production ingestion system would stream files and summarize hierarchically.

## 9. Search Design

### Client search

Clients do not receive embeddings. PostgreSQL compares the normalized query against `search_text` using:

- trigram similarity;
- word similarity, which can match a query inside a longer email or description; and
- a case-insensitive substring boost for exact fragments such as `NevisWealth` inside `john.doe@neviswealth.com`.

The greatest of these signals becomes the client score in the range 0-1.

### Document search

For each `GET /search?q=...` request:

1. The API validates `SearchIn`, maps it to `SearchQueryDTO`, and invokes `FromDishka[SearchUseCase]`.
2. `SearchUseCase` delegates to `SearchService`.
3. `SearchService` normalizes the query and asks the AI service for one embedding using `input_type: "search_query"`.
4. `ClientRepository` receives `ClientSearchRequestDTO` for the `pg_trgm` query while `DocumentRepository` receives `DocumentSearchRequestDTO` and returns nearest `DocumentChunkSearchHitDTO` values by cosine distance.
5. `SearchService` groups the chunk-hit DTOs by `document_id`, keeps the best-scoring chunk for each document, and merges document/client result DTOs.
6. `SearchUseCase` returns `SearchResultDTO`; the API maps each member to the corresponding discriminated `*ResultOut` schema.

Grouping prevents one long document from occupying multiple response positions. It also preserves the passage that explains why the document matched:

```text
query: "address proof"

raw chunk matches:
  document A / chunk 2 / "electricity utility bill" / 0.91
  document B / chunk 0 / "proof of residence"       / 0.84
  document A / chunk 0 / "client profile"           / 0.55

document results:
  document A / best chunk 2 / 0.91
  document B / best chunk 0 / 0.84
```

### Mixed result ranking

Client and document scores are both normalized to 0-1, merged, sorted descending, and capped at the first 20 results after the merge. `SEARCH_RESULT_LIMIT = 20` is an internal service constant, not a setting or API field, because the supplied search schema exposes only `q`. The initial implementation uses fixed relevance cutoffs tuned against the acceptance examples. This is a deliberate assignment-scale simplification; production thresholds should be calibrated from labeled searches rather than exposed as speculative configuration.

The document repository retrieves a larger fixed internal candidate set, initially 60 chunk hits, before grouping by document. This prevents duplicate chunks from reducing the final result count while keeping both numbers out of the public API. A caller-supplied `limit` and pagination can be designed later if explicitly requested.

## 10. API Contract

Every HTTP data structure is a Pydantic schema under `src/api/schemas/`; this directory contains schemas, never modules or classes named `models`. The directional contract is:

| Endpoint data | In schema | Out schema |
| --- | --- | --- |
| `POST /clients` body/result | `CreateClientIn` | `ClientOut` |
| `POST /clients/{id}/documents` path/body/result | `ClientIdIn`, `CreateDocumentIn` | `DocumentOut` |
| `GET /search` query/result | `SearchIn` containing only `q` | `SearchResultsOut`, containing `ClientSearchResultOut` or `DocumentSearchResultOut` |
| `GET /health` | none | `HealthOut` |
| Controlled error body | none | `ErrorOut` |
| AI dependency error body | none | `AIProviderErrorOut` |
| Validation error body | none | `ValidationErrorOut`, containing `ValidationIssueOut` values |

`SearchResultsOut` is a Pydantic `RootModel` so the wire format remains the assignment's top-level array. Custom exception handlers expose the suffixed `ErrorOut` and `ValidationErrorOut` schemas instead of leaving FastAPI's generated `HTTPValidationError` as an unnamed exception to the naming rule.

Routes map explicitly between Pydantic schemas and dataclass DTOs. They do not pass `BaseModel` instances to use cases and do not return DTOs directly. `FastAPI(default_response_class=ORJSONResponse)` makes `orjson` the default response encoder while response annotations and OpenAPI continue to use the `*Out` Pydantic schemas.

References: [FastAPI custom/default response classes](https://fastapi.tiangolo.com/advanced/custom-response/) and [orjson supported native types](https://github.com/ijl/orjson#types).

### Authentication

All client, document, and search requests require:

```http
Authorization: Bearer <API_BEARER_TOKEN>
```

Authentication is mandatory locally and in deployments. The token is supplied through `.env`, shown only as a placeholder in `.env.example`, and shared with reviewers outside the repository.

### Create client

```http
POST /clients
Authorization: Bearer <API_BEARER_TOKEN>
Content-Type: application/json
```

Returns `201` with the specified client schema. Invalid fields return FastAPI's `422` response.

### Create document

```http
POST /clients/{id}/documents
Authorization: Bearer <API_BEARER_TOKEN>
Content-Type: application/json
```

The response extends the supplied `Document` schema with `summary`. A missing client returns `404`. An unavailable AI provider returns `503` and creates no document.

### Search

```http
GET /search?q=address%20proof
Authorization: Bearer <API_BEARER_TOKEN>
```

The response remains an array, as supplied by the assignment, using a discriminated union:

```json
[
  {
    "result_type": "document",
    "score": 0.91,
    "matched_excerpt": "Electricity utility bill issued within the last three months...",
    "document": {
      "id": "...",
      "client_id": "...",
      "title": "Utility bill",
      "content": "...",
      "summary": "...",
      "created_at": "2026-08-29T12:00:00Z"
    }
  },
  {
    "result_type": "client",
    "score": 1.0,
    "client": {
      "id": "...",
      "first_name": "John",
      "last_name": "Doe",
      "email": "john.doe@neviswealth.com",
      "description": null,
      "social_links": []
    }
  }
]
```

`score` communicates rank but is not promised to be a calibrated probability.

Business endpoints always return `401` for a missing or invalid bearer credential, before constructing a command/query or resolving a use case.

### Health

`GET /health` checks that the process is running and PostgreSQL accepts a trivial query. It does not call OpenRouter, avoiding external calls from infrastructure health probes.

## 11. Error Handling

Errors are part of the API contract. Clients must be able to distinguish an application failure from an unavailable external AI dependency, particularly when a free OpenRouter model exhausts its quota, is rate-limited, or has no available provider.

### Custom exceptions and registration

`src/application/exceptions.py` defines custom exceptions rather than returning error objects from use cases:

- `ApplicationError` as the common expected-error base;
- `ClientNotFoundError`;
- `AIProviderError`, carrying one immutable `ProviderFailureDTO`;
- `AIProviderConfigurationError`;
- `AIProviderQuotaExhaustedError`;
- `AIProviderRateLimitedError`;
- `AIProviderUnavailableError`;
- `AIProviderRejectedRequestError`;
- `AIProviderInvalidResponseError`; and
- `DatabaseUnavailableError`.

The OpenRouter adapter translates `httpx` failures, non-2xx statuses, error envelopes, and malformed success envelopes into these exceptions. Services and use cases do not convert them to HTTP concepts; they allow the typed exception to propagate. Routes contain no repeated provider `try/except` blocks.

`src/api/exception_handlers.py` exposes one `register_exception_handlers(app: FastAPI) -> None` function. `create_app()` calls it once immediately after constructing FastAPI. Registration covers `RequestValidationError`, authentication/HTTP errors, every `ApplicationError`, database availability errors, and an unexpected-exception fallback. The `AIProviderError` handler covers all provider subclasses and selects the public status/code from the contained failure kind.

```python
def create_app() -> FastAPI:
    app = FastAPI(default_response_class=ORJSONResponse)
    register_exception_handlers(app)
    # include routers and attach Dishka
    return app
```

This is explicit application-level registration, not middleware that guesses error types from strings.

### Public provider error

Provider failures use `AIProviderErrorOut`:

```json
{
  "code": "ai_provider_rate_limited",
  "message": "OpenRouter is temporarily unavailable because its free-tier limit has been reached.",
  "provider": "openrouter",
  "upstream_status": 429,
  "retryable": true,
  "retry_after_seconds": 3600,
  "request_id": "4e85b76c-5007-4cd5-9a66-9b67c2c302f9"
}
```

The response deliberately identifies OpenRouter so a caller does not misdiagnose the API or PostgreSQL as broken. `code` and `message` are stable application-owned values. `upstream_status` may be exposed, and a valid bounded `Retry-After` value may be forwarded as both `retry_after_seconds` and a response header. Raw OpenRouter messages, metadata, bodies, prompts, client documents, and credentials are never returned.

### OpenRouter mapping

| OpenRouter/httpx condition | Custom exception | API status/code | Retryable |
| --- | --- | --- | --- |
| `401` | `AIProviderConfigurationError` | `503 ai_provider_configuration_error` | No |
| `402` credits/quota unavailable | `AIProviderQuotaExhaustedError` | `503 ai_provider_quota_exhausted` | No; deployment owner action or quota reset is required |
| `429` free-tier/rate limit | `AIProviderRateLimitedError` | `503 ai_provider_rate_limited` | Yes, using valid `Retry-After` guidance when supplied |
| `404` no configured model/provider available | `AIProviderUnavailableError` | `503 ai_provider_unavailable` | Yes |
| `408`, connection failure, DNS/TLS error, or timeout | `AIProviderUnavailableError` | `503 ai_provider_unavailable` | Yes |
| `500`, `502`, `503`, `504`, `524`, or `529` overload | `AIProviderUnavailableError` | `503 ai_provider_unavailable` | Yes |
| `400`, `403`, `413`, or `422` provider rejection | `AIProviderRejectedRequestError` | `502 ai_provider_rejected_request` | No |
| Non-JSON or schema-invalid 2xx response, missing summary, wrong embedding count, non-finite number, or wrong vector dimension | `AIProviderInvalidResponseError` | `502 ai_provider_invalid_response` | No |
| Any other non-2xx response | `AIProviderError` | `502 ai_provider_error` | No by default |

An upstream `429` is returned as this API's `503`, not `429`, because the exhausted limit belongs to the application's shared OpenRouter dependency rather than to the API caller. The response body and `provider` field preserve the precise cause.

The same exception/handler contract supports the optional direct DeepSeek summary endpoint with `provider: "deepseek"`; the detailed status mapping above is mandatory for OpenRouter.

### Other application errors

| Condition | HTTP status/schema | Behavior |
| --- | --- | --- |
| Invalid JSON, email, URL, UUID, length, or empty value | `422 ValidationErrorOut` | Stable validation issues; no use-case call where validation occurs first |
| Client not found | `404 ErrorOut` | No AI call |
| Empty search query after trimming | `422 ValidationErrorOut` | No database or AI call |
| Missing or invalid bearer token | `401 ErrorOut` | No use-case, database, or AI call |
| Database unavailable | `503 ErrorOut` | `database_unavailable`; no connection details exposed |
| Unexpected server error | `500 ErrorOut` | `internal_error` plus request ID; full exception only in server logs |

All documented error responses are added to FastAPI route metadata so OpenAPI shows `ErrorOut`, `ValidationErrorOut`, or `AIProviderErrorOut` for the relevant status codes.

### Retry, logging, and health behavior

External calls use explicit connect and response timeouts. The adapter may perform one short jittered retry only for connection/timeouts and upstream `5xx`/overload statuses. It does not automatically retry `400`-class rejections, authentication errors, `402`, or `429`; sleeping through a free-tier reset would tie up the request and repeated calls could worsen throttling. The client receives retry guidance instead.

Structured logs include request ID, provider, configured model, operation (`embedding` or `summary`), latency, custom failure kind, upstream status, safe upstream request/generation ID when present, and retry count. They never include provider keys, prompts, document content, embeddings, raw provider messages, metadata, or response bodies.

Provider failure during document creation still inserts nothing because AI work completes before the database transaction. Provider failure during search returns the explicit provider error rather than silently falling back to incomplete client-only results. `/health` continues to report process/PostgreSQL health and does not fail merely because OpenRouter is temporarily unavailable; the failing AI-dependent request identifies that dependency directly.

References: [OpenRouter documented API error statuses](https://openrouter.ai/docs/client-sdks/python/api-reference/responses), [free-model limits](https://openrouter.ai/docs/faq), and [free-model availability constraints](https://openrouter.ai/docs/guides/routing/routers/free-router).

## 12. Testing Strategy

Tests do not consume live OpenRouter or DeepSeek quota by default.

### Unit tests

- Every concrete use case inherits `UseCase[CommandDTO | QueryDTO, ResultDTO]`, passes its DTO to the correct service, and returns the service result DTO unchanged.
- API handlers resolve only use cases through `FromDishka`; they do not receive services, repositories, pools, or AI adapters.
- API `*In` schemas map to command/query DTOs and result DTOs map to `*Out` schemas without losing or renaming data accidentally.
- HTTP and provider Pydantic schemas stay at their respective boundaries; service/repository signatures contain DTOs and never `BaseModel`, `asyncpg.Record`, or repository-model types.
- OpenRouter DTOs map to `EmbeddingRequestOut`/`ChatCompletionRequestOut`, and malformed `EmbeddingResponseIn`/`ChatCompletionResponseIn` payloads are rejected before they reach services.
- `OpenRouterErrorResponseIn` parses known error envelopes while accepting unknown additive metadata; malformed error JSON still maps from its HTTP status.
- Every OpenRouter status group and `httpx` network failure maps to the specified custom exception and `ProviderFailureDTO`.
- `register_exception_handlers(app)` produces the documented `AIProviderErrorOut`, `ErrorOut`, and `ValidationErrorOut` shapes without route-level provider handling.
- Provider error responses contain stable application messages and request IDs but never raw upstream bodies, provider messages, metadata, prompts, content, embeddings, or keys.
- Only network/timeouts and upstream overload/`5xx` errors receive one retry; `402` and `429` do not retry inside the request.
- API responses and provider payloads are valid UTF-8 JSON encoded/decoded through `orjson`.
- Services apply business behavior against fake repositories and a fake AI service without importing FastAPI, Dishka, asyncpg, or PyPika.
- Passage splitting preserves order, overlap, and all input text.
- Oversized paragraphs split into bounded chunks.
- Duplicate chunk matches collapse into one document using the best score.
- Client and document results merge in score order and stop at the internal 20-result cap.
- Document retrieval considers 60 internal chunk candidates before grouping, while `SearchIn` and OpenAPI expose no `limit` field.
- Malformed vectors are rejected.
- Generated queries contain numbered placeholders, keep values separate, and produce PostgreSQL syntax accepted by `asyncpg`.
- The custom pgvector distance term renders `<=> $1` without embedding the query vector in the SQL text.
- Repository model mappers consume `asyncpg.Record`-shaped data and produce the expected result DTOs; repository models never escape public repository methods.

### PostgreSQL/API integration tests

- A blank database upgrades to dbmate head, reports no pending migration, rolls back the initial migration, and upgrades cleanly again.
- The generated `db/schema.sql` matches a fresh database at migration head.
- Creating a valid client returns `201` and persists normalized data.
- Invalid email and social links return `422`.
- Creating a document for an unknown client returns `404` without calling the fake AI gateway.
- Creating a document stores its summary and all vectors transactionally.
- OpenRouter `402` and `429` responses become explicit provider-attributed `503` responses and leave no partial document.
- OpenRouter timeouts/overload become provider-attributed `503`; rejected or malformed responses become provider-attributed `502`.
- A valid upstream `Retry-After` is safely reflected for rate-limit failures, and `/health` remains independent of OpenRouter availability.
- `NevisWealth` finds `john.doe@neviswealth.com` through `pg_trgm`.
- A deterministic fake embedding makes `address proof` return a document chunk containing `utility bill`.
- Multiple matching chunks from the same document produce one search result.
- OpenAPI contains all required routes, suffixed schemas, and documented provider error responses, and the search operation exposes `q` but no `limit`.
- Mandatory authentication rejects missing and invalid bearer tokens before resolving a use case or calling the database/AI provider.

The test suite uses the Compose PostgreSQL service with both extensions enabled. Its test Dishka container replaces the external AI adapter while preserving the same UseCase -> Service -> Repository graph. Every pytest fixture is decorated normally and has the requested return annotation:

```python
T = TypeVar("T")
Fixture = Annotated[T, pytest.fixture]
```

For example, `def client(...) -> Fixture[AsyncClient]: ...`.

A separately documented opt-in smoke command may call OpenRouter once, but it is not part of CI.

## 13. Docker and Local Operation

`compose.yaml` contains three services:

- `db`: a pinned PostgreSQL image with pgvector, a persistent volume, and a `pg_isready` health check;
- `migrate`: a pinned dbmate image, the versioned `db/migrations/` directory, and a dependency on healthy PostgreSQL; it runs pending migrations once and must exit successfully;
- `api`: the project image, one Granian worker, environment loaded from `.env`, and a dependency on successful migration completion.

The database image receives no ad-hoc `/docker-entrypoint-initdb.d` schema file. dbmate is the only owner of schema creation and evolution, including on the first boot.

Only the API port is published. PostgreSQL remains on the Compose network. The API image runs as a non-root user and installs the locked `uv` environment without development dependencies.

Expected local workflow:

```bash
cp .env.example .env
docker compose up --build
docker compose run --rm migrate status
curl http://localhost:8000/health
open http://localhost:8000/docs
```

Developers create a revision with `docker compose run --rm migrate new <name>`, implement both directions, run `migrate`, then regenerate and commit `db/schema.sql`. `docker compose run --rm migrate rollback` is a development/CI verification command, not an automatic production recovery strategy.

The README includes complete authenticated `curl` examples for client creation, document creation, trigram search, and semantic search.

## 14. Deployment Options

Free offerings are suitable only for a demonstration and may change. The README should link to current provider limits instead of promising permanent free hosting.

| Option | Runs Compose unchanged? | Persistence | Recommendation |
| --- | --- | --- | --- |
| Oracle Cloud Always Free VM | Yes | Persistent block volume | Recommended always-on demo |
| Google Compute Engine free `e2-micro` | Yes | 30 GB standard persistent disk within current allowance | Fallback; memory is tight |
| Local Compose + Cloudflare Quick Tunnel | Yes, locally | Local disk | Fastest temporary reviewer demo |
| Render free services | No; deploy API and PostgreSQL separately | Free PostgreSQL expires after 30 days | Convenient temporary PaaS demo only |

### Recommended: Oracle Cloud Always Free VM

Oracle currently lists AMD and Arm compute plus block storage among its Always Free services. Provision one Ubuntu VM, install Docker Engine and the Compose plugin, clone the repository, create `.env` with both provider and API bearer keys, and run:

```bash
docker compose up -d --build
```

Expose only HTTP/HTTPS in the cloud firewall; never expose PostgreSQL. For a stable HTTPS endpoint, use a named Cloudflare Tunnel or a small Caddy reverse-proxy override with a domain. Confirm that selected images support the VM architecture before choosing Arm; otherwise select AMD compute.

Oracle notes that free capacity is limited and accounts idle for 30 days may be suspended, so this is not an SLA-backed service.

Reference: [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/)

### Fallback: Google Compute Engine

Google's current free tier includes one non-preemptible `e2-micro` instance in selected US regions, 30 GB-months of standard persistent disk, and limited outbound transfer. It can run Docker Compose unchanged, but its small memory allocation leaves little headroom for PostgreSQL and image builds. Build the API image in CI or locally, keep one Granian worker, and avoid local AI models.

Reference: [Google Cloud free Compute Engine allowance](https://docs.cloud.google.com/free/docs/free-cloud-features)

### Fastest temporary demo: Cloudflare Quick Tunnel

Run Compose locally, then expose the API without opening a router port:

```bash
cloudflared tunnel --url http://localhost:8000
```

Cloudflare assigns a random `trycloudflare.com` URL. Quick Tunnels are explicitly for testing, have no uptime guarantee, and disappear when the local process stops. They are useful for a scheduled review call, not as the submitted permanent deployment.

Reference: [Cloudflare Quick Tunnels](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)

### Render alternative

Render can build the API Dockerfile and supplies a public URL, but it does not run this Compose project as one unit. The database must become a separate Render PostgreSQL resource. Free web services sleep after inactivity, while free PostgreSQL currently expires after 30 days. This is acceptable for a short-lived review link but less faithful to the reproducible Compose deliverable.

Reference: [Render free-service limitations](https://render.com/docs/free)

## 15. Stack Roast and Alternatives

- **Python, uv, FastAPI:** good choices. They optimize reviewer familiarity and setup time.
- **orjson:** a small, maintained dependency that produces fast, strict UTF-8 JSON bytes and natively handles UUID, datetime, and dataclass values. Pydantic still owns validation; orjson is only serialization/deserialization plumbing.
- **Granian:** technically sound but unnecessary at assignment scale; Uvicorn would be the conventional choice. Keep Granian because it is requested and costs little once containerized.
- **Dishka:** used as the explicit composition root. API handlers receive concrete use cases through `FromDishka[...]`; services, repositories, pools, and adapters remain hidden behind the UseCase boundary and are constructor-injected by Dishka.
- **PostgreSQL + pg_trgm:** exactly the right client-search tool. Elasticsearch would solve the problem too, but at a much higher operational cost.
- **asyncpg:** a better fit than Psycopg for this async-only, ORM-free service. Its built-in pool and `fetch`/`fetchrow` API keep database code small.
- **PyPika:** adds a dependency, but it is explicitly requested and gives composable PostgreSQL query construction. Keep query construction inside repositories, use `Parameter("$n")` for every value, and do not interpolate application values.
- **dbmate:** a better fit than Alembic here because migrations are hand-written SQL and the application intentionally has no SQLAlchemy dependency. It provides versioning, transactional upgrades, rollback, status, and schema dumps without influencing repository code. Yoyo is not selected because its release activity is stale; Alembic remains the alternative if SQLAlchemy is later adopted.
- **pgvector:** the simplest semantic-search extension because the canonical records already belong in PostgreSQL. Its `<=>` operator requires only a minimal custom PyPika term.
- **OpenRouter free tier:** excellent for a reproducible demo, weak for production reliability and privacy. Configuration must allow later provider replacement.
- **DeepSeek summaries:** a good inexpensive alternative when the user supplies a key, but not an embeddings replacement.
- **Elasticsearch:** justified only when corpus size, advanced analyzers, faceting, operational search tooling, or measured PostgreSQL search limitations demand it. None is present here.

## 16. Implementation Order

1. Initialize the `uv` project with FastAPI, Granian, Dishka, asyncpg, PyPika, Pydantic settings, httpx, pgvector, orjson, and test dependencies; add the Dockerfile and Compose database.
2. Add pinned dbmate tooling, the initial up/down migration, and generated `db/schema.sql`; verify migrate -> status -> rollback -> migrate against fresh PostgreSQL.
3. Define the immutable slotted dataclass command/query/result/AI/failure DTOs, custom application exceptions, and the generic abstract `UseCase` contract.
4. Define internal repository dataclass models and their record/DTO mappers.
5. Implement required bearer authentication, settings, the `asyncpg` pool lifecycle, and Dishka composition.
6. Implement repositories with PyPika `Parameter("$n")` queries and PostgreSQL integration tests.
7. Implement services with fake-repository unit tests, including AI DTO coordination, chunk grouping, the internal 20-result cap, and the larger fixed chunk candidate set.
8. Implement the OpenRouter boundary with directional success/error Pydantic schemas, orjson, httpx, status-to-exception mapping, bounded retries, and mocked provider tests.
9. Implement concrete use cases and register the full API -> UseCase -> Service -> Repository graph in Dishka.
10. Implement API `schemas/`, global exception-handler registration, and routes that map `*In` -> DTO -> `*Out` while injecting only use cases through `FromDishka[...]`.
11. Verify trigram search, vector search, chunk grouping/capping, provider-error attribution, authentication, DTO boundaries, OpenAPI, migrations, and manual Compose workflows.
12. Add README examples and deployment notes, then make small Conventional Commits aligned with independently working increments.

The implementation plan will expand these items into test-first tasks with exact files, commands, expected failures, and commit boundaries after this design is reviewed.

## 17. Acceptance Criteria

- A clean checkout starts with `docker compose up --build` after adding the required API bearer token and OpenRouter key.
- The one-shot dbmate service upgrades a blank database before the API starts; status, rollback, re-upgrade, and schema-dump workflows are documented and tested.
- Swagger UI documents every required endpoint and extended response.
- FastAPI handlers resolve concrete use cases through `FromDishka[...]` and never receive services or repositories directly.
- API Pydantic schemas live under `src/api/schemas/`, use directional `In`/`Out` suffixes, and never use the API term `model`.
- OpenRouter request/response Pydantic schemas use application-relative `Out`/`In` suffixes and are confined to the provider adapter.
- Every concrete use case inherits the generic abstract `UseCase[CommandDTO | QueryDTO, ResultDTO]` contract.
- Every inter-layer value is an immutable slotted dataclass DTO; Pydantic schemas, dictionaries, `asyncpg.Record`, and repository models do not cross their owning boundaries.
- Repositories define internal typed persistence models, own all asyncpg/PyPika queries, and expose DTO-based methods to services.
- API responses and OpenRouter JSON payloads use orjson.
- `GET /search` exposes only the requested `q` parameter, applies a fixed internal 20-result cap, and retrieves a larger internal chunk candidate set before grouping.
- Client creation and document creation return `201` with generated IDs.
- `NevisWealth` returns a client whose email contains `neviswealth.com`.
- `address proof` can return a document passage containing `utility bill` using semantic similarity rather than a hard-coded synonym list.
- Long documents are searchable across all chunks and appear only once per response.
- Created documents include a stored quick summary.
- Every business endpoint requires a privately shared bearer credential in local and deployed environments and never exposes provider keys.
- OpenRouter quota exhaustion, rate limiting, unavailability, rejection, and malformed responses map through registered custom exception handlers to explicit provider-attributed `AIProviderErrorOut` responses.
- Provider errors never expose upstream bodies, messages, metadata, prompts, document content, embeddings, or credentials, and external AI failure produces no partial data.
- Core and edge-case tests pass without live external API calls.
- The README explains setup, examples, provider limitations, DeepSeek summary configuration, and free deployment options.
- `requirements.md` and `task.pdf` remain untracked.
