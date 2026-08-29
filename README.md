# Nevis Search API

A small REST API that stores clients and their documents and searches across both:

- **clients** by fuzzy text over name, email and description (PostgreSQL `pg_trgm`);
- **documents** by meaning rather than keywords (dense embeddings in `pgvector`), so
  `address proof` finds a document about a `utility bill`;
- every document gets a **stored 2–3 sentence summary** at ingestion time.

One PostgreSQL instance provides relational storage, fuzzy search and vector search in
the same transactional system. There is no separate search service to keep in sync.

## Quick start

```bash
cp .env.example .env          # then set API_BEARER_TOKEN and OPENROUTER_API_KEY
docker compose up --build     # db -> migrate (one-shot) -> api
```

Then open <http://localhost:8000/docs> and authorise with your bearer token.

```bash
curl -s localhost:8000/health          # {"status":"ok","is_database_ready":true}
export TOKEN=...                       # the value of API_BEARER_TOKEN
uv run python seed.py                  # optional: load the demo corpus
```

An OpenRouter key is required. The free tier is sufficient; get one at
<https://openrouter.ai/keys>.

## API

Every endpoint except `/health`, `/docs` and `/openapi.json` requires
`Authorization: Bearer $TOKEN`. A missing or wrong token returns `401` before any
database or AI work happens.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/clients` | Create a client |
| POST | `/clients/{id}/documents` | Add a document; embeds and summarises it |
| GET | `/search?q=...` | Search clients and documents |
| GET | `/health` | Process + database liveness |

### Create a client

```bash
curl -s -X POST localhost:8000/clients \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"first_name":"John","last_name":"Doe","email":"john.doe@neviswealth.com",
       "description":"Long-standing private client, balanced growth mandate.",
       "social_links":["https://www.linkedin.com/in/johndoe"]}'
```

`201 Created` returns the client with its generated `id`.

### Add a document

```bash
curl -s -X POST localhost:8000/clients/$CLIENT_ID/documents \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"title":"Utility bill","content":"Electricity utility bill issued by City Power ..."}'
```

`201 Created`. The response extends the assignment's `Document` schema with `summary`:

```json
{
  "id": "ef5c0500-d5fe-4595-b325-6db52075186e",
  "client_id": "d404e22c-b872-4e52-930c-dee1c133aaee",
  "title": "Utility bill",
  "content": "Electricity utility bill issued by City Power ...",
  "summary": "The City Power electricity bill for John Doe at 14 Oak Lane, Bristol, covers the period 1 March 2026–31 March 2026 and shows a total due of GBP 84.20, payable by 15 April 2026. This bill is commonly used as proof of residential address during onboarding verification.",
  "created_at": "2026-08-29T17:19:21.003426Z"
}
```

A missing client returns `404` **before** any AI call is made. If the AI provider fails,
nothing is written — all external calls complete before the database transaction opens.

### Search

Fuzzy client match — the assignment's first example:

```bash
curl -s -G localhost:8000/search --data-urlencode 'q=NevisWealth' -H "Authorization: Bearer $TOKEN"
```

```json
[
  {
    "result_type": "client",
    "score": 1.0,
    "client": {
      "id": "d404e22c-b872-4e52-930c-dee1c133aaee",
      "first_name": "John", "last_name": "Doe",
      "email": "john.doe@neviswealth.com",
      "description": "Long-standing private client, balanced growth mandate.",
      "social_links": ["https://www.linkedin.com/in/johndoe"]
    }
  }
]
```

Semantic document match — the assignment's second example. Note that nothing in the
query appears in the document; the match is by meaning:

```bash
curl -s -G localhost:8000/search --data-urlencode 'q=address proof' -H "Authorization: Bearer $TOKEN"
```

```json
[
  {
    "result_type": "document",
    "score": 0.288,
    "matched_excerpt": "Utility bill\n\nElectricity utility bill issued by City Power for the billing period 1 March 2026 to 31 March 2026. Service address: 14 Oak Lane, Bristol ...",
    "document": { "id": "ef5c0500-...", "title": "Utility bill", "summary": "...", "...": "..." }
  },
  {
    "result_type": "document",
    "score": 0.2467,
    "matched_excerpt": "Signed tenancy agreement\n\nAssured shorthold tenancy agreement confirming that the tenant resides at 42 Cathedral Road, Cardiff ...",
    "document": { "id": "79855a07-...", "title": "Signed tenancy agreement", "...": "..." }
  }
]
```

The response is a bare top-level array, as specified. Each element is discriminated by
`result_type`. `matched_excerpt` shows the passage that actually matched, so a caller can
see *why* a document was returned.

## Design decisions

### One database, two search strategies

Client identity fields are short and typo-prone, so they use trigram similarity — this
matches `NevisWealth` against `john.doe@neviswealth.com` without any embedding call.
Documents are prose, so they use embeddings. Elasticsearch would solve both, but adds a
second datastore to synchronise for no gain at this size.

The trigram query keeps the index-usable operators in `WHERE` and computes the score in
`SELECT`, so the GIN index is genuinely used rather than decorative:

```sql
WHERE search_text % $1 OR search_text %> $1 OR search_text LIKE '%' || lower($1) || '%'
```

`pg_trgm.similarity_threshold` is lowered to `0.15` by the initial migration
(`ALTER DATABASE ... SET`). At the 0.3 default the `%` operator matched *nothing*:
`search_text` concatenates name, email and description, which dilutes whole-string
similarity.

This is set on the database rather than per query on purpose. `%` is the only
index-usable trigram predicate — the explicit `similarity(a, b) > 0.15` form is a
sequential scan, verified with `EXPLAIN` — and `%` reads that threshold, so the value is a
*precondition of the query*, not an incidental tweak. Setting it once beside the index it
partners with means no extra round-trip per search and no way for a future query using `%`
to silently match nothing because someone forgot to set it.

Two consequences worth knowing: it applies to new sessions only (fine here — the one-shot
migrate service finishes before the API opens its pool), and `pg_dump --schema-only` does
not emit `ALTER DATABASE` settings, so `db/schema.sql` does not show it. The migration
remains the authoritative definition. Applying it also requires ownership of the database,
which the migration user has.

### Client and document scores are not merged into one ranking

Trigram similarity and cosine similarity are different scales — a trigram score of 0.4 is
a strong name match, a cosine of 0.4 is excellent, and 0.25 is already a good semantic
match. Sorting one merged list by raw score produces arbitrary interleaving. Instead each
source is filtered by its own threshold and sorted within itself, and clients are listed
before documents. `score` communicates rank within a group; it is not a calibrated
probability.

Thresholds were calibrated against the seeded corpus rather than guessed. Across eight
queries, clearly-unrelated documents scored at most **0.174** while genuine matches
started at **0.233**, so the document floor is **0.20**. `banana bread recipe` correctly
returns nothing.

### Search degrades instead of failing

Semantic search needs a live embedding call. If the provider is rate-limited or down,
`GET /search` still returns `200` with the client results and sets `X-Search-Degraded:
true`, rather than failing the whole request. The wire format is a bare array, so the
signal goes in a header. Document *creation* behaves the opposite way — a provider
failure there is a hard error, because storing a document without its embedding would
leave it permanently unsearchable.

Query embeddings are cached in-process (bounded LRU, 256 entries), so repeated searches
do not spend free-tier quota.

### Chunking

Documents are split on paragraph boundaries into ~1200-character passages with ~150
characters of overlap, and each passage is embedded separately. The chosen model has a
32k-token context, so chunking is *not* needed to fit the input — it earns its place by
producing `matched_excerpt` and by keeping precision on long documents. Chunk hits are
grouped by document, keeping the best-scoring chunk, so one long document cannot occupy
several result slots.

### Model choices, and why they are pinned

Both were selected by measurement, not by reputation.

**Embeddings — `nvidia/nemotron-3-embed-1b:free`.** The obvious free candidate,
`liquid/lfm-2.5-embedding-350m:free`, *fails the assignment's own example*: for
`address proof` it ranks the literal string `Utility bill` fifth of six, below a banana
bread recipe. Adding `query:`/`passage:` or instruction prefixes made it worse. Nemotron
ranks both utility-bill documents above every unrelated one.

**Summaries — `nvidia/nemotron-3-super-120b-a12b:free`, pinned deliberately.**
`openrouter/free` is a *router* that picks an arbitrary free model per call. Measured over
six calls it selected a content-safety model and a code model, and returned an empty
message 33% of the time: reasoning models spent the whole token budget on reasoning and
stopped at `finish_reason: "length"` before emitting content. `max_tokens` is 400 for
headroom.

**`input_type` is not used.** OpenRouter accepts `input_type: "search_query"` /
`"search_document"` and silently ignores it — a garbage value returns `200`, and the same
text embedded both ways yields byte-identical vectors (cosine `1.000000`). Queries and
documents are therefore embedded identically.

### `halfvec`, not `vector`

Nemotron returns 2048 dimensions and will not reduce them (`dimensions must be one of
2048`). pgvector caps HNSW indexes on `vector` at 2000 dimensions, so a `vector(2048)`
column cannot be indexed at all. Chunks are stored as `halfvec(2048)` — two bytes per
component instead of four — and indexed with `halfvec_cosine_ops`. Half precision does
not measurably affect cosine ranking at this scale.

Ordering is by `embedding <=> $1` ascending rather than by the derived score descending,
because only the former can use the HNSW index.

### Privacy

The free OpenRouter embedding endpoints state that successful requests and embeddings may
be retained and used for training. **The demo uses synthetic data only.** A real
deployment must move to a zero-data-retention endpoint, a provider covered by a data
processing agreement, or local inference before accepting client documents.

## Architecture

```
API (FastAPI)        bearer auth, validation, HTTP mapping
  |  *In schema -> command/query DTO ;  result DTO -> *Out schema
  v
UseCase              one application workflow each
  v
Service              normalisation, chunking, AI coordination, ranking
  v
Repository ------> OpenRouter adapter (httpx + orjson)
  v
PostgreSQL (asyncpg, raw SQL) + pg_trgm + pgvector
```

Dependencies point one way only. A route never receives a service, repository, pool or AI
client — Dishka injects a concrete use case via `FromDishka[...]`, and everything below is
constructor-injected. Pydantic schemas live at the two trust boundaries (HTTP and the AI
provider) and never cross into services; between layers, everything is a frozen slotted
dataclass DTO.

```
src/
  main.py config.py di.py auth.py errors.py
  schemas.py     HTTP boundary *In/*Out
  dto.py         inter-layer messages
  routes.py use_cases.py services.py repo.py
  ai/            provider adapter + its own boundary schemas
db/migrations/   dbmate, raw SQL, up and down
```

Deliberately excluded: user accounts and roles (one shared bearer key protects the API and
the provider quota), update/delete endpoints (not requested), background jobs (synchronous
ingestion is sufficient here), an ORM, and caller-supplied `limit`/pagination (not in the
given contract; the internal cap is 20).

## Migrations

`dbmate` owns the schema. It runs as a one-shot Compose service that waits for a healthy
database, applies pending migrations and exits; the API only starts once it has succeeded,
so no worker can race to alter the schema. `db/schema.sql` is generated (`dbmate dump`) and
committed for review — never hand-edited.

```bash
docker compose run --rm migrate status
docker compose run --rm migrate new add_something
docker compose run --rm migrate rollback
```

## Tests

```bash
uv run pytest
```

Tests never call OpenRouter — a deterministic fake AI gateway is substituted in the Dishka
container while the real UseCase → Service → Repository graph is exercised. Integration
tests start a throwaway `pgvector/pgvector:pg17` container automatically, so no manual
setup is needed. Set `TEST_DATABASE_URL` to point at your own database instead.

## Deployment

Compose runs the whole system as one unit, so anything that runs Docker works.

| Option | Notes |
| --- | --- |
| **Fly.io** | Builds the Dockerfile directly, managed Postgres, stays up. Best first choice. |
| **Cloudflare named tunnel** | `cloudflared tunnel --url http://localhost:8000` over local Compose. Free and instant, but only up while your machine is. |
| **Oracle Cloud Always Free** | Genuinely free and always-on *if* you can get capacity — new accounts frequently hit "Out of host capacity". |
| **Render** | API and Postgres deploy separately; free Postgres expires after 30 days. |

Expose only the API port. PostgreSQL stays on the internal network. Free tiers change
often — check current limits before relying on one.

## Using DeepSeek for summaries

The summary client is a plain OpenAI-compatible chat-completions client, so any such
endpoint works. No code change:

```dotenv
SUMMARY_BASE_URL=https://api.deepseek.com
SUMMARY_API_KEY=your-deepseek-key
SUMMARY_MODEL=deepseek-chat
```

DeepSeek is an alternative for summaries only — its first-party API exposes no embeddings
endpoint. There is no automatic fallback between providers: a configured provider either
works or returns a clearly attributed error, which keeps failures visible instead of
silently degrading quality.

## Errors

Provider problems are reported as provider problems, so a caller never misdiagnoses the
API or PostgreSQL as broken:

```json
{
  "code": "ai_provider_rate_limited",
  "message": "OpenRouter is temporarily unavailable because its free-tier limit has been reached.",
  "provider": "openrouter",
  "upstream_status": 429,
  "is_retryable": true,
  "retry_after_seconds": 60,
  "request_id": "4e85b76c-5007-4cd5-9a66-9b67c2c302f9"
}
```

An upstream `429` becomes this API's `503`, not `429`: the exhausted quota belongs to the
application's shared dependency, not to the caller. Upstream request bodies, provider
messages, prompts, document content, embeddings and API keys are never returned or logged.
`/health` deliberately does not call the AI provider, so a provider outage never marks the
container unhealthy.
