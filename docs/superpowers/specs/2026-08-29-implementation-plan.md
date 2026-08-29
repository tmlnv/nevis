# Nevis — Implementation Plan (revised)

Supersedes the build-order and layout sections of `2026-08-29-nevis-search-api-design.md`.
Design decisions not mentioned here are kept as written in that document.

## Status: implemented 2026-08-29

All phases complete. Verified on a clean `docker compose down -v && up --build`:
migrations apply via the one-shot dbmate job, the corpus seeds against the real
OpenRouter API, both acceptance queries pass, and `uv run pytest` is 47 green.
Three defects that live testing caught are recorded in the README's model-choice
section: the `openrouter/free` router returning empty summaries ~33% of the time,
a document score threshold too loose to filter noise, and the deprecated
`ORJSONResponse`.

## Guiding constraint

10–14 hours. The task PDF grades "correctness, clarity, and simplicity" and
`requirements.md` says "keep number of lines of code as minimal as possible".
Every layer must earn its place against that.

## Revised layout (~13 files, was ~30)

```text
src/
  main.py        # create_app, lifespan (asyncpg pool), dishka setup
  config.py      # pydantic-settings
  di.py          # dishka providers (APP: settings, pool, httpx, repos, ai;
                 #                   REQUEST: services, use cases)
  auth.py        # bearer dependency
  errors.py      # exceptions + register_exception_handlers
  schemas.py     # all HTTP *In/*Out pydantic schemas
  dto.py         # frozen slotted dataclasses: commands, queries, results, AI results
  use_cases.py   # UseCase ABC + CreateClient / CreateDocument / Search
  routes.py      # 4 routes, FromDishka[ConcreteUseCase]
  services.py    # ClientService, DocumentService, SearchService
  repo.py        # ClientRepo, DocumentRepo — raw SQL, $n params
  ai/
    client.py    # OpenRouter adapter (httpx + orjson)
    schemas.py   # provider-boundary *Out/*In pydantic schemas
db/
  migrations/20260829000100_initial_schema.sql
  schema.sql     # dbmate dump, committed, never hand-edited
tests/
  conftest.py
  test_clients.py
  test_documents.py
  test_search.py
  test_migrations.py
compose.yaml
Dockerfile
.env.example
README.md
pyproject.toml
```

Dependency direction unchanged: routes → use cases → services → repo/ai.

## Cut from the original design

| Cut | Why | Add back when |
| --- | --- | --- |
| PyPika | not requested anywhere in `requirements.md`; cannot express `<=>`, `similarity`, `word_similarity` without custom Terms | queries become dynamically composed |
| Repository `*Model` dataclasses | `ClientModel` and `ClientResultDTO` carry identical fields; one exists only to be copied into the other. Map `Record` → result DTO in a one-line mapper | persistence shape genuinely diverges from the result shape |
| `UsageIn` | nothing reads provider token counts | usage is logged or billed |
| `OpenRouterErrorIn`, `OpenRouterErrorResponseIn` | the error body must never be exposed or logged, so `response.status_code` is the entire signal taken from it | — |
| `EmbeddingRequestDTO`, `SummaryRequestDTO` | wrappers around a list of strings and a string | they carry options beyond the text |
| 9 exception classes + 8-row mapping table | one `AIProviderError(status, code, retryable)` + one handler keeps the reviewer-visible value | — |
| `tests/test_boundaries.py` | asserts the code is shaped like the doc, not that it works | — |
| DeepSeek as a second provider path | `SUMMARY_BASE_URL`/`SUMMARY_MODEL` are already OpenAI-compatible; it is 3 env vars and a README line | — |
| Tuned 60-chunk candidate set | take top 20 chunks, group by document | measured recall loss |

Kept deliberately after review: the `UseCase` layer and generic ABC, dishka,
dbmate, chunking, and the directional `*In`/`*Out` schemas at **both** trust
boundaries.

### The AI boundary

OpenRouter's response is untrusted input and gets the same treatment as an HTTP
request body. Shape validation is not what the service does — the service checks
*semantics* (vector count, 1024 dimensions, finite floats, non-empty summary),
pydantic checks *shape*. Both are real and different; a malformed envelope should
fail in the adapter, not as a `KeyError` three layers up.

`src/ai/schemas.py` keeps 7 of the original 11, named from this application's
viewpoint (a request to OpenRouter is `Out`, its response is `In`):

| Direction | Schema |
| --- | --- |
| Out | `EmbeddingRequestOut` |
| In | `EmbeddingResponseIn`, `EmbeddingDataIn` |
| Out | `ChatCompletionRequestOut`, `ChatMessageOut` |
| In | `ChatCompletionResponseIn`, `ChatChoiceIn`, `ChatMessageIn` |

All `*In` schemas set `extra="ignore"` so additive provider metadata never breaks
parsing.

The adapter exposes two methods, both raising `AIProviderError`:

```python
async def embed(self, texts: Sequence[str]) -> EmbeddingResultDTO: ...
async def summarize(self, text: str) -> SummaryResultDTO: ...
```

`EmbeddingResultDTO(vectors, model)` and `SummaryResultDTO(text, model)` are
returned rather than bare values because `documents.embedding_model` and
`documents.summary_model` persist that provenance. Provider schemas stay inside
the adapter and never cross into services.

### Record mapping

Register the codecs once on pool init rather than building a parallel dataclass
hierarchy to work around them: `set_type_codec` for `jsonb` (orjson) and
pgvector's `register_vector`. Then `social_links` arrives as a `list` and
`embedding` as a sequence of floats, and each mapper is one line
(`ClientResultDTO(**record)`).

## Phases

Each phase ends in one conventional commit on `main`. Do not batch commits to the end.

### Phase 0 — spike — DONE, 2026-08-29

Run against the live API. Two blockers found; decisions below are settled.

**1. `input_type` is a no-op — removed from the design.**
`input_type: "total_nonsense_value"` returns `200`. Embedding the same text with
`search_query` and `search_document` yields cos = `1.000000` — byte-identical
vectors. OpenRouter accepts and discards the field. Embed queries and documents
identically; delete every mention of asymmetric embedding.

**2. `liquid/lfm-2.5-embedding-350m:free` fails the task's own acceptance
example — model changed to `nvidia/nemotron-3-embed-1b:free`.**

For `q = "address proof"`, the liquid model ranks the literal string
`"Utility bill"` **5th of 6, below a banana bread recipe**. Adding e5-style
(`query:`/`passage:`) or explicit-instruction prefixes made it worse, not better.

| Rank | liquid (bare) | nemotron-3-embed-1b |
| --- | --- | --- |
| 1 | tenancy agreement `0.265` | tenancy agreement `0.172` |
| 2 | passport copy `0.165` | electricity utility bill `0.164` |
| 3 | electricity utility bill `0.156` | **Utility bill `0.140`** |
| 4 | banana bread recipe `0.080` | passport copy `0.118` |
| 5 | **Utility bill `0.045`** | banana bread recipe `0.096` |
| 6 | portfolio report `-0.028` | portfolio report `0.047` |

Nemotron puts both utility-bill documents above every irrelevant one, which is
what the task requires. (A tenancy agreement outranking them is defensible — it
*is* address proof.) It is free, and its context is 32k tokens rather than 512.

**3. Nemotron is 2048-dimensional and pgvector cannot HNSW-index that.**
`dimensions` is not reducible — the API replies `"dimensions must be one of
2048"`. pgvector caps `hnsw`/`ivfflat` on `vector` at **2000** dimensions. Store
the column as `halfvec(2048)` and index with `halfvec_cosine_ops` (pgvector
≥ 0.7, present in `pgvector/pgvector:pg17`). Half precision is irrelevant at
this recall scale. Note this tradeoff in the README — it is exactly the kind of
constraint the reviewer is looking for.

**4. Confirmed working.** Response envelope is
`{object, data: [{object, embedding, index}], model, usage, provider, id}`;
a 32-input batch returns 32 items with `index` in order. `openrouter/free`
routes to `nvidia/nemotron-3-super-120b-a12b:free` and returns a clean
two-sentence summary with `finish_reason: "stop"`.

**Consequences for later phases**

- `.env.example`: `EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b:free`, and the
  real key goes in `.env` only (already gitignored).
- Migration: `embedding halfvec(2048)`, HNSW `halfvec_cosine_ops`.
- Chunking is no longer needed for the context limit (32k tokens covers the
  100k-char cap). Keep it anyway, but for the honest reason: it produces
  `matched_excerpt` and improves precision on long documents. Say so in the README.
- Nemotron's scores are low and compressed — ~`0.17` relevant vs ~`0.10` noise.
  The document threshold must be tuned low against the seed data in Phase 6, and
  the raw cosine should be rescaled before it is exposed as `score`.

### Phase 1 — skeleton

`uv init`; deps: fastapi, granian, dishka, asyncpg, pgvector, pydantic-settings,
httpx, orjson, pytest, pytest-asyncio, ruff.

`compose.yaml` with three services as the design specifies: `db`
(`pgvector/pgvector:pg17`, `pg_isready` healthcheck, volume), `migrate` (pinned
`ghcr.io/amacneil/dbmate`, one-shot, `depends_on: db healthy`), `api`
(`depends_on: migrate completed_successfully`). Only the API port published.

Initial migration enables `pg_trgm` + `vector` and creates all three tables and
indexes with a reversing `-- migrate:down`. Commit the `dbmate dump` output as
`db/schema.sql`. `GET /health` does `SELECT 1`. `.env.example`. Dockerfile,
non-root, `uv sync --no-dev`.

Done when `docker compose up --build` serves `/health` and `/docs`, and
`migrate → status → rollback → migrate` round-trips against a fresh database.

`chore: project skeleton with compose, dbmate migrations and health check`

### Phase 2 — clients + trigram search

`POST /clients` and the client half of `GET /search`.

Client search SQL — put the index-usable operators in `WHERE`, the score in `SELECT`:

```sql
SELECT *, GREATEST(similarity(search_text, $1), word_similarity($1, search_text)) AS score
FROM clients
WHERE search_text %  $1
   OR search_text %> $1
   OR search_text LIKE '%' || lower($1) || '%'
ORDER BY score DESC LIMIT 20
```

Integration test against real Postgres: `NevisWealth` → `john.doe@neviswealth.com`.

`feat: client creation and trigram client search`

### Phase 3 — documents

`POST /clients/{id}/documents`. Split on paragraphs, ~1200-char windows with
overlap, one batched embeddings call (batching verified: 32 inputs, `index`
ordered), one summary call, then a single transaction inserting document +
chunks. 404 before any AI call. 100k char cap.

Tests use a fake `ai.py` bound through the dishka container — no live calls.

`feat: document ingestion with chunked embeddings and summary`

### Phase 4 — semantic search + merge

Query embedding → top 20 chunks by `embedding <=> $1` (cast the parameter to
`halfvec`) → group by `document_id`, keep best chunk as `matched_excerpt`.

**Ranking (changed from the design):** trigram similarity and cosine similarity
are different scales; do not sort one merged list by raw score. Apply a
per-source threshold, emit clients then documents, each internally sorted.
Cap 20 total. Document this choice in the README as a deliberate tradeoff.

Cache the query embedding — `@lru_cache` on the normalized query string — so a
reviewer repeating a query does not spend free-tier quota.

`feat: semantic document search and merged search endpoint`

### Phase 5 — auth and errors

Bearer dependency with `hmac.compare_digest`, `WWW-Authenticate: Bearer` on 401.
`errors.py`: `ClientNotFound`, `AIProviderError(status, code, retryable)`,
`DatabaseUnavailable` + `register_exception_handlers(app)`. Map upstream
401/402/429/5xx/timeout → `503 ai_provider_unavailable` with `provider` and
`retryable`; upstream 4xx rejections and malformed 2xx → `502`. Never log or
return keys, prompts, or document text.

**Deviation from the design:** if the provider fails during *search*, return the
client results with a `degraded: true` marker rather than 503. The graded
endpoint should not go dark because a free tier is throttled.

`feat: bearer auth and provider-attributed error handling`

### Phase 6 — README, seed, deploy

README: setup, authenticated curl examples with real responses, the ranking
tradeoff, the embedding-retention warning (verified: the model card does state
requests may be retained for training — synthetic data only), DeepSeek env vars,
deploy notes. A `seed.py` or `seed.sql` with ~6 clients and ~8 documents so the
reviewer sees results immediately.

`docs: readme with setup, examples and deployment notes`

## Deployment

Reordered from the design. Oracle Always Free is a poor primary choice — new
accounts routinely hit "Out of host capacity" for days, which is not a risk worth
carrying into a deadline.

1. **Fly.io** — runs the Dockerfile, has a managed Postgres, real URL, stays up.
2. **Cloudflare named tunnel** over local compose — free, stable hostname,
   zero infra, but only up while your machine is.
3. Oracle Always Free — best if you already have a provisioned instance.
4. Render — API and Postgres split, free Postgres expires in 30 days.

Whichever is chosen, share the bearer token with the reviewer out-of-band.

## Git

No `main` branch currently exists — the two commits sit on
`docs/nevis-search-api-design`. Both are trunk content, so rename rather than
branch off:

```bash
git branch -m docs/nevis-search-api-design main
```

No remote for now. Work locally, **trunk-based**: small conventional commits
straight onto `main`, one per phase. A solo three-day assignment with no reviewer
gains nothing from feature branches, and a linear log is what a grader actually
reads. Branch only for a risky spike you might throw away (e.g.
`spike/openrouter-embeddings`) and delete it after.

Add the remote whenever convenient — nothing in the workflow above changes:

```bash
git remote add origin git@gitlab.com:tmlnv/nevis.git
git push -u origin main
```

Keep `task.pdf` and `requirements.md` untracked — already covered by `.gitignore`.
