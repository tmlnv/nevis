-- migrate:up
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE clients (
    id uuid PRIMARY KEY,
    first_name text NOT NULL,
    last_name text NOT NULL,
    email text NOT NULL,
    description text,
    social_links jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    search_text text GENERATED ALWAYS AS (
        lower(first_name || ' ' || last_name || ' ' || email || ' ' || coalesce(description, ''))
    ) STORED
);

CREATE INDEX clients_search_text_trgm_idx ON clients USING gin (search_text gin_trgm_ops);

CREATE TABLE documents (
    id uuid PRIMARY KEY,
    client_id uuid NOT NULL REFERENCES clients (id) ON DELETE CASCADE,
    title text NOT NULL,
    content text NOT NULL,
    summary text NOT NULL,
    embedding_model text NOT NULL,
    summary_model text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX documents_client_id_idx ON documents (client_id);

-- 2048 dimensions exceeds pgvector's 2000-dimension limit for an hnsw index on
-- `vector`, so chunks are stored as `halfvec` (2 bytes per component instead of
-- 4). Half precision does not measurably change cosine ranking at this scale.
CREATE TABLE document_chunks (
    id uuid PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    position integer NOT NULL,
    content text NOT NULL,
    embedding halfvec(2048) NOT NULL,
    UNIQUE (document_id, position)
);

CREATE INDEX document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding halfvec_cosine_ops);

-- migrate:down
DROP TABLE document_chunks;
DROP TABLE documents;
DROP TABLE clients;
DROP EXTENSION vector;
DROP EXTENSION pg_trgm;
