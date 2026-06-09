-- =============================================================================
-- TaxBot baseline migration
--
-- Provisions the parent-child retrieval schema used by the RAG pipeline:
--   * Enables `pgvector`
--   * Creates `parent_nodes` (full markdown sections / tables)
--   * Creates `child_nodes` (sentence-level summaries with 1024-d embeddings)
--   * Creates `ingested_documents` (state table for the metadata-or-hash delta)
--   * Adds FTS, JSONB metadata, and HNSW indexes for hybrid retrieval
--
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;     -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- trigram indexes for fuzzy FTS

-- -----------------------------------------------------------------------------
-- Ingestion state: one row per IRS document URL we have processed.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingested_documents (
    doc_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_number    TEXT        NOT NULL,
    doc_title     TEXT        NOT NULL,
    pdf_url       TEXT        NOT NULL UNIQUE,
    revision_date TEXT        NOT NULL,
    posted_date   TEXT        NOT NULL,
    tax_year      INTEGER,
    category      TEXT        NOT NULL,
    language      TEXT        NOT NULL DEFAULT 'en',
    pdf_sha256    TEXT,
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ingested_documents_doc_number_idx
    ON ingested_documents (doc_number);
CREATE INDEX IF NOT EXISTS ingested_documents_tax_year_idx
    ON ingested_documents (tax_year);
CREATE INDEX IF NOT EXISTS ingested_documents_category_idx
    ON ingested_documents (category);
CREATE INDEX IF NOT EXISTS ingested_documents_language_idx
    ON ingested_documents (language);

-- -----------------------------------------------------------------------------
-- Parent nodes: large, contextual blocks (full sections, complete tables).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parent_nodes (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id       UUID NOT NULL
        REFERENCES ingested_documents (doc_id) ON DELETE CASCADE,
    text_content TEXT        NOT NULL,
    metadata     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS parent_nodes_doc_id_idx
    ON parent_nodes (doc_id);

-- Metadata pre-filters used by stage 1 of the hybrid retriever.
CREATE INDEX IF NOT EXISTS parent_nodes_tax_year_idx
    ON parent_nodes ((metadata ->> 'tax_year'));
CREATE INDEX IF NOT EXISTS parent_nodes_form_number_idx
    ON parent_nodes ((metadata ->> 'form_number'));
CREATE INDEX IF NOT EXISTS parent_nodes_doc_type_idx
    ON parent_nodes ((metadata ->> 'doc_type'));

-- Full-text search across the verbatim parent content (English config).
CREATE INDEX IF NOT EXISTS parent_nodes_text_fts_idx
    ON parent_nodes
    USING GIN (to_tsvector('english', text_content));

-- -----------------------------------------------------------------------------
-- Child nodes: small semantic units (sentences, table abstracts) + vectors.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS child_nodes (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_id    UUID NOT NULL
        REFERENCES parent_nodes (id) ON DELETE CASCADE,
    text_summary TEXT          NOT NULL,
    embedding    VECTOR(1024)  NOT NULL,
    metadata     JSONB         NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS child_nodes_parent_id_idx
    ON child_nodes (parent_id);

CREATE INDEX IF NOT EXISTS child_nodes_tax_year_idx
    ON child_nodes ((metadata ->> 'tax_year'));
CREATE INDEX IF NOT EXISTS child_nodes_form_number_idx
    ON child_nodes ((metadata ->> 'form_number'));
CREATE INDEX IF NOT EXISTS child_nodes_doc_type_idx
    ON child_nodes ((metadata ->> 'doc_type'));

-- HNSW cosine index for vector search. HNSW yields constant-time recall at
-- query time and only requires a single fast build at ingestion. IVFFlat is
-- preserved below as a fallback for older pgvector versions.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename = 'child_nodes'
          AND indexname = 'child_nodes_embedding_hnsw_cos'
    ) THEN
        BEGIN
            EXECUTE 'CREATE INDEX child_nodes_embedding_hnsw_cos
                     ON child_nodes
                     USING hnsw (embedding vector_cosine_ops)
                     WITH (m = 16, ef_construction = 64)';
        EXCEPTION WHEN feature_not_supported THEN
            EXECUTE 'CREATE INDEX child_nodes_embedding_ivfflat_cos
                     ON child_nodes
                     USING ivfflat (embedding vector_cosine_ops)
                     WITH (lists = 200)';
        END;
    END IF;
END
$$;

-- FTS index also on child summaries so stage 1 can rank candidates that match
-- form numbers and keyword tokens even when the underlying vector is weak.
CREATE INDEX IF NOT EXISTS child_nodes_text_fts_idx
    ON child_nodes
    USING GIN (to_tsvector('english', text_summary));

-- -----------------------------------------------------------------------------
-- updated_at trigger for ingested_documents.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION taxbot_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ingested_documents_set_updated_at
    ON ingested_documents;
CREATE TRIGGER ingested_documents_set_updated_at
    BEFORE UPDATE ON ingested_documents
    FOR EACH ROW EXECUTE FUNCTION taxbot_set_updated_at();
