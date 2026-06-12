-- =============================================================================
-- TaxBot baseline migration
--
-- Provisions the parent-child retrieval schema used by the RAG pipeline:
--   * Creates `parent_nodes` (full markdown sections / tables)
--   * Creates `child_nodes` (sentence-level summaries; embeddings live in Qdrant)
--   * Creates `ingested_documents` (state table for the metadata-or-hash delta)
--
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;     -- gen_random_uuid()

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

-- -----------------------------------------------------------------------------
-- Child nodes: small semantic units (sentences, table abstracts).
-- Embeddings and BM25 sparse vectors are stored in Qdrant, not here.
-- The `text_summary` column is kept so ingestion can re-encode on backfill
-- and for diagnostic queries.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS child_nodes (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_id    UUID NOT NULL
        REFERENCES parent_nodes (id) ON DELETE CASCADE,
    text_summary TEXT        NOT NULL,
    metadata     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS child_nodes_parent_id_idx
    ON child_nodes (parent_id);

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
