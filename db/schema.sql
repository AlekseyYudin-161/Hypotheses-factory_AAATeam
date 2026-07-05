-- Схема из triplet_contract_v1. vector(256) — ПОДТВЕРДИ размерность смоук-тестом.
-- Имя БД здесь НЕ указывается: скрипт выполняется в БД из POSTGRES_DB (hypothesis_factory).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  doi TEXT PRIMARY KEY,
  title TEXT, pub_year INT, source TEXT, ner_tag TEXT
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
  chunk_id    BIGSERIAL PRIMARY KEY,
  doi         TEXT,
  chunk_text  TEXT,
  embedding   vector(1536),
  text_search tsvector
);
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON knowledge_chunks USING GIN(text_search);
CREATE INDEX IF NOT EXISTS idx_chunks_emb ON knowledge_chunks
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS triplets (
  triplet_id         BIGSERIAL PRIMARY KEY,
  doi                TEXT,
  chunk_id           BIGINT,
  material_raw       TEXT,
  material_canonical TEXT,
  condition          TEXT,
  effect_property    TEXT,
  relation           TEXT,   -- контролируемый словарь (enum на уровне приложения)
  relation_raw       TEXT,
  value              DOUBLE PRECISION,
  unit               TEXT,
  direction          TEXT
);

CREATE TABLE IF NOT EXISTS material_aliases (
  alias              TEXT,
  material_canonical TEXT,
  composition        TEXT
);
