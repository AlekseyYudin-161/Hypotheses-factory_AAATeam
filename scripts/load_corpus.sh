#!/usr/bin/env bash
# Загрузка схемы и корпуса в pgvector.
#
#   1) DDL из db/schema.sql (extension vector, таблицы, индексы);
#   2) данные из corpus_dump.sql (documents + knowledge_chunks с эмбеддингами).
#
# Использование:
#   DB_DSN="postgresql://postgres:postgres@localhost:5432/hypothesis_factory" \
#     bash scripts/load_corpus.sh
#
# Если поднимаете через docker compose — schema.sql применяется автоматически
# (см. docker-compose.yml), и достаточно залить только corpus_dump.sql.

set -euo pipefail
cd "$(dirname "$0")/.."

: "${DB_DSN:?Задайте DB_DSN, напр. postgresql://postgres:postgres@localhost:5432/hypothesis_factory}"

echo "==> Применяю схему db/schema.sql"
psql "$DB_DSN" -v ON_ERROR_STOP=1 -f db/schema.sql

echo "==> Загружаю корпус corpus_dump.sql"
psql "$DB_DSN" -v ON_ERROR_STOP=1 -f corpus_dump.sql

echo "==> Проверка:"
psql "$DB_DSN" -c "SELECT count(*) AS documents FROM documents;"
psql "$DB_DSN" -c "SELECT count(*) AS chunks FROM knowledge_chunks;"
echo "OK"
