"""
Реальный клиент базы знаний (pgvector) — ШАГ 2 (поиск) и ШАГ 4 (новизна).

Схема соответствует db/schema.sql и дампу corpus_dump.sql:
  documents(doi PK, title, pub_year, source, ner_tag)
  knowledge_chunks(chunk_id PK, doi, chunk_text, embedding vector(1536), text_search tsvector)
  triplets(... material_canonical, effect_property, relation_raw ...)   -- для новизны
  material_aliases(alias, material_canonical, composition)

Эмбеддинги корпуса — 1536-мерные (OpenAI text-embedding-3-small через ProxyAPI).
Запрос ОБЯЗАН эмбеддиться ТЕМ ЖЕ моделью, иначе вектора несопоставимы.

Класс реализует тот же интерфейс, что KnowledgeBaseStub (hybrid_search /
check_novelty_triplet), поэтому подставляется в FabrikaPipeline без правок ядра.
"""

from __future__ import annotations

import os
from typing import List, Optional

import psycopg2
import psycopg2.extras

from .stubs import Chunk


# --- SQL: гибридный поиск (вектор + полнотекст) с RRF-ранжированием -----------
# RRF (Reciprocal Rank Fusion): score = Σ 1/(k + rank_i), k=60 — устойчиво
# объединяет два разных ранкинга без калибровки их «сырых» скорингов.
_HYBRID_SQL = """
WITH
vector_search AS (
    SELECT chunk_id, vr FROM (
        SELECT chunk_id,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %(vec)s::vector) AS vr
        FROM knowledge_chunks
    ) v WHERE vr <= 100
),
text_search AS (
    SELECT chunk_id, tr FROM (
        SELECT chunk_id,
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank(text_search, plainto_tsquery('russian', %(q)s)) DESC
               ) AS tr
        FROM knowledge_chunks
        WHERE text_search @@ plainto_tsquery('russian', %(q)s)
    ) t WHERE tr <= 100
)
SELECT c.chunk_id,
       c.chunk_text,
       c.doi,
       COALESCE(d.title, c.doi)        AS title,
       COALESCE(d.source, 'internal')  AS source,
       COALESCE(1.0 / (60 + v.vr), 0.0) + COALESCE(1.0 / (60 + t.tr), 0.0) AS rrf_score
FROM knowledge_chunks c
LEFT JOIN documents d      ON c.doi = d.doi
LEFT JOIN vector_search v  ON c.chunk_id = v.chunk_id
LEFT JOIN text_search t    ON c.chunk_id = t.chunk_id
WHERE v.chunk_id IS NOT NULL OR t.chunk_id IS NOT NULL
ORDER BY rrf_score DESC
LIMIT %(k)s;
"""

# Новизна: встречалась ли связка «материал → эффект/отношение» в графе триплетов.
_NOVELTY_SQL = """
SELECT count(*) FROM triplets
WHERE (material_canonical ILIKE %(mat)s OR material_raw ILIKE %(mat)s)
  AND (effect_property   ILIKE %(prop)s OR relation_raw ILIKE %(prop)s);
"""


class Embeddings:
    """Обёртка над OpenAI-совместимым эмбеддером (ProxyAPI)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI  # локальный импорт: не тянем в офлайн-тестах

        self.model = model or os.getenv("EMB_MODEL", "text-embedding-3-small")
        self.client = OpenAI(
            api_key=api_key or os.getenv("PROXYAPI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("PROXYAPI_BASE_URL") or None,
        )

    def encode(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding


class DatabaseClient:
    """
    Боевой клиент pgvector. Одно короткоживущее соединение на запрос — под
    2-минутный дедлайн этого более чем достаточно и не держим пул зря.
    """

    def __init__(self, dsn: Optional[str] = None, embeddings: Optional[Embeddings] = None):
        self.dsn = dsn or os.getenv("DB_DSN")
        if not self.dsn:
            raise ValueError("DB_DSN не задан: укажите строку подключения к pgvector.")
        self.embeddings = embeddings or Embeddings()

    # ------------------------------------------------------------------ #
    # ШАГ 2: гибридный поиск
    # ------------------------------------------------------------------ #
    def hybrid_search(
        self,
        whitelist: str,
        blacklist: str,
        mechanism: str,
        top_k: int = 5,
    ) -> List[Chunk]:
        """
        Возвращает top_k чанков, релевантных механизму и целевому свойству.

        whitelist — target_property (усиливает текстовый запрос);
        blacklist — target_material (в текущей схеме без entity-метаданных
                    используется как контекст запроса, а не жёсткий фильтр);
        mechanism — конкретная сущность для векторного поиска.
        """
        # Вектор строим по механизму + целевому свойству — так эмбеддинг ближе
        # к нужному переделу, а не к абстрактному термину.
        query_text = f"{mechanism}. {whitelist}".strip(". ")
        vec = self.embeddings.encode(query_text)
        vec_literal = "[" + ",".join(f"{x:.7f}" for x in vec) + "]"

        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    _HYBRID_SQL,
                    {"vec": vec_literal, "q": query_text, "k": top_k},
                )
                rows = cur.fetchall()

        return [
            Chunk(
                doi=r["doi"],
                chunk_text=r["chunk_text"],
                title=r["title"],
                rrf_score=float(r["rrf_score"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------ #
    # ШАГ 4: новизна по графу триплетов
    # ------------------------------------------------------------------ #
    def check_novelty_triplet(self, subject: str, obj: str) -> float:
        """
        1.0 — связка «материал → эффект» в базе не встречалась (новизна),
        0.2 — уже описана. Пустая таблица triplets ⇒ максимальная новизна.
        """
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(_NOVELTY_SQL, {"mat": f"%{subject}%", "prop": f"%{obj}%"})
                count = cur.fetchone()[0]
        return 1.0 if count == 0 else 0.2
