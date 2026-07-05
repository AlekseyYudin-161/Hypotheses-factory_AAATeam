# RAG-ядро «Фабрики гипотез»

Пайплайн генерации гипотез по доизвлечению металлов и снижению затрат для
обогатительно-металлургического передела. Вход — «богатый промпт» жюри
(текст + бюджет/сроки + таблица материального баланса, опционально фото/схемы),
выход — топ-3 отранжированные гипотезы с PMDco-цепочкой, источниками и скором.

## Архитектура (4 шага)

| Шаг | Файл | Что делает |
|---|---|---|
| 1. Декомпозиция | [pipeline.py](pipeline.py) `_decompose` | LLM (structured output) → `IndustrialConstraints`: target_material (black-list), target_property (white-list), budget, time_limit, mass_smt, mechanisms. Поддерживает фото/схемы (vision). |
| 2. Гибридный поиск | [database.py](database.py) `hybrid_search` | pgvector: вектор (`<=>`) + полнотекст (`tsvector`) с RRF-ранжированием по `knowledge_chunks`. |
| 3. Синтез | [pipeline.py](pipeline.py) `synth_chain` | **Один** вызов LLM → сразу 3 гипотезы с `causal_chain` (Entity→Activity→Entity) и DOI из контекста. |
| 4. Ранжирование | [pipeline.py](pipeline.py) `_score` | Risk (экономика, [stubs.py](stubs.py) `ChemistStub`) · Value (LLM) · Novelty (триплеты). Агрегат, сортировка, отсев ниже порога. |

Жёсткие ограничения: весь `generate()` ≤ 120 с (hard deadline через поток),
кап генерации = 3 гипотезы за вызов, легальный отказ «Текущее решение оптимально,
улучшения нерентабельны» при пустом контексте / нулевом бюджете / всех гипотезах
ниже порога.

## Данные

- `db/schema.sql` — DDL (extension `vector`, таблицы, ivfflat + GIN индексы).
- `corpus_dump.sql` — данные: материальный баланс хвостов КГМК, 91 чанк,
  эмбеддинги **1536-мерные** (`text-embedding-3-small`). Element 28 = Ni, 29 = Cu.

## Запуск

Офлайн-демо без БД и ключей (заглушки):
```bash
python -m rag.pipeline
```

Боевой прогон на реальном корпусе:
```bash
docker compose up -d db                       # поднять pgvector (schema.sql применится сам)
DB_DSN=postgresql://postgres:postgres@localhost:5432/hypothesis_factory \
  psql "$DB_DSN" -f corpus_dump.sql           # залить данные
# .env: PROXYAPI_API_KEY, PROXYAPI_BASE_URL, DB_DSN, EMB_MODEL
python -m scripts.rag_smoke                    # end-to-end смоук
```
(или разом: `bash scripts/load_corpus.sh` для схемы + корпуса).

## Интеграция с UI

[service.py](service.py) даёт drop-in `generate_hypotheses(kpi, constraints, language,
knowledge_bases, images)` с той же сигнатурой, что `services.generator`, но на реальном
пайплайне (возвращает «богатые» гипотезы под `hypothesis_schema.json`). Подключение —
одной строкой в `app.py`:

```python
# было:   from services.generator import generate_hypotheses
from rag.service import generate_hypotheses   # когда заданы DB_DSN + PROXYAPI_*
```
При легальном отказе список пуст — UI показывает `REFUSAL_MESSAGE`.

## Подмена компонентов

Всё внедряется через конструктор `FabrikaPipeline(llm=, knowledge_base=, chemist=)`:
- `knowledge_base` — `DatabaseClient` (боевой) или `KnowledgeBaseStub` (офлайн);
- `chemist` — реальный модуль экономиста вместо `ChemistStub`;
- `llm` — любой ChatOpenAI-совместимый (ProxyAPI / локальная GPU).
`FabrikaPipeline.from_env()` собирает боевую конфигурацию из переменных окружения.
