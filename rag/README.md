# RAG-ядро «Фабрики гипотез»

Модуль генерирует **гипотезы по доизвлечению цветных металлов (Ni, Cu) и снижению
затрат** для обогатительно-металлургического передела (хвосты, отвалы, шлаки).

- **Вход** — «богатый промпт» жюри: текстовое описание проблемы + бюджет/сроки +
  Markdown-таблица материального баланса, опционально фото/схемы установки.
- **Выход** — топ-3 отранжированные гипотезы, каждая с причинно-следственной
  цепочкой в формате PMDco, ссылками на источники (DOI) и численным скором по трём
  осям (Risk / Value / Novelty). Либо **легальный отказ**, если улучшать нечего.

Идея RAG здесь: не просить LLM «придумать из головы», а извлечь из базы знаний
реальные успешные эксперименты и заставить модель перенести их механизмы по
**аналогии** на наш целевой материал, оставаясь строго внутри бюджета и цитируя
источники.

---

## 1. Карта файлов

| Файл | Роль |
|---|---|
| [pipeline.py](pipeline.py) | **Оркестратор.** Класс `FabrikaPipeline`, все 4 шага, дедлайн, скоринг, промпты LLM. Точка входа — `generate()`. |
| [schemas.py](schemas.py) | **Контракт данных** (Pydantic). Структуры, которыми обмениваются шаги, и `response_format` для structured-output LLM. |
| [database.py](database.py) | **Боевой клиент** pgvector: гибридный поиск (`hybrid_search`) и новизна по графу триплетов (`check_novelty_triplet`) + эмбеддер запроса. |
| [stubs.py](stubs.py) | **Офлайн-заглушки** тех же интерфейсов (`KnowledgeBaseStub`, `ChemistStub`) — чтобы прогонять пайплайн без БД и ключей. |
| [service.py](service.py) | **Адаптер под фронтенд.** Конвертирует узкую `Hypothesis` в «богатый» JSON под `hypothesis_schema.json`; drop-in `generate_hypotheses(...)`. |
| [__init__.py](__init__.py) | Публичный экспорт пакета. |
| [requirements.txt](requirements.txt) | Зависимости ядра (отдельно от streamlit-фронтенда). |

Смежные файлы проекта (вне пакета `rag/`), которые обслуживают модуль:

- [../db/schema.sql](../db/schema.sql) — DDL базы знаний.
- [../corpus_dump.sql](../corpus_dump.sql) — данные корпуса (1 документ, 91 чанк с 1536-мерными эмбеддингами).
- [../scripts/load_corpus.sh](../scripts/load_corpus.sh) — заливка схемы + корпуса.
- [../scripts/rag_smoke.py](../scripts/rag_smoke.py) — end-to-end смоук на реальном корпусе.
- [../docker-compose.yml](../docker-compose.yml) — pgvector (авто-инициализация схемой и корпусом) + app.

---

## 2. Пайплайн: 4 шага

Точка входа — `FabrikaPipeline.generate(rich_input, images=None) -> PipelineResult`.
Вся тяжёлая работа (`_run`) выполняется в отдельном потоке под **жёстким дедлайном
120 секунд**: если LLM «зависла», фронтенд всё равно получит ответ (частичный или
отказ). См. `generate()` в [pipeline.py](pipeline.py).

```
rich_input (+ images)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ ШАГ 1 — Декомпозиция «богатого входа»            _decompose()  │
│  LLM (structured output) → IndustrialConstraints              │
│  • target_material  → BLACK-LIST (что перерабатываем)          │
│  • target_property  → WHITE-LIST (что улучшаем: KPI)           │
│  • budget / time_limit / mass_smt / mechanisms[]              │
│  • картинки → мультимодальный vision-вход                      │
└───────────────────────────────────────────────────────────────┘
        │           ранний отказ, если budget ≤ 0
        ▼
┌───────────────────────────────────────────────────────────────┐
│ ШАГ 2 — Гибридный поиск контекста          _retrieve_context() │
│  для каждого mechanism → db.hybrid_search(white, black, mech)  │
│  вектор (<=>) + полнотекст (tsvector), RRF-фьюжн, дедуп по DOI  │
└───────────────────────────────────────────────────────────────┘
        │           отказ, если контекст пуст / вышло время
        ▼
┌───────────────────────────────────────────────────────────────┐
│ ШАГ 3 — Синтез топ-3 гипотез (ОДИН вызов LLM)     synth_chain  │
│  контекст с метками [DOI: ...] → HypothesisBatch (ровно 3)     │
│  каждая: text + PMDco causal_chain + key_process + DOI + KPI   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ ШАГ 4 — Ранжирование и отсев                         _score()  │
│  Risk (экономика) · Value (LLM) · Novelty (триплеты)           │
│  final = 0.30·Novelty + 0.40·(1−Risk) + 0.30·Value            │
│  отсев < 0.35, сортировка по final, срез top-3                 │
└───────────────────────────────────────────────────────────────┘
        │           отказ, если никто не прошёл порог
        ▼
   PipelineResult(hypotheses=[...])  ИЛИ  refused=True
```

### Шаг 1 — Индустриальная декомпозиция

`_decompose()` вызывает LLM с `DECOMPOSE_PROMPT` (few-shot: примеры по
пирротиновым хвостам и медному шлаку) и просит structured-output по схеме
`IndustrialConstraints`. Ключевая идея — разложить вход на **white/black-list**:

- **`target_material`** («отвальные пирротиновые хвосты») → **BLACK-LIST**: мы НЕ
  хотим искать статьи про сам мусор, иначе ретривал вернёт публикации про хвосты,
  а не про способ их переработки.
- **`target_property`** («доизвлечение никеля») → **WHITE-LIST**: ищем только то,
  что реально помогает достичь цели.
- `budget`, `time_limit`, `mass_smt` — экономические ограничения (если не заданы
  явно — модель обязана вернуть `null`, а не выдумывать).
- `mechanisms[]` — конкретные **сущности** (реагенты, аппараты, физ-хим механизмы),
  а не действия; именно они станут поисковыми запросами шага 2.

**Мультимодальность:** если переданы `images` (пути к файлам / http-URL / data-URI),
few-shot остаётся тем же, но последнее человеческое сообщение собирается вручную как
блок `text + image_url` (`_image_block` кодирует локальные файлы в base64 data-URI)
и уходит в vision-модель.

После шага — ранний легальный отказ, если `budget` задан и `≤ 0`.

### Шаг 2 — Гибридный поиск контекста

`_retrieve_context()` перебирает все `mechanisms` (fallback — сам `target_property`,
если механизмов нет) и для каждого зовёт `db.hybrid_search(...)`. Результаты
**дедуплицируются по DOI** и сливаются в один общий контекст — чтобы синтез уложился
в **один** вызов LLM.

Боевая реализация — `DatabaseClient.hybrid_search` в [database.py](database.py):

1. Строит текст запроса `"{mechanism}. {target_property}"` и эмбеддит его той же
   моделью, что и корпус (`text-embedding-3-small`, 1536 измерений) — иначе вектора
   несопоставимы.
2. Выполняет один SQL (`_HYBRID_SQL`) с двумя под-ранкингами:
   - **векторный** — `ORDER BY embedding <=> %(vec)s` (косинусное расстояние pgvector), top-100;
   - **полнотекстовый** — `ts_rank(text_search, plainto_tsquery('russian', q))`, top-100;
3. Объединяет их через **RRF (Reciprocal Rank Fusion)**:
   `score = Σ 1/(60 + rank_i)`. RRF устойчиво сливает два разных ранкинга без
   калибровки их «сырых» скорингов. Возврат — top-k чанков с DOI и заголовком.

Если контекст пуст → легальный отказ «в базе нет релевантного контекста».

> Замечание по текущей схеме: `blacklist` (target_material) передаётся, но в базе без
> entity-метаданных используется как контекст запроса, а не жёсткий SQL-фильтр.

### Шаг 3 — Синтез топ-3 гипотез (кап генерации)

Контекст форматируется `_format_context()` — каждый чанк префиксуется меткой
`[DOI: ...] (title) текст`, чтобы модель цитировала источники точно. Затем **один**
вызов `synth_chain` (промпт `SYNTH_PROMPT` + structured-output `HypothesisBatch`)
возвращает **ровно 3** гипотезы. Это жёсткий «кап генерации» — экономим round-trip'ы
под 120-секундный бюджет.

Требования, зашитые в промпт/схему для каждой гипотезы:
- опираться **только** на предоставленный контекст; DOI брать строго из меток `[DOI: ...]`;
- строго соблюдать бюджет и срок;
- `causal_chain` — цепочка звеньев PMDco `Сырьё(Entity) → Процесс(Activity) → Продукт(Entity)`;
- `key_process` — один ключевой процесс/аппарат (нужен экономисту для оценки риска);
- `expected_kpi_gain` — честная оценка прироста KPI, `0..1` (это будущий Value Score).

DOI пост-обрабатываются `_clean_doi()`: модель иногда копирует всю метку
`«DOI: file:...»`, оставляем только сам идентификатор.

### Шаг 4 — Ранжирование Risk / Value / Novelty и отсев

`_score()` считает три оси для каждого черновика:

| Ось | Источник | Смысл |
|---|---|---|
| **Risk** | `ChemistStub.risk_score(key_process, mass_smt, budget)` | Экономика первична: стоимость реагентов на всю массу vs бюджет. Выше — хуже. |
| **Value** | `draft.expected_kpi_gain` (оценка LLM) | Теоретический прирост KPI. Выше — лучше. |
| **Novelty** | `db.check_novelty_triplet(target_material, key_process)` | Встречалась ли связка «материал → процесс» в графе триплетов. Выше — новее. |

Агрегация (риск инвертируется — меньше риск даёт больший вклад):

```
final = W_NOVELTY·novelty + W_RISK·(1 − risk) + W_VALUE·value
      =    0.30·novelty  +   0.40·(1 − risk) +   0.30·value
```

Экономика (риск) весит больше всего — 0.40. Дальше: отсев всех гипотез с
`final < 0.35` (`SCORE_THRESHOLD`), сортировка по убыванию, срез `top-3`. Если после
отсева не осталось никого — легальный отказ «улучшения нерентабельны».

**Экономика риска (заглушка `ChemistStub`):** по `key_process` берётся удельная
стоимость реагентов (у.е./т), считается `total_cost = cost_per_t · mass_smt`,
`ratio = total_cost / budget`. При `ratio ≤ 0.5` — дёшево (риск ~0), при `ratio ≥ 1.5`
— выходим за бюджет (риск ~1). Нет массы/бюджета → умеренный риск 0.5.

---

## 3. Легальный отказ (`refused=True`)

Пайплайн возвращает пустой список гипотез и сообщение
`«Текущее решение оптимально, улучшения нерентабельны.»` в четырёх случаях:

1. **Нулевой/отрицательный бюджет** — заявлено «нет средств» (после шага 1).
2. **Пустой контекст** — в базе знаний нет ничего релевантного (после шага 2).
3. **Таймаут** — превышен лимит 120 с (в любой точке; либо весь `generate` не успел).
4. **Все ниже порога** — ни одна гипотеза не набрала `final ≥ 0.35` (после шага 4).

Это не ошибка, а часть контракта: система честно говорит, что улучшать нечего.

---

## 4. Контракт данных ([schemas.py](schemas.py))

Все структуры — Pydantic-модели. Их `Field(description=...)` служат одновременно
валидацией и **инструкцией для LLM** (через `with_structured_output`).

- **`IndustrialConstraints`** — результат шага 1 (white/black-list + ограничения + механизмы).
- **`CausalLink`** — одно звено PMDco (`entity_in → activity → entity_out`), с методом `as_arrow()` для UI/логов.
- **`HypothesisDraft`** — сырая гипотеза от LLM (шаг 3): `hypothesis_text`, `causal_chain[]`, `key_process`, `doi_sources[]`, `expected_kpi_gain`.
- **`HypothesisBatch`** — контейнер «ровно 3 гипотезы» (кап генерации).
- **`Hypothesis`** — `HypothesisDraft` + посчитанные `risk_score` / `value_score` / `novelty_score` / `final_score` (выход шага 4).
- **`PipelineResult`** — финал: `hypotheses[]` **или** `refused=True` + `message`.

---

## 5. База знаний ([../db/schema.sql](../db/schema.sql))

Расширение `vector` (pgvector). Основные таблицы:

- **`documents`** (`doi` PK, `title`, `pub_year`, `source`, `ner_tag`) — источники.
- **`knowledge_chunks`** (`chunk_id` PK, `doi`, `chunk_text`, `embedding vector(1536)`,
  `text_search tsvector`) — чанки корпуса. Индексы: **GIN** на `text_search`
  (полнотекст) и **ivfflat** `vector_cosine_ops, lists=100` (ANN-поиск по вектору).
- **`triplets`** (`material_canonical`, `effect_property`, `relation_raw`, ...) — граф
  знаний для оценки новизны (шаг 4).
- **`material_aliases`** (`alias`, `material_canonical`, `composition`) — нормализация названий материалов.

**Корпус** ([../corpus_dump.sql](../corpus_dump.sql)): материальный баланс хвостов КГМК
(`file:Хвосты_КГМК.xlsx`), **91 чанк** с **1536-мерными** эмбеддингами
(`text-embedding-3-small`). Размерность запроса ОБЯЗАНА совпадать с размерностью
корпуса, иначе вектора несопоставимы.

---

## 6. Интеграция с UI ([service.py](service.py))

Ядро отдаёт узкую `Hypothesis`, а фронтенд ждёт «богатую» гипотезу
(`novelty/risk/value/economic_value/success_probability`, `evidence[]` с `source_url`,
`roadmap`, `causal_chain` строками и т.д. — под `hypothesis_schema.json`).
`service.py` делает эту конвертацию и даёт drop-in функцию с той же сигнатурой, что
`services.generator.generate_hypotheses`:

```python
# было:   from services.generator import generate_hypotheses
from rag.service import generate_hypotheses   # когда заданы DB_DSN + PROXYAPI_*
```

Детали адаптера:
- `economic_value = 1 − risk_score` (экономика «зашита» в риск, инвертируем);
- `_source_url_for(doi)` — настоящие DOI/URL пробрасывает как есть, у внутренних
  корпусных файлов (`file:...`) нет URL → отдаёт стабильную ссылку на материалы кейса (`SOURCE_ROOT_URL`);
- `final_score` пересчитывается формулой фронтенда `calculate_final_score(...)` в шкале `0..100`;
- при легальном отказе список пуст — UI показывает `REFUSAL_MESSAGE`.

---

## 7. Подмена компонентов (dependency injection)

Всё внедряется через конструктор — ядро не трогаем, меняем только объекты с тем же интерфейсом (duck typing):

```python
FabrikaPipeline(
    llm=...,             # любой ChatOpenAI-совместимый (ProxyAPI / локальная GPU)
    knowledge_base=...,  # DatabaseClient (боевой pgvector) или KnowledgeBaseStub (офлайн)
    chemist=...,         # реальный модуль экономиста или ChemistStub
    deadline_seconds=120,
)
```

- `FabrikaPipeline()` без аргументов — дефолтная LLM (`gpt-4.1-mini` через ProxyAPI) + заглушки.
- `FabrikaPipeline.from_env()` — **боевая сборка**: реальная LLM + `DatabaseClient` по `DB_DSN`.

---

## 8. Запуск

**Офлайн-демо** без БД и ключей (фейковая LLM + заглушки, детерминированный вывод):

```bash
python -m rag.pipeline
```

**Боевой прогон** на реальном корпусе:

```bash
# 1. поднять pgvector (schema.sql и corpus_dump.sql применяются авто-init'ом)
docker compose up -d db

# 2. (если не через compose) залить схему + корпус вручную
DB_DSN="postgresql://postgres:postgres@localhost:5432/hypothesis_factory" \
  bash scripts/load_corpus.sh

# 3. заполнить .env: PROXYAPI_API_KEY, PROXYAPI_BASE_URL, DB_DSN, EMB_MODEL, EMB_DIM
python -m scripts.rag_smoke        # end-to-end смоук на кейсе КГМК
```

---

## 9. Переменные окружения

| Переменная | Назначение |
|---|---|
| `PROXYAPI_API_KEY` | Ключ ProxyAPI (LLM и эмбеддинги). Фолбэк — `OPENAI_API_KEY`. |
| `PROXYAPI_BASE_URL` | Base URL прокси для OpenAI-совместимого API. |
| `DB_DSN` | Строка подключения к pgvector (обязательна для боевого режима). |
| `LLM_MODEL` | Модель генерации, дефолт `gpt-4.1-mini`. |
| `EMB_MODEL` | Модель эмбеддингов запроса, дефолт `text-embedding-3-small` (**должна совпадать с моделью индексации корпуса**). |
| `EMB_DIM` | Размерность эмбеддингов, `1536` (совпадает со схемой). |

---

## Жёсткие ограничения (CRITICAL)

- Весь `generate()` ≤ **120 с** (hard deadline через `ThreadPoolExecutor` + `timeout`).
- Генерация топ-3 гипотез — **один** вызов LLM (кап генерации).
- **Легальный отказ**, если всё нерентабельно/невозможно — часть контракта, не баг.
- Модель эмбеддинга запроса = модель индексации корпуса (иначе поиск сломан).
