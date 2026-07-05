"""
Ядро «Фабрики гипотез» — оркестрация RAG-пайплайна.

Зона ответственности этого файла (моя):
  ШАГ 1  Индустриальная декомпозиция «богатого входа»  (LLM, structured output)
  ШАГ 2  Гибридный поиск контекста                     (заглушка БД, stubs.py)
  ШАГ 3  Синтез сразу ТОП-3 гипотез за один вызов LLM   (LLM, structured output)
  ШАГ 4  Ранжирование Risk / Value / Novelty + отсев    (scorer + заглушки)

Жёсткие ограничения (CRITICAL):
  * весь цикл generate() укладывается в 120 секунд (иначе — отдаём что успели);
  * генерация топ-3 гипотез — ОДИН вызов LLM (кап генерации);
  * легальный отказ, если всё нерентабельно.

LLM: ChatOpenAI(model="gpt-4o-mini"). Модель можно подменить (для тестов/локальной
GPU) — конструктор принимает готовый объект `llm`.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from .schemas import (
    Hypothesis,
    HypothesisBatch,
    HypothesisDraft,
    IndustrialConstraints,
    PipelineResult,
)
from .stubs import ChemistStub, KnowledgeBaseStub

# --- Настройки оркестрации ---------------------------------------------------
DEADLINE_SECONDS = 120           # жёсткий таймаут всего цикла (2 минуты)
SCORE_THRESHOLD = 0.35           # ниже этого балла гипотезы отбрасываем
TOP_N = 3                        # кап генерации: ровно 3 гипотезы за вызов

# Веса агрегации трёх осей (в сумме 1.0). Экономика (риск) первична.
W_NOVELTY = 0.30
W_RISK = 0.40
W_VALUE = 0.30

REFUSAL_MESSAGE = "Текущее решение оптимально, улучшения нерентабельны."


# ---------------------------------------------------------------------------
# Системные промпты
# ---------------------------------------------------------------------------
# ШАГ 1: разбор богатого входа в жёсткую структуру IndustrialConstraints.
DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — ведущий технолог обогатительной фабрики и металлург. "
            "Тебе дают «богатый вход» жюри: описание проблемы передела + бюджет/сроки + "
            "Markdown-таблицу материального баланса (отходы/хвосты, их масса в СМТ, потери металла). "
            "Извлеки из него строгую структуру ограничений.\n"
            "ПРАВИЛА:\n"
            "  * target_material — тот отход/материал, который перерабатываем (пойдёт в BLACK-LIST "
            "поиска: НЕ ищем статьи про сам мусор);\n"
            "  * target_property — что улучшаем: доизвлечение конкретного металла или снижение затрат "
            "(пойдёт в WHITE-LIST поиска);\n"
            "  * mass_smt бери из таблицы материального баланса (сухие метрические тонны);\n"
            "  * mechanisms — конкретные СУЩНОСТИ (реагенты, аппараты, физ-хим механизмы), а не действия;\n"
            "  * если бюджет/срок/масса не заданы явно — оставь null, не выдумывай.",
        ),
        (
            "user",
            "Проблема: КГМК накапливает отвальные пирротиновые хвосты с потерями никеля. "
            "Бюджет 5 млн у.е., срок 6 месяцев.\n"
            "| Поток | Масса, СМТ | Потери Ni, % |\n"
            "|---|---|---|\n"
            "| Пирротиновые хвосты | 120000 | 0.6 |",
        ),
        (
            "assistant",
            '{{"target_material": "отвальные пирротиновые хвосты", '
            '"target_property": "доизвлечение никеля", '
            '"budget": 5000000, "time_limit": "6 месяцев", "mass_smt": 120000, '
            '"mechanisms": ["автоклавное выщелачивание", "магнитная сепарация", "флотация", "биовыщелачивание"]}}',
        ),
        # Металлургия: шлаки/отвалы (в реальном промпте — плюс фото/схемы, они уходят
        # отдельным image-блоком в том же человеческом сообщении, см. _decompose).
        (
            "user",
            "На фото и схеме — отвальный шлак медеплавильного производства, теряем медь. "
            "Бюджет не указан, срок 4 месяца.\n"
            "| Продукт | Масса, СМТ | Cu, % |\n|---|---|---|\n| Отвальный шлак | 80000 | 0.9 |",
        ),
        (
            "assistant",
            '{{"target_material": "отвальный медеплавильный шлак", '
            '"target_property": "доизвлечение меди", '
            '"budget": null, "time_limit": "4 месяца", "mass_smt": 80000, '
            '"mechanisms": ["флотация шлака", "обеднение шлака в электропечи", "медленное охлаждение шлака", "измельчение"]}}',
        ),
        ("user", "{rich_input}"),
    ]
)

# ШАГ 3: синтез сразу ТОП-3 гипотез. Аналогический перенос механизмов из контекста
# на наш целевой материал, в рамках бюджета, с обязательной PMDco-цепочкой и DOI.
SYNTH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты — главный технолог-исследователь по обогащению руд и металлургии: переработка "
            "хвостов, отвалов и шлаков, доизвлечение цветных металлов (Ni, Cu), снижение затрат. "
            "Сгенерируй РОВНО 3 гипотезы за один ответ (это жёсткий кап).\n"
            "ТРЕБОВАНИЯ к каждой гипотезе:\n"
            "  * опирайся ТОЛЬКО на предоставленный контекст; DOI бери строго из меток [DOI: ...];\n"
            "  * строго соблюдай бюджет и срок — не предлагай того, что заведомо в них не влезает;\n"
            "  * причинно-следственная цепочка обязана быть в формате PMDco: список звеньев "
            "Сырьё(Entity) -> Процесс(Activity) -> Продукт(Entity);\n"
            "  * key_process — один ключевой процесс/аппарат гипотезы (для экономической оценки);\n"
            "  * expected_kpi_gain — честная оценка теоретического прироста KPI по контексту (0..1).",
        ),
        (
            "user",
            "Целевой материал (что перерабатываем): {target_material}\n"
            "Целевое улучшение (KPI): {target_property}\n"
            "Бюджет: {budget}; Срок: {time_limit}; Масса отходов (СМТ): {mass_smt}\n\n"
            "КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ (успешные эксперименты, с источниками):\n{context_str}",
        ),
    ]
)


class FabrikaPipeline:
    """
    Оркестратор пайплайна. Точка входа — `generate(rich_input) -> PipelineResult`.

    Внешние зависимости внедряются через конструктор (dependency injection),
    чтобы пайплайн можно было тестировать офлайн и подменять заглушки реальными
    модулями команды без правок ядра.
    """

    def __init__(
        self,
        llm=None,
        knowledge_base: Optional[KnowledgeBaseStub] = None,
        chemist: Optional[ChemistStub] = None,
        deadline_seconds: int = DEADLINE_SECONDS,
    ):
        self.llm = llm or self._default_llm()
        self.db = knowledge_base or KnowledgeBaseStub()
        self.chemist = chemist or ChemistStub()
        self.deadline_seconds = deadline_seconds

        # Две отдельные structured-цепочки: своя схема на каждый шаг.
        self.decompose_chain = DECOMPOSE_PROMPT | self.llm.with_structured_output(
            IndustrialConstraints
        )
        self.synth_chain = SYNTH_PROMPT | self.llm.with_structured_output(HypothesisBatch)

    # ------------------------------------------------------------------ #
    # LLM по умолчанию
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_llm():
        """ChatOpenAI(gpt-4o-mini) через ProxyAPI. request-таймаут ниже дедлайна цикла."""
        from langchain_openai import ChatOpenAI  # локальный импорт: не тянем зависимость в тестах

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            timeout=45,  # per-call request timeout, с запасом под 120с бюджет
            api_key=os.getenv("PROXYAPI_API_KEY") or os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=os.getenv("PROXYAPI_BASE_URL") or None,
        )

    @classmethod
    def from_env(cls) -> "FabrikaPipeline":
        """
        Боевая сборка: реальная LLM (ProxyAPI) + pgvector по DB_DSN.

        Требует переменные окружения: DB_DSN, PROXYAPI_API_KEY, PROXYAPI_BASE_URL
        (и EMB_MODEL при необходимости). Используется бэкендом/адаптером в проде.
        """
        from .database import DatabaseClient

        return cls(knowledge_base=DatabaseClient())

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #
    def generate(self, rich_input: str, images: Optional[List[str]] = None) -> PipelineResult:
        """
        Полный цикл с жёстким 2-минутным дедлайном.

        rich_input — текст «богатого промпта» (описание + бюджет/сроки + Markdown-таблица).
        images     — опциональные фото/схемы из промпта жюри (пути к файлам, http(s)-URL
                     или data:-URI). Уходят в мультимодальный вход шага 1 (gpt-4o-mini vision).

        Тяжёлую работу выполняем в отдельном потоке и ждём с таймаутом: даже если
        LLM «зависла», фронтенд гарантированно получит ответ (частичный/отказ).
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._run, rich_input, images)
            try:
                return future.result(timeout=self.deadline_seconds)
            except FuturesTimeout:
                # Поток продолжит фоновую работу, но контракт по времени соблюдён.
                return PipelineResult(
                    refused=True,
                    message="Превышен лимит времени (120 c). " + REFUSAL_MESSAGE,
                )

    # ------------------------------------------------------------------ #
    # Основная последовательность шагов
    # ------------------------------------------------------------------ #
    def _run(self, rich_input: str, images: Optional[List[str]] = None) -> PipelineResult:
        start = time.monotonic()

        def out_of_time() -> bool:
            # Оставляем ~2с на сборку/сериализацию ответа.
            return (time.monotonic() - start) > (self.deadline_seconds - 2)

        # --- ШАГ 1: Индустриальная декомпозиция (текст + опциональные фото/схемы) ---
        constraints: IndustrialConstraints = self._decompose(rich_input, images)

        # Ранний легальный отказ: нет бюджета вообще и заявлено «нет средств».
        if constraints.budget is not None and constraints.budget <= 0:
            return PipelineResult(refused=True, message=REFUSAL_MESSAGE)

        # --- ШАГ 2: Гибридный поиск контекста ---
        # Один общий контекст по всем механизмам (дедуп по DOI), чтобы уложиться
        # в один вызов синтеза (кап генерации).
        context = self._retrieve_context(constraints)
        if not context:
            return PipelineResult(
                refused=True,
                message="В базе знаний нет релевантного контекста. " + REFUSAL_MESSAGE,
            )
        if out_of_time():
            return PipelineResult(refused=True, message="Таймаут на этапе поиска. " + REFUSAL_MESSAGE)

        # --- ШАГ 3: Синтез топ-3 гипотез (ОДИН вызов LLM) ---
        context_str = self._format_context(context)
        batch: HypothesisBatch = self.synth_chain.invoke(
            {
                "target_material": constraints.target_material,
                "target_property": constraints.target_property,
                "budget": constraints.budget,
                "time_limit": constraints.time_limit,
                "mass_smt": constraints.mass_smt,
                "context_str": context_str,
            }
        )

        # Нормализуем DOI: модель иногда копирует всю метку «DOI: file:...»
        # вместе с префиксом — оставляем только сам идентификатор.
        for draft in batch.hypotheses:
            draft.doi_sources = [self._clean_doi(s) for s in draft.doi_sources]

        # --- ШАГ 4: Ранжирование и отсев ---
        scored = [self._score(d, constraints) for d in batch.hypotheses[:TOP_N]]
        survivors = [h for h in scored if h.final_score >= SCORE_THRESHOLD]
        survivors.sort(key=lambda h: h.final_score, reverse=True)

        if not survivors:
            # Все гипотезы ниже порога → экономически невыгодно.
            return PipelineResult(refused=True, message=REFUSAL_MESSAGE)

        return PipelineResult(hypotheses=survivors[:TOP_N])

    # ------------------------------------------------------------------ #
    # ШАГ 1: декомпозиция (мультимодальная)
    # ------------------------------------------------------------------ #
    def _decompose(
        self, rich_input: str, images: Optional[List[str]]
    ) -> IndustrialConstraints:
        """
        Разбор богатого входа. Без картинок — обычная текстовая цепочка.
        С картинками — тот же промпт + few-shot, но финальное человеческое сообщение
        собираем вручную как мультимодальный блок (text + image_url).
        """
        if not images:
            return self.decompose_chain.invoke({"rich_input": rich_input})

        # Берём отформатированные system + few-shot сообщения, а последний
        # человеческий ход ({rich_input}) заменяем на мультимодальный.
        messages = DECOMPOSE_PROMPT.format_messages(rich_input=rich_input)[:-1]
        content: list[dict] = [{"type": "text", "text": rich_input}]
        content += [self._image_block(img) for img in images]
        messages.append(HumanMessage(content=content))

        structured = self.llm.with_structured_output(IndustrialConstraints)
        return structured.invoke(messages)

    @staticmethod
    def _image_block(img: str) -> dict:
        """Приводим изображение к формату image_url для OpenAI vision.

        Принимает http(s)-URL, готовый data:-URI или путь к локальному файлу
        (последний кодируем в base64 data-URI).
        """
        if img.startswith(("http://", "https://", "data:")):
            url = img
        else:
            mime = mimetypes.guess_type(img)[0] or "image/png"
            with open(img, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
            url = f"data:{mime};base64,{b64}"
        return {"type": "image_url", "image_url": {"url": url}}

    # ------------------------------------------------------------------ #
    # ШАГ 2: ретривал
    # ------------------------------------------------------------------ #
    def _retrieve_context(self, c: IndustrialConstraints) -> List[dict]:
        """Собираем контекст по всем механизмам, дедуплицируем чанки по DOI."""
        seen_doi: set[str] = set()
        merged: List[dict] = []
        mechanisms = c.mechanisms or [c.target_property]  # fallback, если механизмов нет
        for mech in mechanisms:
            for chunk in self.db.hybrid_search(
                whitelist=c.target_property,
                blacklist=c.target_material,
                mechanism=mech,
            ):
                if chunk["doi"] not in seen_doi:
                    seen_doi.add(chunk["doi"])
                    merged.append(chunk)
        return merged

    @staticmethod
    def _clean_doi(raw: str) -> str:
        """Убираем скобки и префикс 'DOI:' из скопированной моделью метки источника."""
        s = raw.strip().lstrip("[").rstrip("]").strip()
        for prefix in ("DOI:", "doi:"):
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
        return s

    @staticmethod
    def _format_context(context: List[dict]) -> str:
        """Префиксуем каждый чанк меткой [DOI: ...] — так LLM цитирует источники точно."""
        return "\n\n".join(
            f"[DOI: {ch['doi']}] ({ch.get('title', '')}) {ch['chunk_text']}" for ch in context
        )

    # ------------------------------------------------------------------ #
    # ШАГ 4: скоринг
    # ------------------------------------------------------------------ #
    def _score(self, draft: HypothesisDraft, c: IndustrialConstraints) -> Hypothesis:
        """
        Три оси:
          Risk    — экономика первична: стоимость реагентов на массу vs бюджет (заглушка химика);
          Value   — теоретический прирост KPI (оценка LLM из контекста);
          Novelty — новизна связки материал↔процесс по графу триплетов (заглушка БД).
        """
        risk = self.chemist.risk_score(draft.key_process, c.mass_smt, c.budget)
        value = float(draft.expected_kpi_gain)
        novelty = self.db.check_novelty_triplet(c.target_material, draft.key_process)

        # Агрегация: риск инвертируем (меньше риск — больше вклад).
        final = W_NOVELTY * novelty + W_RISK * (1.0 - risk) + W_VALUE * value

        return Hypothesis(
            **draft.model_dump(),
            risk_score=round(risk, 3),
            value_score=round(value, 3),
            novelty_score=round(novelty, 3),
            final_score=round(final, 3),
        )


# ---------------------------------------------------------------------------
# Демо-прогон офлайн (без ключа OpenAI): подставляем фейковую LLM.
#   python -m rag.pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    from langchain_core.runnables import RunnableLambda

    # --- Фейковая LLM: возвращает валидные объекты нужной схемы без сети. ---
    class _FakeStructured:
        def __init__(self, kind):
            self.kind = kind

        def invoke(self, _inputs, *args, **kwargs):
            if self.kind is IndustrialConstraints:
                return IndustrialConstraints(
                    target_material="отвальные пирротиновые хвосты",
                    target_property="доизвлечение никеля",
                    budget=5_000_000,
                    time_limit="6 месяцев",
                    mass_smt=120_000,
                    mechanisms=["автоклавное выщелачивание", "магнитная сепарация", "флотация"],
                )
            return HypothesisBatch(
                hypotheses=[
                    HypothesisDraft(
                        hypothesis_text="Автоклавное окислительное выщелачивание пирротиновых "
                        "хвостов переводит никель в раствор с извлечением >90%.",
                        causal_chain=[
                            {"entity_in": "пирротиновые хвосты", "activity": "автоклавное выщелачивание", "entity_out": "никелевый раствор"},
                            {"entity_in": "никелевый раствор", "activity": "экстракция", "entity_out": "катодный никель"},
                        ],
                        key_process="автоклавное выщелачивание",
                        doi_sources=["10.3390/ma15196536"],
                        expected_kpi_gain=0.85,
                    ),
                    HypothesisDraft(
                        hypothesis_text="Предварительная магнитная сепарация обогащает питание "
                        "и снижает нагрузку на последующий передел.",
                        causal_chain=[
                            {"entity_in": "пирротиновые хвосты", "activity": "магнитная сепарация", "entity_out": "магнитный концентрат"},
                        ],
                        key_process="магнитная сепарация",
                        doi_sources=["10.1016/j.mineng.2021.106987"],
                        expected_kpi_gain=0.55,
                    ),
                    HypothesisDraft(
                        hypothesis_text="Дробная подача собирателя и коррекция pH повышают "
                        "извлечение сульфидов Ni/Cu без роста расхода реагентов.",
                        causal_chain=[
                            {"entity_in": "пирротиновые хвосты", "activity": "флотация", "entity_out": "сульфидный концентрат"},
                        ],
                        key_process="флотация",
                        doi_sources=["10.3390/min14040331"],
                        expected_kpi_gain=0.45,
                    ),
                ]
            )

    class _FakeLLM:
        def with_structured_output(self, schema):
            return RunnableLambda(_FakeStructured(schema).invoke)

    demo_input = (
        "КГМК: отвальные пирротиновые хвосты, теряем никель. Бюджет 5 млн у.е., срок 6 мес.\n"
        "| Поток | Масса, СМТ | Потери Ni, % |\n|---|---|---|\n"
        "| Пирротиновые хвосты | 120000 | 0.6 |"
    )

    pipeline = FabrikaPipeline(llm=_FakeLLM())
    result = pipeline.generate(demo_input)

    print("REFUSED:", result.refused, "| MESSAGE:", result.message)
    print(f"Гипотез после отсева: {len(result.hypotheses)}\n")
    for i, h in enumerate(result.hypotheses, 1):
        print(f"#{i}  final={h.final_score}  (risk={h.risk_score} value={h.value_score} novelty={h.novelty_score})")
        print("   ", h.hypothesis_text)
        print("    PMDco:", " ; ".join(link.as_arrow() for link in h.causal_chain))
        print("    DOI:", h.doi_sources)
    print("\nJSON выхода (первая гипотеза):")
    if result.hypotheses:
        print(json.dumps(result.hypotheses[0].model_dump(), ensure_ascii=False, indent=2))
