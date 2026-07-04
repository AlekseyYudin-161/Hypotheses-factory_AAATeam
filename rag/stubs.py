"""
Заглушки внешних модулей — НЕ моя зона ответственности.

Эти классы имитируют интерфейсы, которые готовят другие участники команды:
  * KnowledgeBaseStub  — pgvector от Tech Lead (гибридный поиск + граф триплетов).
  * ChemistStub        — функции risk_score/time_ratio от химика-экономиста.

Пайплайн зависит только от ЭТИХ сигнатур. Когда реальные модули будут готовы,
достаточно подменить объект с тем же интерфейсом (duck typing), не трогая ядро.

Ссылки на реальные реализации-ориентиры лежат в
preparation_NORNIKEL_AI_SCIENCE_HACK_2026/rag/database.py и scorer.py.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class Chunk(TypedDict):
    """Один результат ретривала: кусок текста + привязанный к нему DOI."""

    doi: str
    chunk_text: str
    title: str
    rrf_score: float


class KnowledgeBaseStub:
    """
    Имитация pgvector-базы (гибридный поиск = вектор + BM25 с RRF-ранжированием).

    Реальная версия — DatabaseClient.hybrid_search в reference-проекте:
    она фильтрует по white/black-list, считает косинусное расстояние `<=>`
    и объединяет результаты формулой RRF. Здесь — детерминированная выдача,
    чтобы можно было прогонять пайплайн офлайн, без БД и эмбеддингов.
    """

    def hybrid_search(
        self,
        whitelist: str,
        blacklist: str,
        mechanism: str,
        top_k: int = 5,
    ) -> List[Chunk]:
        """
        ШАГ 2. «Идём» в базу и возвращаем релевантные чанки с DOI.

        whitelist  — target_property (что ищем, например 'доизвлечение никеля');
        blacklist  — target_material (что исключаем, например 'отвальные хвосты');
        mechanism  — конкретная сущность из декомпозиции для векторного поиска.
        """
        # Заглушка: небольшой корпус «как из статей по металлургии».
        corpus = [
            Chunk(
                doi="10.3390/min14040331",
                title="Machine Learning for Optimal Flotation Performance",
                chunk_text=(
                    f"Тонкое регулирование pH и дробная подача собирателя при {mechanism} "
                    "повышали извлечение сульфидов Ni/Cu на 4–7% без роста расхода реагентов."
                ),
                rrf_score=0.031,
            ),
            Chunk(
                doi="10.3390/ma15196536",
                title="Froth Flotation of Chalcopyrite/Pyrite Ore: A Critical Review",
                chunk_text=(
                    f"Автоклавное окислительное выщелачивание пирротиновых концентратов ({mechanism}) "
                    "переводит никель в раствор с извлечением >90% при давлении кислорода 0.7 МПа."
                ),
                rrf_score=0.029,
            ),
            Chunk(
                doi="10.1016/j.mineng.2021.106987",
                title="Magnetic separation of pyrrhotite tailings",
                chunk_text=(
                    f"Предварительная магнитная сепарация ({mechanism}) обогащает питание "
                    "по магнитной фракции и снижает нагрузку на последующее выщелачивание."
                ),
                rrf_score=0.026,
            ),
        ]
        # Реальный поиск учитывал бы whitelist/blacklist; для стаба просто отдаём top_k.
        return corpus[:top_k]

    def check_novelty_triplet(self, subject: str, obj: str) -> float:
        """
        ШАГ 4 (новизна). Есть ли уже связь subject-*-object в графе триплетов?

        Возвращает 1.0 — связь не встречалась (высокая новизна),
        ниже — если аналогичные переделы уже описаны в корпусе.
        """
        known_pairs = {
            ("пирротиновые хвосты", "флотация"),
            ("медь", "цементация"),
        }
        return 0.2 if (subject.lower(), obj.lower()) in known_pairs else 1.0


class ChemistStub:
    """
    Имитация модуля химика-экономиста (risk_score / time_ratio).

    Экономика первична: если процесс дорогой относительно бюджета или масса
    отходов не окупает реагенты — риск высокий (гипотеза получит пенальти).
    """

    # грубые удельные затраты реагентов, у.е. на тонну, для типовых процессов
    _PROCESS_COST_PER_T = {
        "автоклавное выщелачивание": 45.0,
        "флотация": 8.0,
        "магнитная сепарация": 3.0,
        "цементация": 12.0,
        "биовыщелачивание": 6.0,
    }
    _DEFAULT_COST_PER_T = 20.0

    def risk_score(
        self,
        key_process: str,
        mass_smt: Optional[float],
        budget: Optional[float],
    ) -> float:
        """
        Возвращает экономико-технологический риск 0..1 (выше — рискованнее/дороже).

        Логика-заглушка: оцениваем полную стоимость реагентов на массу отходов и
        сравниваем с бюджетом. Нет бюджета/массы — берём умеренный риск 0.5.
        """
        cost_per_t = self._PROCESS_COST_PER_T.get(
            key_process.strip().lower(), self._DEFAULT_COST_PER_T
        )
        if mass_smt is None or budget is None or budget <= 0:
            # Недостаточно данных для экономики — умеренная неопределённость.
            return 0.5

        total_cost = cost_per_t * mass_smt
        ratio = total_cost / budget  # доля бюджета, съедаемая реагентами
        # ratio<=0.5 -> дёшево (низкий риск); ratio>=1.5 -> выходим за бюджет (риск ~1).
        risk = (ratio - 0.5) / 1.0
        return max(0.0, min(1.0, risk))
