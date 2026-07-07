"""
Мост между RAG-ядром и фронтендом (Streamlit / API_CONTRACT.md).

FabrikaPipeline отдаёт узкий объект Hypothesis (text / causal_chain / DOI / score).
UI же ждёт «богатую» гипотезу (novelty/risk/value/economic_value/success_probability,
evidence[] с source_url, roadmap, causal_chain и т.д.). Здесь — конвертация и
готовая функция generate_hypotheses(...) с той же сигнатурой, что в services.generator,
чтобы app.py мог переключиться на реальный пайплайн одной строкой импорта.
"""

from __future__ import annotations

import os
from typing import Any, List

from .pipeline import REFUSAL_MESSAGE, FabrikaPipeline
from .schemas import Hypothesis, PipelineResult

# Совместимость с фронтендом: единая формула агрегата 0..100 и корень источников.
try:  # избегаем жёсткой зависимости, если services недоступен в изоляции
    from services.generator import SOURCE_ROOT_URL, calculate_final_score
except Exception:  # pragma: no cover - фолбэк для автономного запуска rag-пакета
    SOURCE_ROOT_URL = os.getenv("SOURCE_ROOT_URL", "https://disk.yandex.ru/d/qE55fooRQGNVVA")

    def calculate_final_score(novelty, risk, value, economic_value, w_n, w_r, w_v):
        total = w_n + w_r + w_v
        combined = 0.65 * value + 0.35 * economic_value
        return round(((w_n * novelty + w_r * (1 - risk) + w_v * combined) / total) * 100, 2)


def _source_url_for(doi: str) -> str:
    """
    У внутренних корпусных файлов (doi вида 'file:Хвосты_КГМК.xlsx') нет URL —
    фронтенд требует source_url, поэтому отдаём стабильную ссылку на материалы кейса.
    Настоящие DOI/URL (http...) пробрасываем как есть.
    """
    if doi.startswith(("http://", "https://", "doi:", "10.")):
        return doi if doi.startswith("http") else f"https://doi.org/{doi.removeprefix('doi:')}"
    return SOURCE_ROOT_URL


def hypothesis_to_frontend(h: Hypothesis, kpi: str) -> dict[str, Any]:
    """Конвертирует одну Hypothesis в dict под hypothesis_schema.json."""
    economic = round(1.0 - h.risk_score, 3)  # экономика «зашита» в risk → инвертируем
    chain_strings = [link.as_arrow() for link in h.causal_chain]

    evidence = [
        {
            "triplet": chain_strings[0] if chain_strings else "",
            "source": doi,
            "source_url": _source_url_for(doi),
            "page": None,
            "chunk_id": f"{doi}",
            "evidence_fragment": h.hypothesis_text,
            "support_type": "analogy",
            "confidence": round(min(1.0, 0.5 + h.value_score / 2), 2),
            "knowledge_base": "process_data",
        }
        for doi in (h.doi_sources or ["internal"])
    ]

    return {
        "id": f"H-{abs(hash(h.hypothesis_text)) % 100000}",
        "hypothesis": h.hypothesis_text,
        "rationale": h.hypothesis_text,
        "mechanism": h.key_process,
        "kpi_link": f"Гипотеза направлена на достижение KPI: «{kpi}».",
        "constraints_fit": "",
        "expected_effect": "",
        "industrial_scale": "",
        "novelty": {"score": h.novelty_score, "why": "Оценка по совпадению триплетов в графе знаний."},
        "risk": {"score": h.risk_score, "why": "Экономико-технологический риск (стоимость реагентов на массу vs бюджет)."},
        "value": {"score": h.value_score, "why": "Теоретический прирост KPI по контексту корпуса."},
        "economic_value": {"score": economic, "why": "Инверсия экономического риска доизвлечения."},
        "success_probability": {"score": h.final_score, "why": "Агрегированный балл трёх осей."},
        "verification_recommendation": "Провести базовый и опытный прогоны, зафиксировать материальный баланс и изменение KPI.",
        "causal_chain": chain_strings,
        "resource_estimate": {"time": "", "cost": "", "volume": ""},
        "evidence": evidence,
        "roadmap": [],
        "status": "draft",
        "final_score": calculate_final_score(
            h.novelty_score, h.risk_score, h.value_score, economic, 0.4, 0.3, 0.3
        ),
    }


def result_to_frontend(result: PipelineResult, kpi: str) -> List[dict[str, Any]]:
    """PipelineResult → список богатых гипотез. Отказ ⇒ пустой список."""
    if result.refused:
        return []
    return [hypothesis_to_frontend(h, kpi) for h in result.hypotheses]


def build_rich_input(kpi: str, constraints: str) -> str:
    """Склеиваем KPI и ограничения фронтенда в «богатый вход» для декомпозиции."""
    parts = [f"Задача (KPI): {kpi}"]
    if constraints and constraints.strip():
        parts.append(f"Ограничения: {constraints.strip()}")
    return "\n".join(parts)


def generate_hypotheses(
    kpi: str,
    constraints: str = "",
    language: str = "ru",
    knowledge_bases: list[str] | None = None,
    images: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Drop-in замена services.generator.generate_hypotheses, но на реальном RAG-пайплайне.

    Требует DB_DSN + PROXYAPI_* в окружении. Возвращает список богатых гипотез
    для UI; при легальном отказе — пустой список (UI показывает сообщение об
    оптимальности текущего решения).
    """
    pipeline = FabrikaPipeline.from_env()
    result = pipeline.generate(build_rich_input(kpi, constraints), images=images)
    return result_to_frontend(result, kpi)


# Экспортируем сообщение отказа, чтобы UI мог показать его при пустом ответе.
__all__ = [
    "generate_hypotheses",
    "hypothesis_to_frontend",
    "result_to_frontend",
    "build_rich_input",
    "REFUSAL_MESSAGE",
]
