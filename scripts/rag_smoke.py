"""
Смоук-тест полного пайплайна на реальном корпусе КГМК.

Требует поднятой pgvector с загруженным corpus_dump.sql и заданных переменных:
    DB_DSN, PROXYAPI_API_KEY, PROXYAPI_BASE_URL  (EMB_MODEL — опционально)

Запуск из корня проекта:
    python -m scripts.rag_smoke
или:
    python scripts/rag_smoke.py
"""

from __future__ import annotations

import json
import os
import sys

# Позволяем запускать как файл (python scripts/rag_smoke.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

from rag import FabrikaPipeline  # noqa: E402
from rag.service import result_to_frontend  # noqa: E402

load_dotenv()

RICH_INPUT = (
    "Проблема: КГМК теряет никель с отвальными пирротиновыми хвостами. "
    "Нужно предложить способы доизвлечения ценных металлов.\n"
    "Ограничения: не увеличивать расход реагентов, использовать существующее оборудование.\n"
    "| Материал | СМТ | Ni, % |\n|---|---|---|\n| Отвальные хвосты | 5824591 | 0.178 |"
)


def main() -> None:
    if not os.getenv("DB_DSN"):
        raise SystemExit("DB_DSN не задан — сначала подними pgvector и загрузи корпус.")

    pipeline = FabrikaPipeline.from_env()
    result = pipeline.generate(RICH_INPUT)

    print("refused:", result.refused, "| message:", result.message)
    print(f"гипотез: {len(result.hypotheses)}\n")
    for i, h in enumerate(result.hypotheses, 1):
        print(f"#{i} final={h.final_score} risk={h.risk_score} value={h.value_score} novelty={h.novelty_score}")
        print("   ", h.hypothesis_text)
        print("    PMDco:", " ; ".join(l.as_arrow() for l in h.causal_chain))
        print("    DOI:", h.doi_sources, "\n")

    # Проверяем и адаптер под фронтенд.
    frontend = result_to_frontend(result, kpi="Доизвлечение никеля из отвальных хвостов")
    print("frontend-формат (первая гипотеза):")
    print(json.dumps(frontend[0] if frontend else {}, ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
