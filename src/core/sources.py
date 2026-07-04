# src/core/sources.py — единый список категорий баз знаний (для КОРПУСА и фильтра DA).

SOURCE_CATEGORIES = [
    "Научные публикации", "Патенты", "Внутренние отчёты",
    "Исторические эксперименты", "Данные о материалах",
    "Данные о процессах", "Открытые источники",
]

DEFAULT_SOURCE = "uncategorized"   # заглушка, пока категория не проставлена

ALL_SOURCES = SOURCE_CATEGORIES + [DEFAULT_SOURCE]

def is_valid_source(s: str) -> bool:
    return s in ALL_SOURCES
