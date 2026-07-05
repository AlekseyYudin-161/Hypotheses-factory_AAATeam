"""общий движок парсинга таблиц. Два режима:
- КОРПУС (колонки известны): row_to_triplet по MAPPING -> триплеты.
- ЗАГРУЖЕННОЕ (колонки неизвестны): row_to_generic_text -> просто текст в контекст."""

import pandas as pd
from src.core.relations import normalize_relation, direction_from_relation

# Для КОРПУСА: подставь реальные имена колонок после inventory.py.
EXAMPLE_MAPPING = {
    "material": "Материал", "condition": "Режим", "effect_property": "Свойство",
    "relation": "Эффект", "value": "Значение", "unit": "Ед",
}

def read_table(path: str, header: int = 0) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    df = pd.read_excel(path, header=header)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]  # выкинуть пустые Unnamed
    df = df.dropna(how="all")                                          # выкинуть пустые строки
    return df

def is_empty_table(df: pd.DataFrame) -> bool:
    return df is None or df.empty or df.dropna(how="all").empty

def _get(row, mapping, key):
    col = mapping.get(key)
    if col and col in row and pd.notna(row[col]):
        return row[col]
    return None

def _to_float(v):
    if v is None:
        return None
    try:
        return float(str(v).replace(" ", "").replace(",", ".").split()[0])
    except (ValueError, IndexError):
        return None

# --- режим КОРПУС ---
def row_to_text(row, mapping) -> str:
    parts = [f"{k}: {_get(row, mapping, k)}" for k in
             ("material","condition","effect_property","relation","value","unit")
             if _get(row, mapping, k) is not None]
    return "; ".join(parts)

def row_to_triplet(row, mapping) -> dict:
    raw = _get(row, mapping, "relation")
    rel = normalize_relation(raw)
    def s(k):
        v=_get(row,mapping,k)
        return str(v) if v is not None else None

    return {
        "material_raw": s("material"), "material_canonical": None,
        "condition": s("condition"), "effect_property": s("effect_property"),
        "relation": rel, "relation_raw": str(raw) if raw is not None else None,
        "value": _to_float(_get(row,mapping,"value")), "unit": s("unit"),
        "direction": direction_from_relation(rel),
    }

# --- режим ЗАГРУЖЕННОЕ (колонки неизвестны) ---
def row_to_generic_text(row) -> str:
    return "; ".join(f"{c}: {row[c]}" for c in row.index if pd.notna(row[c]))
