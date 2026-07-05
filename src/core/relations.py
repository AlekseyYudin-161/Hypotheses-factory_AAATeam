"""Контролируемый словарь связей триплета (для КОРПУСА, где колонки известны).
relation = одно из RELATIONS; исходный текст -> relation_raw."""

RELATIONS = ["increases", "decreases", "enables", "suppresses", "no_effect", "correlates"]

_KEYS = {
    "increases":  ["повыш", "увелич", "рост", "improv", "increas", "↑"],
    "decreases":  ["сниж", "умень", "паден", "decreas", "reduc", "↓"],
    "no_effect":  ["не влия", "без эффект", "no effect"],
    "correlates": ["коррел", "correlat", "связан"],
    "enables":    ["позвол", "обеспеч", "enabl"],
    "suppresses": ["подавл", "ингиб", "suppress"],
}

def normalize_relation(raw):
    if raw is None:
        return None
    s = str(raw).lower()
    for rel, keys in _KEYS.items():
        if any(k in s for k in keys):
            return rel
    return None

def direction_from_relation(rel):
    return {"increases": "up", "decreases": "down"}.get(rel, "flat")
