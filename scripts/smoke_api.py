"""СМОУК-ТЕСТ. Запусти ПЕРВЫМ:  python scripts/smoke_api.py
Проверяет, что живы (1) генерация и (2) эмбеддер, и печатает размерность вектора."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.llm import llm_generate, embed_doc

print("== LLM (генерация) ==")
print(llm_generate("Придумай 3 необычные идеи для стартапа в сфере путешествий.", max_output_tokens=20))

print("\n== EMBED (эмбеддер) ==")
v = embed_doc("тестовый чанк про обогащение руды, хвосты и шлаки")
print("dim =", len(v), " <- впиши в .env (EMB_DIM) и в db/schema.sql vector(N)")
print("первые 5:", v[:5])
