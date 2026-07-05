"""Единая точка эмбеддинга. MOCK-режим на случай простоя API:
  MOCK_EMBEDDINGS=1 -> вектор нулей нужной размерности."""

import os
from src.core.config import EMB_DIM

MOCK = os.environ.get("MOCK_EMBEDDINGS", "0") == "1"

def get_embedding(text: str):
    if MOCK:
        return [0.0] * EMB_DIM
    from src.core.llm import embed_doc
    return embed_doc(text)
