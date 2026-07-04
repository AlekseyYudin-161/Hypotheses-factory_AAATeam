"""Общий слой доступа к LLM через ProxyAPI (OpenAI-совместимый API).
Импортируют и индексация, и query (ML/RL). Стандартный OpenAI SDK, без префиксов."""

from openai import OpenAI
from src.core.config import (
    PROXYAPI_API_KEY, PROXYAPI_BASE_URL,
    MODEL_MAIN, EMB_MODEL,
)

client = OpenAI(
    api_key=PROXYAPI_API_KEY,
    base_url=PROXYAPI_BASE_URL,
)

def llm_generate(prompt: str,
                 model: str = MODEL_MAIN,
                 temperature: float = 0.2,
                 max_tokens: int = 1_500) -> str:
    """Генерация. Модель передаёт вызывающий (по умолчанию MODEL_MAIN = gpt-4.1-mini)."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def _embed(text: str):
    e = client.embeddings.create(
        model=EMB_MODEL,        # text-embedding-3-small; без префикса, без encoding_format
        input=text,
    )
    return e.data[0].embedding

# У OpenAI одна модель эмбеддинга (нет doc/query-разделения как у Yandex).
def embed_doc(text: str):       # ИНДЕКСАЦИЯ (мой таск)
    return _embed(text)

def embed_query(text: str):     # RETRIEVAL (у ML/RL)
    return _embed(text)
