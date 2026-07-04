"""Общий слой доступа к Yandex AI Studio. Импортируют и индексация, и query (ML/RL).
Yandex — OpenAI-совместимый API. Модель задаётся как gpt://{folder}/{model}."""

from openai import OpenAI
from src.core.config import (
    YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_BASE_URL,
    MODEL_PRO, EMB_DOC, EMB_QUERY,
)

client = OpenAI(
    api_key=YANDEX_API_KEY,
    project=YANDEX_FOLDER_ID,
    base_url=YANDEX_BASE_URL,
)

def _gpt_uri(model: str) -> str:
    return f"gpt://{YANDEX_FOLDER_ID}/{model}"

def llm_generate(prompt: str, model: str = MODEL_PRO,
                 temperature: float = 0.2, max_output_tokens: int =1_500) -> str:
    """Генерация. model=MODEL_PRO для IE/синтеза, MODEL_LITE для скорости."""

    response = client.responses.create(
        model=_gpt_uri(model),
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return response.output[0].content[0].text
    # try:
    #     return response.output[0].content[0].text
    # except AttributeError:
    #     # handle the case when response is None or does not have the expected attributes
    #     return "Error: No response received"

def _embed(text: str, emb_model: str):
    e = client.embeddings.create(
        model=f"emb://{YANDEX_FOLDER_ID}/{emb_model}",
        input=text,
        encoding_format="float",                        # <-- Yandex не поддерживает base64
    )
    return e.data[0].embedding

def embed_doc(text: str):                               # ИНДЕКСАЦИЯ (мой таск)
    return _embed(text, EMB_DOC)

def embed_query(text: str):                             # RETRIEVAL (у ML/RL)
    return _embed(text, EMB_QUERY)
