import os
from dotenv import load_dotenv

load_dotenv()

# YANDEX_API_KEY   = os.environ["YANDEX_API_KEY"]
# YANDEX_FOLDER_ID = os.environ["YANDEX_FOLDER_ID"]
# YANDEX_BASE_URL  = os.environ.get("YANDEX_BASE_URL", "https://ai.api.cloud.yandex.net/v1")

# # --- Генеративные модели ---
# MODEL_PRO    = "yandexgpt-5.1"          # RU-домен: IE-триплеты, генерация
# MODEL_BIG    = "qwen3-235b-a22b-fp8"    # синтез гипотез: reasoning + контекст 256k
# MODEL_LITE   = "yandexgpt-5-lite"       # скорость / суммаризация / массовый IE
# MODEL_VISION = "qwen3.6-35b-a3b"        # изображения Base64

# # --- Эмбеддеры ---
# EMB_DOC    = "text-search-doc/latest"    # эмбеддинг ДОКУМЕНТОВ/чанков (индексация)
# EMB_QUERY  = "text-search-query/latest"  # эмбеддинг ЗАПРОСА (retrieval, у ML/RL)

# DB_DSN  = os.environ.get("DB_DSN", "postgresql://postgres:postgres@localhost:5432/hypothesis_factory")
# EMB_DIM = int(os.environ.get("EMB_DIM", "256"))   # ПОДТВЕРДИ смоук-тестом

PROXYAPI_API_KEY = os.environ["PROXYAPI_API_KEY"]
PROXYAPI_BASE_URL = os.environ.get("PROXYAPI_BASE_URL", "https://api.proxyapi.ru/openai/v1")

MODEL_MAIN = "gpt-4.1-mini"
MODEL_VISION = "gpt-4.1-mini"
EMB_MODEL  = "text-embedding-3-small"
EMB_DIM = 1536

DB_DSN  = os.environ.get("DB_DSN", "postgresql://postgres:postgres@localhost:5432/hypothesis_factory")