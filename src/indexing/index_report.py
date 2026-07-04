"""Индексация ОТЧЁТНЫХ таблиц (балансы, не «материал-условие-эффект»): путь A.
Строка -> текст (все колонки) -> чанк + эмбеддинг в knowledge_chunks. БЕЗ триплетов.
Положить в: src/indexing/index_report.py

Запуск:
  python -m src.indexing.index_report data/corpus/Хвосты_КГМК.xlsx --header 1 --source "Данные о процессах"
"""

import os
import argparse
from src.core.db import connect, upsert_document, insert_chunk
from src.core.embedding import get_embedding
from src.core.sources import DEFAULT_SOURCE, is_valid_source
from src.indexing.parse_tables import read_table, is_empty_table, row_to_generic_text

def index_report_table(path, source=DEFAULT_SOURCE, header=0) -> int:
    assert is_valid_source(source), f"неизвестная категория: {source}"
    df = read_table(path, header=header)
    if is_empty_table(df):
        print(f"[skip] пустая таблица: {path}")
        return 0

    doc_doi = f"file:{os.path.basename(path)}"
    conn = connect()
    cur = conn.cursor()
    upsert_document(cur, doi=doc_doi, title=os.path.basename(path), source=source)
    n = 0
    for _, row in df.iterrows():
        text = row_to_generic_text(row)
        if not text.strip():
            continue
        insert_chunk(cur, doc_doi, text, get_embedding(text))
        n += 1
    conn.commit()
    cur.close()
    conn.close()
    return n

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--header", type=int, default=0)
    a = ap.parse_args()
    print(f"проиндексировано строк: {index_report_table(a.path, a.source, a.header)}")
