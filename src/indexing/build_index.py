"""ОФЛАЙН-индексация выданного КОРПУСА: строка -> чанк(+эмбеддинг) -> триплет -> pgvector.
  python -m src.indexing.build_index data/corpus/файл.xlsx --source "Данные о процессах" """

import os
import argparse
from src.core.db import connect, upsert_document, insert_chunk, insert_triplet
from src.core.embedding import get_embedding
from src.core.sources import DEFAULT_SOURCE, is_valid_source
from src.indexing.parse_tables import read_table, is_empty_table, row_to_text, row_to_triplet, EXAMPLE_MAPPING

def index_table(path, source=DEFAULT_SOURCE, mapping=None) -> int:

    assert is_valid_source(source), f"неизвестная категория: {source}"
    mapping = mapping or EXAMPLE_MAPPING
    df = read_table(path)
    if is_empty_table(df):
        print(f"[skip] пустая таблица: {path}")
        return 0

    doc_doi = f"file:{os.path.basename(path)}"
    conn = connect()
    cur = conn.cursor()
    upsert_document(cur, doi=doc_doi, title=os.path.basename(path), source=source)
    n = 0
    for _, row in df.iterrows():
        text = row_to_text(row, mapping)
        if not text.strip():
            continue
        cid = insert_chunk(cur, doc_doi, text, get_embedding(text))
        insert_triplet(cur, doi=doc_doi, chunk_id=cid, **row_to_triplet(row, mapping))
        n += 1

    conn.commit()
    cur.close()
    conn.close()
    return n

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    a = ap.parse_args()
    print(f"проиндексировано строк: {index_table(a.path, a.source)} (source={a.source})")
