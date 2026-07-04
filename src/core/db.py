"""Подключение к pgvector + хелперы записи. Регистрируем vector-тип."""

import psycopg2
from pgvector.psycopg2 import register_vector
from src.core.config import DB_DSN

def connect():
    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)   # чтобы передавать embedding списком float
    return conn

def insert_chunk(cur, doi, chunk_text, embedding):
    cur.execute(
        """INSERT INTO knowledge_chunks (doi, chunk_text, embedding, text_search)
           VALUES (%s, %s, %s, to_tsvector('russian', %s))
           RETURNING chunk_id""",
        (doi, chunk_text, embedding, chunk_text),
    )
    return cur.fetchone()[0]

def insert_triplet(cur, **f):
    cur.execute(
        """INSERT INTO triplets
           (doi, chunk_id, material_raw, material_canonical, condition,
            effect_property, relation, relation_raw, value, unit, direction)
           VALUES (%(doi)s, %(chunk_id)s, %(material_raw)s, %(material_canonical)s,
                   %(condition)s, %(effect_property)s, %(relation)s, %(relation_raw)s,
                   %(value)s, %(unit)s, %(direction)s)""",
        f,
    )
