import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.llm import embed_doc
from src.core.db import connect, insert_chunk


TEXT = "Флотация хвостов повышает извлечение меди при снижении расхода реагента"
conn = connect()
cur = conn.cursor()

cid = insert_chunk(cur, doi="test-1", chunk_text=TEXT, embedding=embed_doc(TEXT))
conn.commit()
cur.execute("SELECT count(*) FROM knowledge_chunks;")
print("chunk_id =", cid, "| всего строк:", cur.fetchone()[0])
cur.close()
conn.close()
