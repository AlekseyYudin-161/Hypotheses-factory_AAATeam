"""СЦЕНАРИЙ: файлы, загруженные ВМЕСТЕ С ПРОМПТОМ (жюри). Быстрый проход под лимит 2 мин.
Возвращает контекст В ПАМЯТИ (не пишем в pgvector -> изоляция между запросами).
ML/RL кладёт этот контекст в промпт генерации, корпус тянет из pgvector отдельно.

  from src.indexing.ingest_uploaded import ingest_uploaded
  ctx = ingest_uploaded(["/tmp/hvosty.xlsx", "/tmp/scheme.png"])
  ctx["context_text"], ctx["image_facts"], ctx["notes"], ctx["truncated"]
"""

import os
import base64
from src.indexing.parse_tables import read_table, is_empty_table, row_to_generic_text

MAX_TABLE_ROWS = 200        # лимит строк на таблицу (бюджет 2 мин)
MAX_PDF_PAGES  = 15
MAX_CHARS      = 20000      # общий потолок текста в контекст

def _ingest_table(path, notes):
    df = read_table(path)
    if is_empty_table(df):
        notes.append(f"{os.path.basename(path)}: пустая таблица, пропущено")
        return "", False

    truncated = len(df) > MAX_TABLE_ROWS
    lines = [row_to_generic_text(r) for _, r in df.head(MAX_TABLE_ROWS).iterrows()]
    return f"[Таблица {os.path.basename(path)}]\n" + "\n".join(lines), truncated

def _ingest_text(path, notes):
    ext = path.lower()
    if ext.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            r = PdfReader(path); pages = r.pages[:MAX_PDF_PAGES]
            txt = "\n".join((p.extract_text() or "") for p in pages)
            return f"[PDF {os.path.basename(path)}]\n{txt}", len(r.pages) > MAX_PDF_PAGES
        except Exception as e:
            notes.append(f"{os.path.basename(path)}: PDF не прочитан ({e})")
            return "", False

    with open(path, encoding="utf-8", errors="ignore") as f:
        return f"[Текст {os.path.basename(path)}]\n{f.read()}", False

def _ingest_image(path, notes):
    """P1: извлечение фактов из схемы/картинки через vision (баллы жюри)."""

    try:
        from src.core.llm import client
        from src.core.config import MODEL_VISION

        b64 = base64.b64encode(open(path, "rb").read()).decode()
        media = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        r = client.chat.completions.create(
            model=MODEL_VISION, max_tokens=800,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Извлеки факты из технологической схемы/изображения: "
                 "оборудование, потоки, материалы, параметры. Кратко, по пунктам."},
                {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
            ]}],
        )
        return r.choices[0].message.content
    except Exception as e:
        notes.append(f"{os.path.basename(path)}: картинка не обработана ({e})")
        return ""

def ingest_uploaded(paths, do_images=True) -> dict:
    text_parts, image_facts, notes = [], [], []
    truncated = False
    for path in paths:
        ext = path.lower()
        if ext.endswith((".xlsx", ".xls", ".csv", ".tsv")):
            t, tr = _ingest_table(path, notes)
            truncated |= tr
            if t:
                text_parts.append(t)
        elif ext.endswith((".pdf", ".txt", ".md")):
            t, tr = _ingest_text(path, notes)
            truncated |= tr
            if t:
                text_parts.append(t)
        elif ext.endswith((".png", ".jpg", ".jpeg")):
            if do_images:
                f = _ingest_image(path, notes)
                if f:
                    image_facts.append(f"[Схема {os.path.basename(path)}]\n{f}")
        else:
            notes.append(f"{os.path.basename(path)}: неподдерживаемый формат, пропущено")

    context_text = "\n\n".join(text_parts)[:MAX_CHARS]
    if len("\n\n".join(text_parts)) > MAX_CHARS:
        truncated = True
    return {"context_text": context_text,
            "image_facts": image_facts,
            "notes": notes,
            "truncated": truncated}
