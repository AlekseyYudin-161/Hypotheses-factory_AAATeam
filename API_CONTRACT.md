# API contract

## POST /generate

```json
{
  "kpi": "Повысить извлечение никеля из руды на 8%",
  "constraints": "Не увеличивать расход реагентов; использовать существующее оборудование",
  "language": "ru",
  "use_open_sources": true,
  "industrial_scale": true
}
```

Backend самостоятельно определяет количество релевантных гипотез.

```json
{
  "hypotheses": [
    {
      "id": "H-001",
      "hypothesis": "...",
      "rationale": "...",
      "mechanism": "...",
      "expected_effect": "...",
      "industrial_scale": "...",
      "kpi_link": "...",
      "constraints_fit": "...",
      "novelty": {"score": 0.82, "why": "..."},
      "risk": {"score": 0.34, "why": "..."},
      "value": {"score": 0.91, "why": "..."},
      "economic_value": {"score": 0.78, "why": "..."},
      "success_probability": {"score": 0.71, "why": "..."},
      "verification_recommendation": "...",
      "causal_chain": ["воздействие", "механизм", "промежуточный эффект", "KPI"],
      "resource_estimate": {
        "time": "2–6 недель",
        "cost": "средние затраты",
        "volume": "1 промышленный тест"
      },
      "evidence": [
        {
          "triplet": "...",
          "source": "...",
          "source_url": "https://...",
          "page": 125,
          "chunk_id": "CH-001",
          "evidence_fragment": "...",
          "support_type": "supports",
          "confidence": 0.87
        }
      ],
      "roadmap": [],
      "status": "draft"
    }
  ]
}
```

`source_url` обязателен для каждого источника. Открытые источники подключаются системой автоматически.


## Дополнительные документы

UI передаёт загруженные документы в поле `documents`:

```json
{
  "documents": [
    {
      "name": "report.pdf",
      "mime_type": "application/pdf",
      "size_bytes": 125000,
      "page_count": 14,
      "status": "text_extracted",
      "text": "Извлечённый текст документа..."
    }
  ]
}
```

Для сканированных PDF без текстового слоя `status` равен `requires_ocr`, а поле `text` может быть пустым. Backend может отправить такой документ в OCR-конвейер.
