# API contract

## POST /generate

```json
{
  "kpi": "Повысить извлечение никеля из руды на 8%",
  "constraints": "Не увеличивать расход реагентов; использовать существующее оборудование",
  "language": "ru",
  "knowledge_bases": [
    "scientific_publications",
    "patents",
    "internal_reports",
    "historical_experiments",
    "materials_data",
    "process_data",
    "open_sources"
  ],
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

`source_url` обязателен для каждого источника. Поле `knowledge_bases` принимает одну, несколько или все поддерживаемые базы знаний.
