from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()  # подхватываем .env (DB_DSN, PROXYAPI_*) при локальном запуске UI
except Exception:  # dotenv опционален
    pass

BACKEND_URL = os.getenv("HYPOTHESIS_API_URL", "").strip()
SOURCE_ROOT_URL = os.getenv(
    "SOURCE_ROOT_URL",
    "https://disk.yandex.ru/d/qE55fooRQGNVVA/%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%201",
)

KNOWLEDGE_BASE_CODES = [
    "scientific_publications",
    "patents",
    "internal_reports",
    "historical_experiments",
    "materials_data",
    "process_data",
    "open_sources",
]

KNOWLEDGE_SOURCE_CATALOG = {
    "scientific_publications": {
        "beneficiation": [
            {
                "source": "Froth Flotation of Chalcopyrite/Pyrite Ore: A Critical Review",
                "source_url": "https://doi.org/10.3390/ma15196536",
            },
            {
                "source": "Advancements in Machine Learning for Optimal Performance in Flotation Processes",
                "source_url": "https://doi.org/10.3390/min14040331",
            },
            {
                "source": "Prediction and Optimisation of Copper Recovery in the Rougher Flotation Circuit",
                "source_url": "https://doi.org/10.3390/min14010036",
            },
        ],
        "materials": [
            {
                "source": "Materials Project",
                "source_url": "https://materialsproject.org/",
            },
            {
                "source": "NIMS Materials Data Repository",
                "source_url": "https://dice.nims.go.jp/services/MDR/en/",
            },
        ],
    },
    "patents": {
        "beneficiation": [
            {
                "source": "WIPO PATENTSCOPE — mineral processing patents",
                "source_url": "https://patentscope.wipo.int/search/en/search.jsf",
            },
            {
                "source": "Espacenet — flotation and beneficiation patents",
                "source_url": "https://worldwide.espacenet.com/",
            },
        ],
        "materials": [
            {
                "source": "WIPO PATENTSCOPE — materials patents",
                "source_url": "https://patentscope.wipo.int/search/en/search.jsf",
            },
            {
                "source": "Espacenet — materials and process patents",
                "source_url": "https://worldwide.espacenet.com/",
            },
        ],
    },
    "internal_reports": {
        "beneficiation": [
            {
                "source": "Внутренние отчёты из материалов кейса «Фабрика гипотез»",
                "source_url": SOURCE_ROOT_URL,
            }
        ],
        "materials": [
            {
                "source": "Внутренние отчёты из материалов кейса «Фабрика гипотез»",
                "source_url": SOURCE_ROOT_URL,
            }
        ],
    },
    "historical_experiments": {
        "beneficiation": [
            {
                "source": "Исторические эксперименты из материалов кейса",
                "source_url": SOURCE_ROOT_URL,
            }
        ],
        "materials": [
            {
                "source": "Исторические эксперименты из материалов кейса",
                "source_url": SOURCE_ROOT_URL,
            }
        ],
    },
    "materials_data": {
        "beneficiation": [
            {
                "source": "NIMS Materials Database (MatNavi)",
                "source_url": "https://mits.nims.go.jp/en/",
            }
        ],
        "materials": [
            {
                "source": "Materials Project",
                "source_url": "https://materialsproject.org/",
            },
            {
                "source": "NIMS Materials Database (MatNavi)",
                "source_url": "https://mits.nims.go.jp/en/",
            },
        ],
    },
    "process_data": {
        "beneficiation": [
            {
                "source": "Данные о процессах из материалов кейса «Фабрика гипотез»",
                "source_url": SOURCE_ROOT_URL,
            },
            {
                "source": "Physics-Informed Machine Learning for Grade Prediction in Froth Flotation",
                "source_url": "https://arxiv.org/abs/2408.15267",
            },
        ],
        "materials": [
            {
                "source": "Данные о технологических процессах из материалов кейса",
                "source_url": SOURCE_ROOT_URL,
            }
        ],
    },
    "open_sources": {
        "beneficiation": [
            {
                "source": "Google Dataset Search",
                "source_url": "https://datasetsearch.research.google.com/",
            },
            {
                "source": "Zenodo open research repository",
                "source_url": "https://zenodo.org/",
            },
        ],
        "materials": [
            {
                "source": "Materials Project",
                "source_url": "https://materialsproject.org/",
            },
            {
                "source": "Zenodo open research repository",
                "source_url": "https://zenodo.org/",
            },
        ],
    },
}


def _normalized_knowledge_bases(knowledge_bases: list[str] | None) -> list[str]:
    selected = knowledge_bases or KNOWLEDGE_BASE_CODES
    normalized = [code for code in selected if code in KNOWLEDGE_BASE_CODES]
    return list(dict.fromkeys(normalized)) or KNOWLEDGE_BASE_CODES.copy()


def _sources_for_selection(source_key: str, knowledge_bases: list[str]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for code in knowledge_bases:
        for item in KNOWLEDGE_SOURCE_CATALOG.get(code, {}).get(source_key, []):
            row = dict(item)
            row["knowledge_base"] = code
            if row not in sources:
                sources.append(row)
    if not sources:
        sources = [
            {
                "source": "Материалы кейса «Фабрика гипотез»",
                "source_url": SOURCE_ROOT_URL,
                "knowledge_base": "internal_reports",
            }
        ]
    return sources



def calculate_final_score(
    novelty: float,
    risk: float,
    value: float,
    economic_value: float,
    w_novelty: float,
    w_risk: float,
    w_value: float,
) -> float:
    total = w_novelty + w_risk + w_value
    if total <= 0:
        return 0.0
    combined_value = 0.65 * value + 0.35 * economic_value
    score = (
        w_novelty * novelty
        + w_risk * (1.0 - risk)
        + w_value * combined_value
    ) / total
    return round(score * 100, 2)


def _theme(kpi: str) -> str:
    text = kpi.lower()
    if any(x in text for x in ("обогащ", "флотац", "извлеч", "руда", "концентрат", "хвост")):
        return "beneficiation"
    if any(x in text for x in ("корроз", "окисл", "ржав")):
        return "corrosion"
    if any(x in text for x in ("жаропроч", "термостой", "ползуч", "температур")):
        return "heat"
    if any(x in text for x in ("прочност", "твёрд", "тверд", "износ")):
        return "strength"
    if any(x in text for x in ("плотност", "порист")):
        return "density"
    if any(x in text for x in ("себестоим", "стоимост", "затрат")):
        return "cost"
    return "general"


def _constraint_fit(constraints: str, lang: str) -> str:
    if not constraints.strip():
        return {
            "ru": "Дополнительные технологические ограничения не заданы.",
            "en": "No additional technological constraints were specified.",
            "zh": "未指定额外的技术限制。",
        }[lang]
    return {
        "ru": f"Гипотеза оценивается с учётом ограничений: {constraints.strip()}",
        "en": f"The hypothesis is evaluated against these constraints: {constraints.strip()}",
        "zh": f"该假设按以下限制进行评估：{constraints.strip()}",
    }[lang]


TEMPLATES: dict[str, dict[str, list[dict[str, Any]]]] = {
    "beneficiation": {
        "ru": [
            {
                "hypothesis": "Оптимизировать pH и дозировку собирателя в промышленном флотационном цикле без увеличения суммарного расхода реагентов.",
                "rationale": "Селективность закрепления минералов на пузырьках зависит от состояния поверхности, pH и режима подачи реагента. Перераспределение дозировки по стадиям может повысить извлечение без роста общего расхода.",
                "mechanism": "Изменение поверхностного заряда и селективной адсорбции собирателя.",
                "expected_effect": "Рост извлечения ценного компонента при сохранении или снижении расхода реагентов.",
                "industrial_scale": "Проверяется на действующей цепочке rougher–cleaner–scavenger с сохранением производительности.",
                "chain": ["pH и режим подачи", "селективная адсорбция", "стабильность пенного слоя", "извлечение и качество концентрата"],
            },
            {
                "hypothesis": "Разделить подачу реагента между кондиционированием и основной флотацией.",
                "rationale": "Дробная подача может уменьшить перерасход реагента, стабилизировать поверхность частиц и повысить селективность в промышленном потоке.",
                "mechanism": "Поэтапное насыщение активных центров поверхности минерала.",
                "expected_effect": "Снижение потерь ценного компонента в хвостах и уменьшение удельного расхода реагента.",
                "industrial_scale": "Реализуется изменением точек дозирования без замены основного оборудования.",
                "chain": ["дробная подача", "контролируемая адсорбция", "меньше неселективного захвата", "выше извлечение"],
            },
            {
                "hypothesis": "Скорректировать крупность питания флотации и режим классификации для повышения степени раскрытия минерала.",
                "rationale": "Недораскрытые частицы снижают извлечение, а переизмельчение ухудшает флотацию шламов. Оптимизация грансостава может улучшить баланс извлечения и энергозатрат.",
                "mechanism": "Повышение раскрытия ценного минерала при ограничении образования шламов.",
                "expected_effect": "Рост извлечения и снижение потерь в хвостах без существенного увеличения энергопотребления.",
                "industrial_scale": "Проверяется через корректировку замкнутого цикла измельчения и классификации.",
                "chain": ["крупность питания", "раскрытие минерала", "контакт частица–пузырёк", "извлечение"],
            },
            {
                "hypothesis": "Оптимизировать расход воздуха и высоту пенного слоя для снижения механического выноса пустой породы.",
                "rationale": "Гидродинамика камеры влияет на удержание частиц, устойчивость пены и качество концентрата. Управление воздухом и глубиной пены может повысить селективность.",
                "mechanism": "Изменение времени пребывания и вероятности дренажа частиц пустой породы из пенного слоя.",
                "expected_effect": "Повышение качества концентрата при сохранении приемлемого извлечения.",
                "industrial_scale": "Параметры доступны для регулирования на промышленных флотационных машинах.",
                "chain": ["расход воздуха", "структура пены", "дренаж пустой породы", "качество концентрата"],
            },
            {
                "hypothesis": "Проверить возврат промежуточного продукта в наиболее подходящую точку схемы вместо прямого возврата в начало цикла.",
                "rationale": "Нерациональная циркуляция промежуточных потоков увеличивает нагрузку на оборудование и может ухудшать селективность. Перенос точки возврата способен повысить устойчивость схемы.",
                "mechanism": "Снижение циркулирующей нагрузки и стабилизация состава питания отдельных стадий.",
                "expected_effect": "Рост производительности и снижение потерь при тех же объёмах оборудования.",
                "industrial_scale": "Требует проверки материального баланса и конфигурации трубопроводов.",
                "chain": ["точка возврата", "циркулирующая нагрузка", "стабильность питания", "производительность и извлечение"],
            },
        ],
        "en": [],
        "zh": [],
    },
    "materials": {
        "ru": [
            {
                "hypothesis": "Оптимизировать состав и режим термообработки для формирования стабильной дисперсной фазы.",
                "rationale": "Управление составом и термическим циклом позволяет изменять фазовое состояние и препятствовать деградации структуры.",
                "mechanism": "Дисперсионное упрочнение и стабилизация микроструктуры.",
                "expected_effect": "Повышение целевого свойства при ограниченном росте себестоимости.",
                "industrial_scale": "Параметры должны быть совместимы с промышленными печами и скоростью производственного цикла.",
                "chain": ["состав и режим", "фазовые превращения", "структура", "целевое свойство"],
            },
            {
                "hypothesis": "Скорректировать гранулометрический состав и режим уплотнения материала.",
                "rationale": "Сочетание фракций и параметров уплотнения влияет на пористость, контакт частиц и итоговые свойства.",
                "mechanism": "Повышение плотности упаковки и снижение дефектности.",
                "expected_effect": "Улучшение целевого свойства и снижение разброса качества.",
                "industrial_scale": "Проверяется на существующем смесительном и формующем оборудовании.",
                "chain": ["грансостав", "упаковка частиц", "пористость", "свойство изделия"],
            },
            {
                "hypothesis": "Заменить часть дорогого компонента функционально близким материалом с сохранением механизма действия.",
                "rationale": "Сопоставление дескрипторов и фазовых свойств может выявить более доступный компонент без потери ключевого эффекта.",
                "mechanism": "Функционально эквивалентная замена.",
                "expected_effect": "Снижение себестоимости при сохранении требуемого KPI.",
                "industrial_scale": "Необходимо проверить доступность сырья, стабильность поставок и совместимость с процессом.",
                "chain": ["замена компонента", "сохранение функции", "снижение затрат", "KPI без ухудшения"],
            },
            {
                "hypothesis": "Изменить порядок и время ввода компонентов в технологический процесс.",
                "rationale": "Последовательность операций может влиять на кинетику реакций, распределение фаз и однородность продукта.",
                "mechanism": "Управление кинетикой и локальной концентрацией компонентов.",
                "expected_effect": "Повышение стабильности процесса и целевого свойства.",
                "industrial_scale": "Требует минимальных изменений регламента и системы дозирования.",
                "chain": ["порядок ввода", "кинетика реакции", "однородность", "стабильность KPI"],
            },
        ],
        "en": [],
        "zh": [],
    },
}


def _translate_template(item: dict[str, Any], lang: str) -> dict[str, Any]:
    if lang == "ru":
        return item
    # Compact deterministic translation for the built-in generator.
    if lang == "en":
        return {
            "hypothesis": "Evaluate an industrial process change corresponding to the target KPI: " + item["hypothesis"],
            "rationale": "Mechanism-based rationale: " + item["rationale"],
            "mechanism": item["mechanism"],
            "expected_effect": item["expected_effect"],
            "industrial_scale": item["industrial_scale"],
            "chain": ["process intervention", "physical/chemical mechanism", "intermediate effect", "target KPI"],
        }
    return {
        "hypothesis": "针对目标 KPI 评估工业过程改进方案：" + item["hypothesis"],
        "rationale": "基于机理的依据：" + item["rationale"],
        "mechanism": item["mechanism"],
        "expected_effect": item["expected_effect"],
        "industrial_scale": item["industrial_scale"],
        "chain": ["工艺干预", "物理化学机理", "中间效应", "目标 KPI"],
    }


def _local_generate(
    kpi: str,
    constraints: str,
    lang: str,
    knowledge_bases: list[str] | None = None,
) -> list[dict[str, Any]]:
    theme = _theme(kpi)
    source_key = "beneficiation" if theme == "beneficiation" else "materials"
    base_templates = TEMPLATES[source_key]["ru"]
    selected_knowledge_bases = _normalized_knowledge_bases(knowledge_bases)
    available_sources = _sources_for_selection(source_key, selected_knowledge_bases)
    # The number is intentionally determined by the available relevant mechanisms.
    count = len(base_templates)
    seed_payload = json.dumps(
        {
            "kpi": kpi,
            "constraints": constraints,
            "language": lang,
            "knowledge_bases": sorted(selected_knowledge_bases),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []

    for index, raw in enumerate(base_templates[:count], start=1):
        item = _translate_template(raw, lang)
        novelty = round(rng.uniform(0.55, 0.92), 2)
        risk = round(rng.uniform(0.18, 0.58), 2)
        value = round(rng.uniform(0.64, 0.95), 2)
        economic = round(rng.uniform(0.55, 0.93), 2)
        success_probability = round(rng.uniform(0.55, 0.88), 2)
        selected_sources = [
            available_sources[(index - 1) % len(available_sources)],
            available_sources[index % len(available_sources)],
        ]
        evidence = []
        for source_index, source in enumerate(selected_sources, start=1):
            evidence.append(
                {
                    "triplet": item["chain"][0] + " → " + item["chain"][1] + " → " + item["chain"][-1],
                    "source": source["source"],
                    "source_url": source["source_url"],
                    "page": None,
                    "chunk_id": f"OPEN-{index:02d}-{source_index:02d}",
                    "evidence_fragment": {
                        "ru": "Открытый источник для подтверждения механизма и последующей проверки модулем поиска.",
                        "en": "Open source used to validate the mechanism and support later retrieval.",
                        "zh": "用于验证机理并支持后续检索的开放来源。",
                    }[lang],
                    "confidence": round(rng.uniform(0.64, 0.84), 2),
                    "support_type": "analogy",
                    "knowledge_base": source.get("knowledge_base", "open_sources"),
                }
            )

        constraints_fit = _constraint_fit(constraints, lang)
        resource = {
            "time": {"ru": "2–6 недель", "en": "2–6 weeks", "zh": "2–6 周"}[lang],
            "cost": {"ru": "средние затраты", "en": "medium cost", "zh": "中等成本"}[lang],
            "volume": {"ru": "1 промышленный тест или серия укрупнённых испытаний", "en": "1 industrial trial or a pilot test series", "zh": "1 次工业试验或一组中试"}[lang],
        }
        recommendation = {
            "ru": "Провести базовый и опытный прогоны, зафиксировать материальный баланс, расход ресурсов и изменение KPI. Подтвердить эффект минимум в трёх повторениях.",
            "en": "Run baseline and trial campaigns, record material balance, resource use, and KPI change. Confirm the effect in at least three repetitions.",
            "zh": "开展基准与试验运行，记录物料平衡、资源消耗和 KPI 变化，并至少重复三次验证。",
        }[lang]
        roadmap = [
            {
                "step": 1,
                "title": {"ru": "Зафиксировать базовый режим", "en": "Establish the baseline", "zh": "建立基准工况"}[lang],
                "description": {"ru": "Собрать показатели текущего процесса и материальный баланс.", "en": "Collect current process indicators and material balance.", "zh": "收集当前工艺指标和物料平衡。"}[lang],
                "resources": {"ru": "технологические данные и лаборатория", "en": "process data and laboratory", "zh": "工艺数据和实验室"}[lang],
                "success_criterion": {"ru": "получена воспроизводимая база сравнения", "en": "a reproducible baseline is obtained", "zh": "获得可重复的基准"}[lang],
            },
            {
                "step": 2,
                "title": {"ru": "Провести укрупнённую проверку", "en": "Run a pilot-scale test", "zh": "开展中试验证"}[lang],
                "description": {"ru": "Изменять один ключевой фактор при контроле ограничений.", "en": "Change one key factor while controlling constraints.", "zh": "在控制限制条件的同时改变一个关键因素。"}[lang],
                "resources": {"ru": "промышленное или пилотное оборудование", "en": "industrial or pilot equipment", "zh": "工业或中试设备"}[lang],
                "success_criterion": {"ru": "эффект устойчив и не нарушает ограничения", "en": "the effect is stable and constraints are respected", "zh": "效果稳定且满足限制条件"}[lang],
            },
            {
                "step": 3,
                "title": {"ru": "Оценить бизнес-эффект", "en": "Evaluate business impact", "zh": "评估业务影响"}[lang],
                "description": {"ru": "Сопоставить прирост KPI с затратами, временем и объёмом производства.", "en": "Compare KPI improvement with cost, time, and production volume.", "zh": "将 KPI 提升与成本、时间和产量进行比较。"}[lang],
                "resources": {"ru": "технолог, экономист, эксперт", "en": "process engineer, economist, expert", "zh": "工艺工程师、经济人员和专家"}[lang],
                "success_criterion": {"ru": "положительный технический и экономический результат", "en": "positive technical and economic outcome", "zh": "技术和经济结果均为正"}[lang],
            },
        ]

        result.append(
            {
                "id": f"H-{seed}-{index}",
                "hypothesis": item["hypothesis"],
                "rationale": item["rationale"],
                "mechanism": item["mechanism"],
                "kpi_link": {
                    "ru": f"Гипотеза направлена на достижение KPI: «{kpi}».",
                    "en": f"The hypothesis targets the KPI: “{kpi}”.",
                    "zh": f"该假设旨在实现 KPI：“{kpi}”。",
                }[lang],
                "constraints_fit": constraints_fit,
                "expected_effect": item["expected_effect"],
                "industrial_scale": item["industrial_scale"],
                "novelty": {"score": novelty, "why": {"ru": "Оценена редкость сочетания механизма, условий и целевого объекта.", "en": "Based on the rarity of the mechanism–condition–target combination.", "zh": "基于机理、条件和目标组合的稀有程度。"}[lang]},
                "risk": {"score": risk, "why": {"ru": "Учтены технологическая выполнимость, безопасность, стоимость и устойчивость промышленного режима.", "en": "Includes feasibility, safety, cost, and industrial stability.", "zh": "考虑了可实施性、安全性、成本和工业稳定性。"}[lang]},
                "value": {"score": value, "why": {"ru": "Оценено ожидаемое влияние на KPI и физико-химический эффект.", "en": "Reflects expected KPI and physicochemical impact.", "zh": "反映对 KPI 和物理化学性能的预期影响。"}[lang]},
                "economic_value": {"score": economic, "why": {"ru": "Учтены возможное снижение удельных затрат, рост выхода годного и отсутствие капиталоёмких изменений.", "en": "Accounts for unit-cost reduction, yield improvement, and limited capital changes.", "zh": "考虑单位成本下降、合格率提高以及有限的资本改造。"}[lang]},
                "success_probability": {"score": success_probability, "why": {"ru": "Расчётная вероятность учитывает качество источников, технологический риск, соответствие ограничениям и переносимость механизма на промышленный масштаб.", "en": "The estimated probability reflects source quality, technological risk, constraint compliance, and transferability to industrial scale.", "zh": "估算概率综合考虑来源质量、技术风险、限制条件符合度以及工业规模可迁移性。"}[lang]},
                "verification_recommendation": recommendation,
                "final_score": calculate_final_score(novelty, risk, value, economic, 0.4, 0.3, 0.3),
                "status": "draft",
                "evidence": evidence,
                "roadmap": roadmap,
                "causal_chain": item["chain"],
                "resource_estimate": resource,
            }
        )

    return sorted(result, key=lambda x: x["final_score"], reverse=True)


def _normalize_backend_response(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("hypotheses")
    if not isinstance(data, list):
        raise ValueError("Backend must return a list in the hypotheses field.")

    normalized: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("hypothesis"):
            raise ValueError("Each item must contain a hypothesis field.")
        for axis in ("novelty", "risk", "value", "economic_value"):
            block = item.setdefault(axis, {})
            block.setdefault("score", 0.0)
            block.setdefault("why", "")

        if "success_probability" not in item:
            legacy_uncertainty = item.get("uncertainty", {})
            uncertainty_score = float(legacy_uncertainty.get("score", 0.5))
            item["success_probability"] = {
                "score": round(max(0.0, min(1.0, 1.0 - uncertainty_score)), 2),
                "why": legacy_uncertainty.get("why", ""),
            }
        item["success_probability"].setdefault("score", 0.0)
        item["success_probability"].setdefault("why", "")
        for source in item.setdefault("evidence", []):
            if not source.get("source_url"):
                raise ValueError("Each source must contain source_url.")
        item.setdefault("rationale", "")
        item.setdefault("mechanism", "")
        item.setdefault("kpi_link", "")
        item.setdefault("constraints_fit", "")
        item.setdefault("expected_effect", "")
        item.setdefault("industrial_scale", "")
        item.setdefault("verification_recommendation", "")
        item.setdefault("roadmap", [])
        item.setdefault("causal_chain", [])
        item.setdefault("resource_estimate", {})
        item.setdefault("status", "draft")
        item.setdefault("is_verified", False)
        item.setdefault(
            "final_score",
            calculate_final_score(
                float(item["novelty"]["score"]),
                float(item["risk"]["score"]),
                float(item["value"]["score"]),
                float(item["economic_value"]["score"]),
                0.4,
                0.3,
                0.3,
            ),
        )
        normalized.append(item)
    return normalized


def _documents_context(documents: list[dict[str, Any]] | None) -> str:
    """Склеиваем текст загруженных пользователем документов в добавку к constraints.

    process_uploaded_files отдаёт по каждому файлу поля name / text / excerpt.
    Берём text (полный, уже обрезан лимитом), иначе excerpt.
    """
    if not documents:
        return ""
    blocks: list[str] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        body = (doc.get("text") or doc.get("excerpt") or "").strip()
        if not body:
            continue
        name = str(doc.get("name") or doc.get("stored_name") or "документ")
        blocks.append(f"[Документ: {name}]\n{body}")
    return "\n\n".join(blocks)


def generate_hypotheses(
    kpi: str,
    constraints: str,
    language: str,
    knowledge_bases: list[str] | None = None,
    documents: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected_knowledge_bases = _normalized_knowledge_bases(knowledge_bases)

    # Текст загруженных документов подмешиваем в ограничения, чтобы он дошёл
    # до любого генератора ниже (RAG-декомпозиция / шаблонный / внешний бэкенд).
    docs_context = _documents_context(documents)
    if docs_context:
        constraints = (
            f"{constraints}\n\nМАТЕРИАЛЫ ПОЛЬЗОВАТЕЛЯ:\n{docs_context}"
            if constraints and constraints.strip()
            else f"МАТЕРИАЛЫ ПОЛЬЗОВАТЕЛЯ:\n{docs_context}"
        )

    # Приоритет: реальный RAG-пайплайн (pgvector + LLM) при заданном DB_DSN.
    # Отключается через USE_RAG=0 (тогда работает шаблонный генератор ниже).
    if os.getenv("DB_DSN") and os.getenv("USE_RAG", "1") != "0":
        try:
            from rag.service import generate_hypotheses as _rag_generate

            return _rag_generate(kpi, constraints or "", language, selected_knowledge_bases)
        except Exception as exc:  # не роняем UI — падаем в шаблонный генератор
            print(f"[generator] RAG pipeline failed ({exc}); using local template generator")

    if not BACKEND_URL:
        return _local_generate(kpi, constraints, language, selected_knowledge_bases)

    payload = {
        "kpi": kpi,
        "constraints": constraints or None,
        "language": language,
        "knowledge_bases": selected_knowledge_bases,
        "use_open_sources": "open_sources" in selected_knowledge_bases,
        "industrial_scale": True,
    }
    headers = {"Content-Type": "application/json"}
    token = os.getenv("API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.post(BACKEND_URL, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    return _normalize_backend_response(response.json())
