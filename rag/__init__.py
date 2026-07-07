"""RAG-ядро «Фабрики гипотез»: декомпозиция → поиск → синтез топ-3 → скоринг."""

from .pipeline import FabrikaPipeline
from .schemas import (
    CausalLink,
    Hypothesis,
    HypothesisBatch,
    HypothesisDraft,
    IndustrialConstraints,
    PipelineResult,
)

__all__ = [
    "FabrikaPipeline",
    "IndustrialConstraints",
    "CausalLink",
    "HypothesisDraft",
    "HypothesisBatch",
    "Hypothesis",
    "PipelineResult",
]
