"""Symbolic, correct-by-construction word-problem generator."""

from .dag import DAG, Node, Op
from .families import FAMILIES, Family, families_for_split
from .features import FEATURES, compute_features, count_borrows, count_carries, feature_vector
from .sampler import GenerationError, GenSpec, Item, generate, generate_many

__all__ = [
    "DAG", "Node", "Op", "FAMILIES", "Family", "families_for_split", "FEATURES",
    "compute_features", "count_borrows", "count_carries", "feature_vector",
    "GenerationError", "GenSpec", "Item", "generate", "generate_many",
]
