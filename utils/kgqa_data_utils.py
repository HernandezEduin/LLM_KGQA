"""Shared KGQA data normalization and JSON serialization helpers."""

import ast
from collections import defaultdict
from numbers import Number
from typing import Any, Set

from utils.kgqa_types import EntityId, PathList, RelationChain


def normalize_reference_paths(value: object) -> PathList:
    """Normalize a dataset Paths cell into a list of candidate triplet paths."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        value = ast.literal_eval(stripped)
    if not isinstance(value, (list, tuple)):
        return []

    def is_triplet(item: object) -> bool:
        return (
            isinstance(item, (list, tuple))
            and len(item) == 3
            and not any(isinstance(part, (list, tuple, dict, set)) for part in item)
        )

    if all(is_triplet(edge) for edge in value):
        return [[tuple(edge) for edge in value]]

    candidates = []
    for path in value:
        if isinstance(path, (list, tuple)) and all(is_triplet(edge) for edge in path):
            candidates.append([tuple(edge) for edge in path])
    return candidates


def normalize_relation_chain(value: object) -> RelationChain | None:
    """Normalize a Path-Key cell into a relation sequence."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith('['):
            value = ast.literal_eval(stripped)
        else:
            value = stripped.split('->')
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def normalize_answer_entities(value: object) -> Set[EntityId]:
    """Normalize Answer-Entity into a set without changing entity ID types."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return set()
        if stripped.startswith('['):
            value = ast.literal_eval(stripped)
        else:
            return {value}
    if isinstance(value, (list, tuple, set)):
        return set(value)
    return {value} if value is not None else set()


def to_jsonable(value: Any) -> Any:
    """Convert nested experiment payload values into JSON-serializable objects."""
    if isinstance(value, defaultdict):
        value = dict(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Number):
        if hasattr(value, 'item'):
            return value.item()
        return value
    if value is None or isinstance(value, (str, bool)):
        return value
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


def get_row_value(row: object, key: str, default: Any = None) -> Any:
    """Read a row value if the key exists, otherwise return the provided default."""
    return row[key] if key in row else default
