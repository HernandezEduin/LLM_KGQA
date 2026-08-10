"""Answer and path-fidelity metrics for iterative graph navigation."""

from __future__ import annotations

from collections import Counter
from numbers import Number
from typing import Any, Iterable, Mapping, Sequence

Triplet = tuple[Any, Any, Any]
MetricValue = float | None
DEFAULT_CONTROL_RELATIONS: set[Any] = set()


def _normalize_path(path: Iterable[Sequence[Any]] | None) -> list[Triplet] | None:
    if path is None:
        return None
    normalized = []
    for edge in path:
        if not isinstance(edge, (list, tuple)) or len(edge) != 3:
            raise ValueError(f"Every path edge must be a triplet; received {edge!r}.")
        normalized.append(tuple(edge))
    return normalized


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def _prefix_length(left: Sequence[Any], right: Sequence[Any]) -> int:
    length = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        length += 1
    return length


def levenshtein_edit_distance(seq_a: Sequence[object], seq_b: Sequence[object]) -> int:
    """Return insertion/deletion/substitution edit distance."""
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    previous = list(range(len(seq_b) + 1))
    for row, item_a in enumerate(seq_a, start=1):
        current = [row] + [0] * len(seq_b)
        for column, item_b in enumerate(seq_b, start=1):
            substitution = previous[column - 1] + (item_a != item_b)
            deletion = previous[column] + 1
            insertion = current[column - 1] + 1
            current[column] = min(substitution, deletion, insertion)
        previous = current
    return previous[-1]


def precision_recall_f1(
    predicted: set[Any],
    reference: set[Any],
) -> tuple[float, float, float]:
    """Return set precision, recall, and F1."""
    if not predicted and not reference:
        return 1.0, 1.0, 1.0
    if not predicted or not reference:
        return 0.0, 0.0, 0.0
    true_positive = len(predicted & reference)
    precision = true_positive / len(predicted)
    recall = true_positive / len(reference)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def remove_control_edges(
    path: Iterable[Sequence[Any]],
    control_relations: set[Any] | None = None,
) -> list[Triplet]:
    """Remove STOP/RESTART-style edges identified by relation ID."""
    controls = DEFAULT_CONTROL_RELATIONS if control_relations is None else control_relations
    normalized = _normalize_path(path)
    return [edge for edge in normalized if edge[1] not in controls]


def semantic_edge_sequence(path: Iterable[Sequence[Any]]) -> list[Triplet]:
    return list(_normalize_path(path))


def relation_sequence(path: Iterable[Sequence[Any]]) -> list[Any]:
    return [edge[1] for edge in _normalize_path(path)]


def score_path_fidelity(
    predicted_path: Iterable[Sequence[Any]],
    reference_path: Iterable[Sequence[Any]] | None = None,
    reference_relation_chain: Sequence[Any] | None = None,
    control_relations: set[Any] | None = None,
) -> dict[str, MetricValue]:
    """Score one directed trajectory against a path or relation reference.

    Public benchmark keys are PED, RED, F1_SG, and F1_REL. Additional exact,
    prefix, and multiset-triplet diagnostics are retained for compatibility.
    """
    predicted = remove_control_edges(predicted_path, control_relations)
    reference = (
        remove_control_edges(reference_path, control_relations)
        if reference_path is not None
        else None
    )
    predicted_relations = relation_sequence(predicted)

    if reference is not None:
        reference_relations = relation_sequence(reference)
    elif reference_relation_chain is not None:
        reference_relations = list(reference_relation_chain)
    else:
        reference_relations = None

    relation_prefix = (
        _prefix_length(predicted_relations, reference_relations)
        if reference_relations is not None
        else 0
    )
    red = (
        float(levenshtein_edit_distance(predicted_relations, reference_relations))
        if reference_relations is not None
        else None
    )
    f1_rel = (
        precision_recall_f1(set(predicted_relations), set(reference_relations))[2]
        if reference_relations is not None
        else None
    )

    result: dict[str, MetricValue] = {
        "PED": None,
        "RED": red,
        "F1_SG": None,
        "F1_REL": f1_rel,
        "path_exact_match": None,
        "relation_chain_exact_match": (
            float(predicted_relations == reference_relations)
            if reference_relations is not None
            else None
        ),
        "relation_prefix_recall": (
            _safe_ratio(relation_prefix, len(reference_relations))
            if reference_relations is not None
            else None
        ),
        "predicted_path_length": float(len(predicted)),
        "reference_path_length": (
            float(len(reference))
            if reference is not None
            else float(len(reference_relations)) if reference_relations is not None else None
        ),
        "triplet_precision": None,
        "triplet_recall": None,
        "triplet_f1": None,
        "triplet_prefix_recall": None,
    }

    if reference is None:
        return result

    result["PED"] = float(levenshtein_edit_distance(predicted, reference))
    result["F1_SG"] = precision_recall_f1(set(predicted), set(reference))[2]
    result["path_exact_match"] = float(predicted == reference)

    predicted_counts = Counter(predicted)
    reference_counts = Counter(reference)
    overlap = sum((predicted_counts & reference_counts).values())
    precision = _safe_ratio(overlap, len(predicted))
    recall = _safe_ratio(overlap, len(reference))
    result["triplet_precision"] = precision
    result["triplet_recall"] = recall
    result["triplet_f1"] = (
        0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )
    result["triplet_prefix_recall"] = _safe_ratio(
        _prefix_length(predicted, reference), len(reference)
    )
    return result


def score_path_fidelity_against_references(
    predicted_path: Iterable[Sequence[Any]],
    reference_paths: Sequence[Iterable[Sequence[Any]]] | None = None,
    reference_relation_chain: Sequence[Any] | None = None,
    control_relations: set[Any] | None = None,
) -> dict[str, MetricValue]:
    """Score multiple references using min PED and max F1_SG oracle rules."""
    if not reference_paths:
        return score_path_fidelity(
            predicted_path,
            reference_relation_chain=reference_relation_chain,
            control_relations=control_relations,
        )

    per_reference = [
        score_path_fidelity(
            predicted_path,
            reference_path=reference,
            control_relations=control_relations,
        )
        for reference in reference_paths
    ]
    best_diagnostic = max(
        per_reference,
        key=lambda score: (
            score["path_exact_match"],
            score["relation_chain_exact_match"],
            score["triplet_f1"],
            score["triplet_prefix_recall"],
        ),
    )
    relation_reference = reference_relation_chain
    if relation_reference is None:
        relation_reference = relation_sequence(
            remove_control_edges(reference_paths[0], control_relations)
        )
    relation_scores = score_path_fidelity(
        predicted_path,
        reference_relation_chain=relation_reference,
        control_relations=control_relations,
    )

    combined = dict(best_diagnostic)
    combined.update({
        "PED": min(score["PED"] for score in per_reference if score["PED"] is not None),
        "RED": relation_scores["RED"],
        "F1_SG": max(
            score["F1_SG"] for score in per_reference if score["F1_SG"] is not None
        ),
        "F1_REL": relation_scores["F1_REL"],
        "relation_chain_exact_match": relation_scores["relation_chain_exact_match"],
        "relation_prefix_recall": relation_scores["relation_prefix_recall"],
    })
    return combined


def score_answer_set(
    predicted_answers: Iterable[Any],
    valid_answers: Iterable[Any],
) -> dict[str, MetricValue]:
    """Score a one-shot final answer set using Hits@1; MRR is unavailable."""
    hit = float(bool(set(predicted_answers) & set(valid_answers)))
    return {"Hits1": hit, "MRR": None, "final_entity_correct": hit}


def score_single_final_entity(
    final_entity: Any,
    valid_answers: Iterable[Any],
) -> dict[str, MetricValue]:
    return score_answer_set({final_entity}, valid_answers)


def mean_available(values: Iterable[MetricValue]) -> MetricValue:
    available = [float(value) for value in values if isinstance(value, Number)]
    return sum(available) / len(available) if available else None


def aggregate_sampled_trajectory_metrics(
    question_results: Sequence[Sequence[Mapping[str, MetricValue]]],
) -> dict[str, MetricValue | int]:
    """Compute mean(question mean(trajectory metric)) for every metric key."""
    metric_names = sorted({
        key
        for samples in question_results
        for score in samples
        for key in score
    })
    result: dict[str, MetricValue | int] = {"count": len(question_results)}
    for metric_name in metric_names:
        per_question = [
            mean_available(score.get(metric_name) for score in samples)
            for samples in question_results
        ]
        result[metric_name] = mean_available(per_question)
        result[f"{metric_name}_support"] = sum(value is not None for value in per_question)
    return result


def aggregate_single_prediction_metrics(
    scores: Sequence[Mapping[str, MetricValue]],
) -> dict[str, MetricValue | int]:
    """Average one predicted trajectory per question."""
    return aggregate_sampled_trajectory_metrics([[score] for score in scores])


def aggregate_answer_metrics(
    scores: Sequence[Mapping[str, MetricValue]],
) -> dict[str, MetricValue | int]:
    hits1 = mean_available(score.get("Hits1") for score in scores)
    mrr = mean_available(score.get("MRR") for score in scores)
    correct = sum(float(score.get("Hits1") or 0.0) for score in scores)
    return {
        "count": len(scores),
        "scored": len(scores),
        "correct": int(correct),
        "accuracy": hits1 or 0.0,
        "Hits1": hits1 or 0.0,
        "MRR": mrr,
    }


def path_edge_counter(
    path: Iterable[Sequence[Any]],
    control_relations: set[Any] | None = None,
) -> Counter:
    return Counter(remove_control_edges(path, control_relations))


def path_relation_counter(
    path: Iterable[Sequence[Any]],
    control_relations: set[Any] | None = None,
) -> Counter:
    return Counter(edge[1] for edge in remove_control_edges(path, control_relations))
