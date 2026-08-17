"""Path-fidelity and answer metrics for iterative KGQA navigation."""

from __future__ import annotations

from collections import Counter
from numbers import Number
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from utils.kgqa_types import (
    AggregateMetricScores,
    EntityId,
    MetricScores,
    MetricValue,
    Path,
    PathList,
    RelationChain,
    RelationId,
    Triplet,
)


DEFAULT_CONTROL_RELATIONS: Set[RelationId] = set()


def _normalize_path(path: Iterable[Sequence[Any]] | None) -> Path | None:
    """
    Convert a raw path-like object into a list of KGQA triplets.

    Args:
        path (Iterable[Sequence[Any]] | None): Path edges from data, model output, or JSON.

    Returns:
        Path | None: Normalized triplet path, or None when no path is supplied.

    Raises:
        ValueError: If any edge is not a three-item sequence.
    """
    if path is None:
        return None
    normalized: Path = []
    for edge in path:
        if not isinstance(edge, (list, tuple)) or len(edge) != 3:
            raise ValueError(f"Every path edge must be a triplet; received {edge!r}.")
        normalized.append((edge[0], edge[1], edge[2]))
    return normalized


def _safe_ratio(numerator: int, denominator: int) -> float:
    """
    Divide two counts while defining useful empty-set behavior.

    Args:
        numerator (int): Numerator count.
        denominator (int): Denominator count.

    Returns:
        float: Ratio, 1.0 for 0/0, and 0.0 for nonzero/0.
    """
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def _prefix_length(left: Sequence[Any], right: Sequence[Any]) -> int:
    """
    Count matching items from the beginning of two sequences.

    Args:
        left (Sequence[Any]): First sequence.
        right (Sequence[Any]): Second sequence.

    Returns:
        int: Length of the common prefix.
    """
    length = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        length += 1
    return length


def levenshtein_edit_distance(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    """
    Return insertion/deletion/substitution edit distance.

    Args:
        seq_a (Sequence[Any]): First sequence.
        seq_b (Sequence[Any]): Second sequence.

    Returns:
        int: Levenshtein edit distance between the sequences.
    """
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
    predicted: Set[Any],
    reference: Set[Any],
) -> Tuple[float, float, float]:
    """
    Compute set precision, recall, and F1.

    Args:
        predicted (Set[Any]): Predicted items.
        reference (Set[Any]): Reference items.

    Returns:
        Tuple[float, float, float]: Precision, recall, and F1.
    """
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
    control_relations: Set[RelationId] | None = None,
) -> Path:
    """
    Remove STOP/RESTART-style edges identified by relation ID.

    Args:
        path (Iterable[Sequence[Any]]): Path to filter.
        control_relations (Set[RelationId] | None): Relation IDs to ignore.

    Returns:
        Path: Path with control-relation edges removed.
    """
    controls = DEFAULT_CONTROL_RELATIONS if control_relations is None else control_relations
    normalized = _normalize_path(path)
    return [edge for edge in normalized if edge[1] not in controls]


def semantic_edge_sequence(path: Iterable[Sequence[Any]]) -> Path:
    """
    Normalize a path into its semantic KG edge sequence.

    Args:
        path (Iterable[Sequence[Any]]): Raw path edges.

    Returns:
        Path: Normalized KGQA triplet path.
    """
    return list(_normalize_path(path))


def relation_sequence(path: Iterable[Sequence[Any]]) -> RelationChain:
    """
    Extract the relation IDs from a path.

    Args:
        path (Iterable[Sequence[Any]]): Raw or normalized triplet path.

    Returns:
        RelationChain: Relation sequence for the path.
    """
    return [edge[1] for edge in _normalize_path(path)]


def score_path_fidelity(
    predicted_path: Iterable[Sequence[Any]],
    reference_path: Iterable[Sequence[Any]] | None = None,
    reference_relation_chain: Sequence[RelationId] | None = None,
    control_relations: Set[RelationId] | None = None,
) -> MetricScores:
    """
    Score one directed trajectory against a reference path or relation chain.

    Public benchmark keys are PED, RED, F1_SG, and F1_REL. Additional exact,
    prefix, and multiset-triplet diagnostics are retained for compatibility.

    Args:
        predicted_path (Iterable[Sequence[Any]]): Predicted navigation path.
        reference_path (Iterable[Sequence[Any]] | None): Gold triplet path, if available.
        reference_relation_chain (Sequence[RelationId] | None): Gold relation chain, if available.
        control_relations (Set[RelationId] | None): Relation IDs to ignore during scoring.

    Returns:
        MetricScores: Path-fidelity metric names mapped to scores or None.
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

    result: MetricScores = {
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
    reference_paths: PathList | None = None,
    reference_relation_chain: Sequence[RelationId] | None = None,
    control_relations: Set[RelationId] | None = None,
) -> MetricScores:
    """
    Score one prediction against multiple references using oracle aggregation rules.

    Args:
        predicted_path (Iterable[Sequence[Any]]): Predicted navigation path.
        reference_paths (PathList | None): Candidate gold triplet paths.
        reference_relation_chain (Sequence[RelationId] | None): Gold relation chain, if available.
        control_relations (Set[RelationId] | None): Relation IDs to ignore during scoring.

    Returns:
        MetricScores: Combined path-fidelity scores using min PED and max F1_SG over references.
    """
    if not reference_paths:
        return score_path_fidelity(
            predicted_path,
            reference_relation_chain=reference_relation_chain,
            control_relations=control_relations,
        )

    per_reference: List[MetricScores] = [
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

    combined: MetricScores = dict(best_diagnostic)
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
    predicted_answers: Iterable[EntityId],
    valid_answers: Iterable[EntityId],
) -> MetricScores:
    """
    Score a predicted answer set against valid KGQA answer entities.

    Args:
        predicted_answers (Iterable[EntityId]): Predicted answer entity IDs.
        valid_answers (Iterable[EntityId]): Valid gold answer entity IDs.

    Returns:
        MetricScores: Hits@1, MRR placeholder, and final-entity correctness.
    """
    hit = float(bool(set(predicted_answers) & set(valid_answers)))
    return {"Hits1": hit, "MRR": None, "final_entity_correct": hit}


def score_single_final_entity(
    final_entity: EntityId,
    valid_answers: Iterable[EntityId],
) -> MetricScores:
    """
    Score one terminal navigation entity as the predicted answer.

    Args:
        final_entity (EntityId): Terminal entity selected by navigation.
        valid_answers (Iterable[EntityId]): Valid gold answer entity IDs.

    Returns:
        MetricScores: Answer metrics for the single final entity.
    """
    return score_answer_set({final_entity}, valid_answers)


def mean_available(values: Iterable[MetricValue]) -> MetricValue:
    """
    Average numeric metric values while ignoring unavailable scores.

    Args:
        values (Iterable[MetricValue]): Scores that may include None.

    Returns:
        MetricValue: Mean numeric score, or None if no numeric values are present.
    """
    available = [float(value) for value in values if isinstance(value, Number)]
    return sum(available) / len(available) if available else None


def aggregate_sampled_trajectory_metrics(
    question_results: Sequence[Sequence[Mapping[str, MetricValue]]],
) -> AggregateMetricScores:
    """
    Average trajectory metrics over samples and then questions.

    Args:
        question_results (Sequence[Sequence[Mapping[str, MetricValue]]]): Per-question trajectory scores.

    Returns:
        AggregateMetricScores: Aggregate metric means plus per-metric support counts.
    """
    metric_names = sorted({
        key
        for samples in question_results
        for score in samples
        for key in score
    })
    result: AggregateMetricScores = {"count": len(question_results)}
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
) -> AggregateMetricScores:
    """
    Average metrics when each question has one predicted trajectory.

    Args:
        scores (Sequence[Mapping[str, MetricValue]]): One metric record per question.

    Returns:
        AggregateMetricScores: Aggregate trajectory metrics.
    """
    return aggregate_sampled_trajectory_metrics([[score] for score in scores])


def aggregate_answer_metrics(
    scores: Sequence[Mapping[str, MetricValue]],
) -> AggregateMetricScores:
    """
    Aggregate final-answer metrics over questions.

    Args:
        scores (Sequence[Mapping[str, MetricValue]]): One answer metric record per question.

    Returns:
        AggregateMetricScores: Count, scored count, correct count, accuracy, Hits@1, and MRR.
    """
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
    control_relations: Set[RelationId] | None = None,
) -> Counter[Triplet]:
    """
    Count non-control edges in a path.

    Args:
        path (Iterable[Sequence[Any]]): Raw or normalized triplet path.
        control_relations (Set[RelationId] | None): Relation IDs to ignore during counting.

    Returns:
        Counter[Triplet]: Multiplicity count for each non-control triplet.
    """
    return Counter(remove_control_edges(path, control_relations))


def path_relation_counter(
    path: Iterable[Sequence[Any]],
    control_relations: Set[RelationId] | None = None,
) -> Counter[RelationId]:
    """
    Count non-control relations in a path.

    Args:
        path (Iterable[Sequence[Any]]): Raw or normalized triplet path.
        control_relations (Set[RelationId] | None): Relation IDs to ignore during counting.

    Returns:
        Counter[RelationId]: Multiplicity count for each non-control relation.
    """
    return Counter(edge[1] for edge in remove_control_edges(path, control_relations))
