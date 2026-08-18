"""Navigation-specific KGQA helpers shared outside the run script."""

from random import Random
from typing import Any

from utils.graph_utils import generate_multi_answer_paths_from_source
from utils.kgqa_data_utils import (
    get_row_value,
    normalize_answer_entities,
    normalize_reference_paths,
    normalize_relation_chain,
)
from utils.kgqa_navigation_metrics import score_path_fidelity_against_references
from utils.kgqa_types import (
    EntityId,
    MetricScores,
    NavigationDemonstrationList,
    OutgoingIndex,
    PathList,
    RelationChain,
    RelationIndex,
    StatusInfo,
    TripletList,
    TripletSet,
)


def best_path_fidelity_score(
    predicted_path: TripletList,
    reference_paths: PathList,
    relation_chain: RelationChain | None,
) -> MetricScores | None:
    """Apply the benchmark multi-reference and relation-only scoring rules."""
    if not reference_paths and relation_chain is None:
        return None
    return score_path_fidelity_against_references(
        predicted_path=predicted_path,
        reference_paths=reference_paths or None,
        reference_relation_chain=relation_chain,
    )


def validate_executed_path(
    predicted_path: TripletList,
    start_entity: EntityId,
    final_entity: EntityId | None,
    all_triplets: TripletSet,
) -> StatusInfo:
    """Check that an executed navigation path is directed and present in the KG."""
    current = start_entity
    violations = []
    for edge_index, edge in enumerate(predicted_path):
        triplet = tuple(edge)
        if triplet not in all_triplets:
            violations.append({
                'edge_index': edge_index,
                'type': 'missing_from_kg',
                'triplet': triplet,
            })
        if triplet[0] != current:
            violations.append({
                'edge_index': edge_index,
                'type': 'current_entity_mismatch',
                'expected_head': current,
                'triplet': triplet,
            })
        current = triplet[2]

    if final_entity is not None and current != final_entity:
        violations.append({
            'type': 'final_entity_mismatch',
            'path_terminal_entity': current,
            'recorded_final_entity': final_entity,
        })

    return {
        'valid': not violations,
        'final_entity_from_path': current,
        'violations': violations,
    }


def get_gold_candidate_paths(
    row: object,
    relation_index: RelationIndex,
) -> PathList:
    """Return complete candidate gold paths for one QA row."""
    paths: PathList = []
    if 'Paths' in row and row['Paths'] != '':
        paths = normalize_reference_paths(row['Paths'])
    elif 'Path-Key' in row and row['Path-Key'] != '':
        relation_chain = normalize_relation_chain(row['Path-Key'])
        if relation_chain is not None:
            paths = generate_multi_answer_paths_from_source(
                source_entity=row['Source-Entity'],
                rel_list=relation_chain,
                relation_index=relation_index,
            )

    answer_entities = normalize_answer_entities(row['Answer-Entity']) if 'Answer-Entity' in row else set()
    if answer_entities:
        paths = [path for path in paths if path and path[-1][2] in answer_entities]
    return paths


def number_to_shot_label(n_shots: int) -> str:
    """Return the filename/result label for a non-negative shot count."""
    if n_shots < 0:
        raise ValueError('n_shots must be non-negative.')

    ones = {
        0: 'zero',
        1: 'one',
        2: 'two',
        3: 'three',
        4: 'four',
        5: 'five',
        6: 'six',
        7: 'seven',
        8: 'eight',
        9: 'nine',
        10: 'ten',
        11: 'eleven',
        12: 'twelve',
        13: 'thirteen',
        14: 'fourteen',
        15: 'fifteen',
        16: 'sixteen',
        17: 'seventeen',
        18: 'eighteen',
        19: 'nineteen',
    }
    tens = {
        20: 'twenty',
        30: 'thirty',
        40: 'forty',
        50: 'fifty',
        60: 'sixty',
        70: 'seventy',
        80: 'eighty',
        90: 'ninety',
    }

    if n_shots in ones:
        shot_count = ones[n_shots]
    elif n_shots < 100:
        base = n_shots // 10 * 10
        remainder = n_shots % 10
        shot_count = tens[base] if remainder == 0 else f"{tens[base]}-{ones[remainder]}"
    else:
        shot_count = str(n_shots)
    return f'{shot_count}-shot'


def get_navigation_demonstration_hop_hint(row: object) -> int:
    """Estimate trajectory length so n-shot demos prefer examples with history."""
    hop_value = get_row_value(row, 'Hops')
    try:
        if hop_value != '':
            return int(hop_value)
    except (TypeError, ValueError):
        pass

    if 'Path-Key' in row and row['Path-Key'] != '':
        relation_chain = normalize_relation_chain(row['Path-Key'])
        if relation_chain is not None:
            return len(relation_chain)

    if 'Paths' in row and row['Paths'] != '':
        paths = normalize_reference_paths(row['Paths'])
        if paths:
            return max(len(path) for path in paths)

    return 0


def is_executable_gold_path(
    start_node: EntityId,
    path: TripletList,
    outgoing_index: OutgoingIndex,
) -> bool:
    """Check that every gold edge is a legal directed action from the current entity."""
    current_entity = start_node
    for triplet in path:
        triplet = tuple(triplet)
        actions = sorted(
            outgoing_index.get(current_entity, []),
            key=lambda action: (action[1], action[2]),
        )
        if triplet[0] != current_entity or triplet not in actions:
            return False
        current_entity = triplet[2]
    return bool(path)


def sample_navigation_demonstrations(
    train_df: Any,
    outgoing_index: OutgoingIndex,
    relation_index: RelationIndex,
    n_shots: int,
    seed: int,
) -> NavigationDemonstrationList:
    """Sample executable complete train trajectories for n-shot navigation prompts."""
    if n_shots == 0:
        return []

    rng = Random(seed)
    row_positions = list(range(len(train_df)))
    rng.shuffle(row_positions)
    row_positions.sort(
        key=lambda row_position: get_navigation_demonstration_hop_hint(
            train_df.iloc[row_position]
        ),
        reverse=True,
    )
    demonstrations: NavigationDemonstrationList = []
    skipped_rows = 0

    for row_position in row_positions:
        row = train_df.iloc[row_position]
        start_node = row['Source-Entity']
        executable_paths: PathList = []
        for candidate_path in get_gold_candidate_paths(row, relation_index):
            path = [tuple(triplet) for triplet in candidate_path]
            if is_executable_gold_path(start_node, path, outgoing_index):
                executable_paths.append(path)

        if not executable_paths:
            skipped_rows += 1
            continue
        selected_path = max(executable_paths, key=len)

        demonstrations.append({
            'question_index': get_row_value(row, 'Question-Number', row_position),
            'row_position': row_position,
            'hop_hint': get_navigation_demonstration_hop_hint(row),
            'question': row['Question'],
            'start_node': start_node,
            'answer_entities': sorted(normalize_answer_entities(row['Answer-Entity'])) if 'Answer-Entity' in row else [],
            'path': selected_path,
            'path_length': len(selected_path),
        })
        if len(demonstrations) == n_shots:
            break

    if len(demonstrations) < n_shots:
        raise ValueError(
            f"Requested {n_shots} n-shot navigation demonstrations, but only found "
            f"{len(demonstrations)} executable complete train trajectories "
            f"({skipped_rows} sampled train rows skipped)."
        )

    return demonstrations
