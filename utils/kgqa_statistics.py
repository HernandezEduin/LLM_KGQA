"""Shared KGQA statistics helpers for navigation and subgraph experiments."""

from collections import defaultdict
from typing import Any, List

from utils.kgqa_types import Statistics, StatusInfo


def average(values: List[Any]) -> float:
    """Return the arithmetic mean of a list, or 0 for an empty list."""
    return sum(values) / len(values) if values else 0


def avg_dict(vals: Statistics) -> Statistics:
    """Average list-valued statistics while preserving scalar values."""
    out = {}
    for key, value in vals.items():
        if isinstance(value, list):
            out[key] = average(value)
        else:
            out[key] = value
    return out


def initialize_subgraph_statistics(total: int) -> Statistics:
    """Create a statistics record for subgraph KGQA experiments."""
    return {
        'accuracy': 0,
        'running_count': 0,
        'total': total,
        'subgraph_sizes': defaultdict(int),
        'prompt_tokens': [],
        'response_tokens': [],
        'total_tokens': [],
        'response_seconds': [],
        'prompt_seconds': [],
        'total_seconds': [],
        'prompt_tps': [],
        'completion_tps': [],
        'unknown': 0,
        'timeouts': 0,
        'errors': 0,
    }


def update_subgraph_stats(
    stats_dict: Statistics,
    status_info: StatusInfo,
    result: bool,
    full_pred: str,
    sub_graph_size: int,
) -> None:
    """Update aggregate statistics for one subgraph QA prediction."""
    stats_dict['accuracy'] += int(result)
    stats_dict['running_count'] += 1
    stats_dict['subgraph_sizes'][sub_graph_size] += 1

    if 'prompt_tokens' in status_info:
        stats_dict['prompt_tokens'].append(status_info['prompt_tokens'])
    if 'response_tokens' in status_info:
        stats_dict['response_tokens'].append(status_info['response_tokens'])
    if 'total_tokens' in status_info:
        stats_dict['total_tokens'].append(status_info['total_tokens'])

    if 'response_seconds' in status_info:
        stats_dict['response_seconds'].append(status_info['response_seconds'])
    if 'prompt_seconds' in status_info:
        stats_dict['prompt_seconds'].append(status_info['prompt_seconds'])
    if 'total_seconds' in status_info:
        stats_dict['total_seconds'].append(status_info['total_seconds'])

    if 'prompt_tps' in status_info:
        stats_dict['prompt_tps'].append(status_info['prompt_tps'])
    if 'completion_tps' in status_info:
        stats_dict['completion_tps'].append(status_info['completion_tps'])

    stats_dict['unknown'] += int(full_pred == 'UNKNOWN')
    stats_dict['timeouts'] += int(full_pred == 'TIMEOUT')
    stats_dict['errors'] += int(full_pred == 'ERROR')


def initialize_navigation_statistics(total: int) -> Statistics:
    """Create a statistics record for iterative navigation KGQA experiments."""
    return {
        'accuracy': 0,
        'running_count': 0,
        'total': total,
        'navigation_steps': defaultdict(int),
        'termination_reasons': defaultdict(int),
        'prompt_tokens': [],
        'response_tokens': [],
        'completion_tokens': [],
        'total_tokens': [],
        'response_seconds': [],
        'prompt_seconds': [],
        'total_seconds': [],
        'prompt_tps': [],
        'completion_tps': [],
        'logical_decisions': [],
        'actual_llm_calls': [],
        'executed_graph_edges': [],
        'unknown': 0,
        'timeouts': 0,
        'errors': 0,
        'max_actions_exceeded': 0,
        'max_actions_truncated': 0,
    }


def update_navigation_stats(
    stats_dict: Statistics,
    status_info: StatusInfo,
    correct: bool,
    prediction: str,
    navigation_steps: int,
) -> None:
    """Update aggregate statistics for one navigation QA episode."""
    stats_dict['accuracy'] += int(correct)
    stats_dict['running_count'] += 1
    stats_dict['navigation_steps'][navigation_steps] += 1
    stats_dict['termination_reasons'][status_info.get('termination_reason', 'unknown')] += 1

    if 'prompt_tokens' in status_info:
        stats_dict['prompt_tokens'].append(status_info['prompt_tokens'])
    if 'response_tokens' in status_info:
        stats_dict['response_tokens'].append(status_info['response_tokens'])
        stats_dict['completion_tokens'].append(status_info['response_tokens'])
    if 'completion_tokens' in status_info:
        stats_dict['completion_tokens'].append(status_info['completion_tokens'])
    if 'total_tokens' in status_info:
        stats_dict['total_tokens'].append(status_info['total_tokens'])

    if 'response_seconds' in status_info:
        stats_dict['response_seconds'].append(status_info['response_seconds'])
    if 'prompt_seconds' in status_info:
        stats_dict['prompt_seconds'].append(status_info['prompt_seconds'])
    if 'total_seconds' in status_info:
        stats_dict['total_seconds'].append(status_info['total_seconds'])

    if 'prompt_tps' in status_info:
        stats_dict['prompt_tps'].append(status_info['prompt_tps'])
    if 'completion_tps' in status_info:
        stats_dict['completion_tps'].append(status_info['completion_tps'])

    stats_dict['logical_decisions'].append(status_info.get('logical_decision_count', 0))
    stats_dict['actual_llm_calls'].append(status_info.get('actual_llm_calls', 0))
    stats_dict['executed_graph_edges'].append(status_info.get('executed_graph_edges', navigation_steps))
    stats_dict['unknown'] += int(prediction == 'UNKNOWN')
    stats_dict['timeouts'] += int(prediction == 'TIMEOUT')
    stats_dict['errors'] += int(prediction == 'ERROR')
    stats_dict['max_actions_exceeded'] += int(bool(status_info.get('max_actions_exceeded')))
    stats_dict['max_actions_truncated'] += int(bool(status_info.get('max_actions_truncated')))
