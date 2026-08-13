"""
Run iterative LLM-based knowledge-graph navigation experiments for KGQA.

The controller owns the symbolic graph. At each navigation state it exposes only
legal outgoing one-hop actions from the current entity, executes the selected KG
edge, and treats the terminal graph entity as the prediction.
"""

import argparse
import ast
import json
import os
from collections import defaultdict
from numbers import Number
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

from model.LLM_KGQA import LLM_KGQA_Client
from model.constants import valid_models
from utils.basic import extract_literals, load_pandas, load_triplets
from utils.graph_utils import build_outgoing_index
from llm_navigation_metrics import (
    aggregate_answer_metrics,
    aggregate_single_prediction_metrics,
    score_path_fidelity_against_references,
    score_single_final_entity,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative KG navigation for QA datasets")

    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path containing the dataset splits.')
    parser.add_argument('--dataset', type=str, default='mquake',
                        help='Name of the dataset to process.')
    parser.add_argument('--hops', type=str, default='n',
                        help='QA dataset hop split to evaluate.')
    parser.add_argument('--max-questions', type=int, default=None,
                        help='Process only the first N test questions (must be positive).')

    # LLM parameters
    parser.add_argument('--llm-model', type=str, default='gemma3',
                        choices=valid_models,
                        help='Model ID to use for the LLM API.')
    parser.add_argument('--use-instruct', action='store_true',
                        help='Whether to use the instruction-tuned version of the model.')
    parser.add_argument('--use-quantized', action='store_true',
                        help='Whether to use the quantized version of the model.')
    parser.add_argument('--quantization-bits', type=int, default=4,
                        help='Number of bits for quantization (if using quantized model).')
    parser.add_argument('--context-window', type=int, default=4096,
                        help='Context window size for the LLM model.')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Timeout in seconds for LLM API requests.')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Sampling temperature for the LLM (0 = deterministic).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for model inference.')

    # Navigation parameters
    parser.add_argument('--max-actions', type=int, default=None,
                        help=('Optional safety cap for the number of options shown in a single prompt. '
                              'If exceeded, the episode is recorded as max_actions_exceeded; legal '
                              'actions are not truncated or sampled.'))
    parser.add_argument('--max-navigation-steps', type=int, default=4,
                        help='Maximum number of graph edges the model may traverse before termination.')
    parser.add_argument('--navigation-approach', type=str, default='tuple',
                        choices=['tuple', 'factorized', 'hybrid'],
                        help='Tuple, factorized relation/entity, or threshold-based hybrid navigation.')
    parser.add_argument('--memory-approach', type=str, default='full',
                        choices=['none', 'full'],
                        help='Observation memory: none hides previous edges; full shows the traversed path.')
    parser.add_argument('--prompting-approach', type=str, default='zero-shot',
                        choices=['io', 'zero-shot', 'one-shot'],
                        help='Only zero-shot navigation is currently implemented.')
    parser.add_argument('--hybrid-threshold', type=int, default=50,
                        help='Use tuple mode when neighborhood size is <= this threshold, else factorized.')
    parser.add_argument('--max-parse-retries', type=int, default=1,
                        help='Retry a navigation decision this many times after malformed JSON output.')

    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug mode with verbose output.')
    parser.add_argument('--show-navigation', '--show-actions', dest='show_navigation', action='store_true',
                        help='Show every navigation prompt, model response, and validated move.')

    # Result parameters
    parser.add_argument('--result-dir', type=str, default='./results',
                        help='Directory to save the results.')

    return parser.parse_args()


def initialize_statistics(total: int) -> Dict:
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
    }


def update_stats(
    stats_dict: Dict,
    status_info: Dict,
    correct: bool,
    prediction: str,
    navigation_steps: int,
) -> None:
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


def average(values):
    return sum(values) / len(values) if values else 0


def avg_dict(vals: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for key, value in vals.items():
        if isinstance(value, list):
            out[key] = average(value)
        else:
            out[key] = value
    return out


def normalize_reference_paths(value):
    """Normalize a dataset Paths cell into a list of candidate triplet paths."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        value = ast.literal_eval(stripped)
    if not isinstance(value, (list, tuple)):
        return []

    def is_triplet(item):
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


def normalize_relation_chain(value):
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


def normalize_answer_entities(value):
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


def best_path_fidelity_score(predicted_path, reference_paths, relation_chain):
    """Apply the benchmark multi-reference and relation-only scoring rules."""
    if not reference_paths and relation_chain is None:
        return None
    return score_path_fidelity_against_references(
        predicted_path=predicted_path,
        reference_paths=reference_paths or None,
        reference_relation_chain=relation_chain,
    )


def validate_executed_path(predicted_path, start_entity, final_entity, all_triplets):
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


def to_jsonable(value: Any):
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


def get_row_value(row, key, default=None):
    return row[key] if key in row else default


if __name__ == '__main__':
    args = parse_args()

    if args.max_navigation_steps < 0:
        raise ValueError('--max-navigation-steps must be non-negative.')
    if args.max_questions is not None and args.max_questions < 1:
        raise ValueError('--max-questions must be positive.')
    if args.max_actions is not None and args.max_actions < 1:
        raise ValueError('--max-actions must be positive when provided.')
    if args.hybrid_threshold < 0:
        raise ValueError('--hybrid-threshold must be non-negative.')
    if args.max_parse_retries < 0:
        raise ValueError('--max-parse-retries must be non-negative.')
    if args.prompting_approach != 'zero-shot':
        raise NotImplementedError(
            f"--prompting-approach {args.prompting_approach!r} is not implemented for "
            'iterative navigation yet. Use --prompting-approach zero-shot.'
        )

    data_dir = os.path.join(args.data_dir, args.dataset)
    qa_file = os.path.join(data_dir, f'qa_{args.hops}hop.csv')
    triplet_file = os.path.join(data_dir, 'triplets.txt')
    entity_file = os.path.join(data_dir, 'node_data.csv')
    relation_file = os.path.join(data_dir, 'relation_data.csv')

    entity_df = load_pandas(entity_file)
    relation_df = load_pandas(relation_file)

    entity_df.set_index('QID', inplace=True)
    relation_df.set_index('Property', inplace=True)

    entity_title = entity_df['Title'].to_dict()
    relation_title = relation_df['Title'].to_dict()

    all_triplets_df = load_triplets(triplet_file)
    all_triplets = set(tuple(triplet) for triplet in all_triplets_df.values)
    outgoing_index = build_outgoing_index(all_triplets)

    qa_df = load_pandas(qa_file)
    qa_df = qa_df[qa_df['SplitLabel'] == 'test'].copy()
    if args.max_questions is not None:
        qa_df = qa_df.head(args.max_questions).copy()

    if not qa_df.empty and qa_df['Answer'].apply(
        lambda value: isinstance(value, str) and value.strip().startswith('[')
    ).all():
        qa_df['Answer'] = extract_literals(qa_df['Answer'])
        qa_df['Answer-Entity'] = extract_literals(qa_df['Answer-Entity'])

    qa_df = qa_df.reset_index(drop=False).rename(columns={'index': 'dataframe_index'})

    config_path = Path(__file__).with_name('openwebui_config.json').parent / 'configs' / 'openwebui_config.json'
    client = LLM_KGQA_Client(
        config_path,
        model_choice=args.llm_model,
        use_instruct=args.use_instruct,
        use_quantized=args.use_quantized,
        quantization_bits=args.quantization_bits,
        context_window=args.context_window,
        seed=args.seed,
        temperature=args.temperature,
        timeout=args.timeout,
        debug=args.debug,
    )

    statistics = {'overall': initialize_statistics(total=len(qa_df))}
    if args.hops == 'n' and 'Hops' in qa_df.columns:
        hop_size_counts = qa_df['Hops'].value_counts().to_dict()
        for hop_size, count in hop_size_counts.items():
            statistics[f'{hop_size}'] = initialize_statistics(total=count)

    navigation_metric_scores = {
        section: {'path': [], 'answer': []}
        for section in statistics
    }
    episodes = []

    with tqdm(range(len(qa_df)), desc='Processing Questions') as pbar:
        for row_pos in pbar:
            row = qa_df.iloc[row_pos]
            question = row['Question']
            start_node = row['Source-Entity']
            hop = get_row_value(row, 'Hops', args.hops)
            question_number = get_row_value(row, 'Question-Number', row_pos)

            pred, navigation_history_txt, status_info = client.process_navigation_question(
                question=question,
                start_node=start_node,
                outgoing_index=outgoing_index,
                entity_title=entity_title,
                relation_title=relation_title,
                max_steps=args.max_navigation_steps,
                max_actions=args.max_actions,
                navigation_approach=args.navigation_approach,
                memory_approach=args.memory_approach,
                prompting_approach=args.prompting_approach,
                hybrid_threshold=args.hybrid_threshold,
                max_parse_retries=args.max_parse_retries,
                trace=pbar.write if args.show_navigation else None,
            )

            predicted_path = status_info.get('predicted_path', [])
            final_entity = status_info.get('final_entity')
            reference_paths = normalize_reference_paths(row['Paths']) if 'Paths' in qa_df.columns else []
            relation_chain = normalize_relation_chain(row['Path-Key']) if 'Path-Key' in qa_df.columns else None
            valid_answer_entities = normalize_answer_entities(row['Answer-Entity'])
            path_score = best_path_fidelity_score(predicted_path, reference_paths, relation_chain)
            answer_entity_score = score_single_final_entity(final_entity, valid_answer_entities) if final_entity is not None else {
                'Hits1': 0.0,
                'MRR': None,
                'final_entity_correct': 0.0,
            }
            correct = bool(answer_entity_score.get('Hits1'))

            metric_sections = ['overall']
            if args.hops == 'n' and f'{hop}' in statistics:
                metric_sections.append(f'{hop}')
            for section in metric_sections:
                if path_score is not None:
                    navigation_metric_scores[section]['path'].append(path_score)
                navigation_metric_scores[section]['answer'].append(answer_entity_score)

            update_stats(
                statistics['overall'],
                status_info,
                correct,
                pred,
                status_info.get('navigation_steps', 0),
            )
            if args.hops == 'n' and f'{hop}' in statistics:
                update_stats(
                    statistics[f'{hop}'],
                    status_info,
                    correct,
                    pred,
                    status_info.get('navigation_steps', 0),
                )

            path_validation = validate_executed_path(
                predicted_path,
                start_node,
                final_entity,
                all_triplets,
            )
            episode = {
                'question_index': question_number,
                'row_position': row_pos,
                'dataframe_index': get_row_value(row, 'dataframe_index'),
                'question': question,
                'dataset': args.dataset,
                'hop_split': args.hops,
                'hops': hop,
                'start_entity': start_node,
                'start_entity_label': entity_title.get(start_node, start_node),
                'gold_answer_entities': sorted(valid_answer_entities),
                'gold_answer_labels': [entity_title.get(entity, entity) for entity in sorted(valid_answer_entities)],
                'gold_answer_text': get_row_value(row, 'Answer'),
                'predicted_terminal_entity': final_entity,
                'predicted_terminal_label': entity_title.get(final_entity, final_entity) if final_entity is not None else None,
                'answer_correct': correct,
                'termination_reason': status_info.get('termination_reason'),
                'navigation_status': status_info.get('status'),
                'status_message': status_info.get('message'),
                'executed_path': predicted_path,
                'readable_executed_path': status_info.get('readable_predicted_path', []),
                'navigation_history_text': navigation_history_txt,
                'executed_graph_edges': status_info.get('executed_graph_edges', status_info.get('navigation_steps', 0)),
                'neighborhood_sizes': status_info.get('neighborhood_sizes', []),
                'selected_actions': status_info.get('selected_actions', []),
                'selected_relation_and_destination': [
                    {
                        'step': record.get('step'),
                        'relation': record.get('selected_relation'),
                        'destination': record.get('selected_destination'),
                    }
                    for record in status_info.get('selected_actions', [])
                ],
                'navigation_approach': status_info.get('navigation_approach', args.navigation_approach),
                'strategy_by_step': status_info.get('strategy_by_step', []),
                'memory_approach': status_info.get('memory_approach', args.memory_approach),
                'prompting_approach': status_info.get('prompting_approach', args.prompting_approach),
                'hybrid_threshold': status_info.get('hybrid_threshold', args.hybrid_threshold),
                'max_actions': status_info.get('max_actions', args.max_actions),
                'max_parse_retries': status_info.get('max_parse_retries', args.max_parse_retries),
                'logical_decisions': status_info.get('logical_decisions', []),
                'logical_decision_count': status_info.get('logical_decision_count', 0),
                'actual_llm_calls': status_info.get('actual_llm_calls', 0),
                'api_retries': status_info.get('api_retries', 0),
                'prompt_tokens': status_info.get('prompt_tokens', 0),
                'completion_tokens': status_info.get('completion_tokens', status_info.get('response_tokens', 0)),
                'response_tokens': status_info.get('response_tokens', 0),
                'total_tokens': status_info.get('total_tokens', 0),
                'prompt_seconds': status_info.get('prompt_seconds'),
                'response_seconds': status_info.get('response_seconds'),
                'total_seconds': status_info.get('total_seconds'),
                'elapsed_time': status_info.get('elapsed_time', 0.0),
                'model_calls': status_info.get('model_calls', []),
                'raw_model_outputs': status_info.get('raw_model_outputs', []),
                'parse_validation_errors': status_info.get('parse_validation_errors', []),
                'path_fidelity': path_score,
                'final_entity_score': answer_entity_score,
                'path_validation': path_validation,
                'graph_directionality': status_info.get('graph_directionality', 'outgoing'),
                'max_actions_exceeded': bool(status_info.get('max_actions_exceeded')),
                'context_window_exceeded': bool(status_info.get('context_window_exceeded')),
                'estimated_prompt_tokens': status_info.get('estimated_prompt_tokens'),
                'context_window': status_info.get('context_window'),
                'context_window_stage': status_info.get('context_window_stage'),
                'context_window_strategy': status_info.get('context_window_strategy'),
            }
            episodes.append(episode)

            if args.debug and not correct:
                pbar.write(f"\nQuestion: {question}")
                pbar.write(f"Gold answer entities: {sorted(valid_answer_entities)}")
                pbar.write(f"Predicted terminal entity: {final_entity}")
                pbar.write(f"Navigation history: {navigation_history_txt}")
                pbar.write(f"Termination: {status_info.get('termination_reason')} ({status_info.get('message', '')})")
                pbar.write(f"Path validation: {path_validation}")
                pbar.write('=========')

            running = statistics['overall']['running_count']
            accuracy = statistics['overall']['accuracy'] / running if running else 0.0
            pbar.set_description(
                f"Processing Questions (Entity Acc: {statistics['overall']['accuracy']}/{running} = {accuracy:.4f})"
            )

    for section, metric_values in navigation_metric_scores.items():
        statistics[section]['path_fidelity'] = aggregate_single_prediction_metrics(metric_values['path'])
        statistics[section]['final_entity'] = aggregate_answer_metrics(metric_values['answer'])

    statistics['overall'] = avg_dict(statistics['overall'])
    acc = statistics['overall']['accuracy']
    total = statistics['overall']['running_count']
    statistics['overall']['avg_accuracy'] = 100 * acc / total if total > 0 else 0
    print(f"\nFinal Entity Accuracy: {acc}/{total} = {statistics['overall']['avg_accuracy']:.2f}%")
    overall_path = statistics['overall']['path_fidelity']
    overall_entity = statistics['overall']['final_entity']
    print(
        'Navigation Metrics: '
        f"PED={overall_path.get('PED')}, "
        f"RED={overall_path.get('RED')}, "
        f"F1_SG={overall_path.get('F1_SG')}, "
        f"F1_REL={overall_path.get('F1_REL')}, "
        f"Hits1={overall_entity.get('Hits1')}, "
        f"MRR={overall_entity.get('MRR')}, "
        f"path_exact={overall_path.get('path_exact_match')}, "
        f"relation_exact={overall_path.get('relation_chain_exact_match')}, "
        f"triplet_f1={overall_path.get('triplet_f1')}"
    )

    if args.hops == 'n':
        for hop_size in sorted(key for key in statistics if key != 'overall'):
            statistics[hop_size] = avg_dict(statistics[hop_size])
            acc = statistics[hop_size]['accuracy']
            total = statistics[hop_size]['running_count']
            statistics[hop_size]['avg_accuracy'] = 100 * acc / total if total > 0 else 0
            print(f"Hop Size {hop_size} Entity Accuracy: {acc}/{total} = {statistics[hop_size]['avg_accuracy']:.2f}%")

    result_path = os.path.join(args.result_dir, args.dataset)
    os.makedirs(result_path, exist_ok=True)
    model_name = args.llm_model
    if args.use_instruct:
        model_name += '-instruct'
        if args.use_quantized:
            model_name += f'-q{args.quantization_bits}'

    question_limit_suffix = f"_questions{len(qa_df)}" if args.max_questions is not None else ''
    hybrid_suffix = f"_hybrid{args.hybrid_threshold}" if args.navigation_approach == 'hybrid' else ''
    max_actions_suffix = f"_maxactions{args.max_actions}" if args.max_actions is not None else '_fullactions'
    results_file = os.path.join(
        result_path,
        f"results_{args.hops}hop_{model_name}_{args.navigation_approach}_{args.memory_approach}_"
        f"{args.prompting_approach}_steps{args.max_navigation_steps}{hybrid_suffix}"
        f"{max_actions_suffix}{question_limit_suffix}_seed{args.seed}.json",
    )

    payload = {
        'config': {
            'model': args.llm_model,
            'use_instruct': args.use_instruct,
            'use_quantized': args.use_quantized,
            'quantization_bits': args.quantization_bits,
            'context_window': args.context_window,
            'temperature': args.temperature,
            'timeout': args.timeout,
            'seed': args.seed,
            'dataset': args.dataset,
            'hop_split': args.hops,
            'data_dir': args.data_dir,
            'navigation_approach': args.navigation_approach,
            'memory_approach': args.memory_approach,
            'prompting_approach': args.prompting_approach,
            'max_navigation_steps': args.max_navigation_steps,
            'max_actions': args.max_actions,
            'hybrid_threshold': args.hybrid_threshold,
            'max_parse_retries': args.max_parse_retries,
            'graph_directionality': 'outgoing',
            'max_questions': args.max_questions,
        },
        'statistics': statistics,
        'episodes': episodes,
    }
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(payload), f, indent=4)
    print(f"Results saved to {results_file}")
