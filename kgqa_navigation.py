"""
Run iterative LLM-based knowledge-graph navigation experiments for KGQA.

The controller owns the symbolic graph. At each navigation state it exposes only
legal outgoing one-hop actions from the current entity, executes the selected KG
edge, and treats the terminal graph entity as the prediction.
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from model.navigation_llm_client import NavigationLLMKGQAClient
from model.constants import valid_models
from utils.basic import extract_literals, load_pandas, load_triplets
from utils.graph_utils import RelationEntityGrapher, build_outgoing_index
from utils.kgqa_data_utils import (
    get_row_value,
    normalize_answer_entities,
    normalize_reference_paths,
    normalize_relation_chain,
    to_jsonable,
)
from utils.kgqa_navigation_utils import (
    best_path_fidelity_score,
    number_to_shot_label,
    sample_navigation_demonstrations,
    validate_executed_path,
)
from utils.kgqa_statistics import (
    avg_dict,
    initialize_navigation_statistics as initialize_statistics,
    update_navigation_stats as update_stats,
)
from utils.kgqa_utils import load_title_maps
from utils.kgqa_navigation_metrics import (
    aggregate_answer_metrics,
    aggregate_single_prediction_metrics,
    score_single_final_entity,
)


def summarize_original_ids(original_ids):
    """Return a compact JSON-friendly summary of selected original option IDs."""
    ids = [int(original_id) for original_id in original_ids]
    if not ids:
        return {'mode': 'empty', 'count': 0}

    contiguous = all(
        original_id == ids[0] + offset
        for offset, original_id in enumerate(ids)
    )
    if contiguous:
        return {
            'mode': 'range',
            'count': len(ids),
            'start': ids[0],
            'end': ids[-1],
        }

    if len(ids) <= 20:
        return {
            'mode': 'ids',
            'count': len(ids),
            'ids': ids,
        }

    return {
        'mode': 'preview',
        'count': len(ids),
        'min': min(ids),
        'max': max(ids),
        'head': ids[:10],
        'tail': ids[-10:],
    }


def compact_max_actions_truncations(truncations):
    """Compact verbose shown_original_ids lists without changing selection metadata."""
    compacted = []
    for truncation in truncations:
        record = dict(truncation)
        original_ids = record.pop('shown_original_ids', None)
        if original_ids is not None:
            record['shown_original_ids_summary'] = summarize_original_ids(original_ids)
        compacted.append(record)
    return compacted


def is_multi_answer_value(value):
    """Return whether a dataset cell encodes the multi-answer list format."""
    if isinstance(value, (list, tuple, set)):
        return True
    return isinstance(value, str) and value.strip().startswith('[')


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative KG navigation for QA datasets")

    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path containing the dataset splits.')
    parser.add_argument('--dataset', type=str, default='mquake_single',
                        help='Name of the dataset to process.')
    parser.add_argument('--hops', type=str, default='n',
                        help='QA dataset hop split to evaluate.')
    parser.add_argument('--max-questions', type=int, default=None,
                        help='Process only the first N test questions (must be positive).')
    parser.add_argument('--question-idxs', type=int, nargs='+', default=None,
                        help='Process only the questions at these indices (0-based). Overrides --max-questions.')

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
    parser.add_argument('--use-think', action='store_true',
                        help='Whether to use the "think" option for the LLM API (may improve quality but consumes more tokens). Not all models support this option.')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Read inactivity timeout in seconds for LLM API requests.')
    parser.add_argument('--connect-timeout', type=int, default=5,
                        help='Connection-establishment timeout in seconds.')
    parser.add_argument('--timeout-cooldown', type=float, default=0.0,
                        help='Seconds to wait after a read timeout before continuing.')
    parser.add_argument('--max-output-tokens', type=int, default=64,
                        help='Maximum number of tokens generated per LLM request.')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Sampling temperature for the LLM (0 = deterministic).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for model inference.')

    # Navigation parameters
    parser.add_argument('--max-actions', type=int, default=None,
                        help=('Optional cap for the number of options shown in a single prompt. '
                              'If exceeded, only the first N sorted options are shown to the LLM.'))
    parser.add_argument('--max-actions-policy', default='first', choices=['first', 'random', 'question-aware'], help='Policy used when --max-actions is exceeded.')
    parser.add_argument('--max-navigation-steps', type=int, default=4,
                        help='Maximum number of graph edges the model may traverse before termination.')
    parser.add_argument('--n-shots', type=int, default=0,
                        help='Number of complete solved train trajectories to prepend as navigation demonstrations.')
    parser.add_argument('--demo-history-mode', type=str, default='full',
                        choices=['full', 'last', 'random'],
                        help='History shown inside demonstrated hops: full path, last hop, or one seeded random hop.')
    parser.add_argument('--demo-max-actions', type=int, default=10,
                        help='Maximum number of available actions shown at each demonstrated hop.')
    # TODO: Factorized and Hybrid need reverification for demonstrations. For now, only tuple is supported for n-shot demos.
    parser.add_argument('--navigation-approach', type=str, default='tuple',
                        choices=['tuple', 'factorized', 'hybrid'],
                        help='Tuple, factorized relation/entity, or threshold-based hybrid navigation.')
    # TODO: Must update demonstration for 'none' memory approach. For now, only full is supported for n-shot demos.
    parser.add_argument('--memory-approach', type=str, default='full',
                        choices=['none', 'full'],
                        help='Observation memory: none hides previous edges; full shows the traversed path.')
    # TODO: Must implement IO prompting or remove the option.
    # TODO: Additionally, add the option for last LLM call to generate the final answer instead of the last entity in the path.
    parser.add_argument('--prompting-approach', type=str, default='zero-shot',
                        choices=['io', 'zero-shot', 'one-shot'],
                        help='Prompting mode label. Use --n-shots for n-shot demonstrations; one-shot sets --n-shots=1 when omitted.')
    parser.add_argument('--hybrid-threshold', type=int, default=50,
                        help='Use tuple mode when neighborhood size is <= this threshold, else factorized.')
    # TODO: Recheck this or be more lenient so long as {action=, stop=} is present in the JSON output.
    parser.add_argument('--max-parse-retries', type=int, default=0,
                        help='Retry a navigation decision this many times after malformed JSON output.')
    parser.add_argument('--structured-output', action='store_true',
                        help=('Constrain each navigation decision with an Ollama JSON Schema. '
                              'The schema permits only legal action/relation IDs and valid stop combinations.'))

    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug mode with verbose output.')
    parser.add_argument('--show-navigation', '--show-actions', dest='show_navigation', action='store_true',
                        help='Show every navigation prompt, model response, and validated move.')

    # Result parameters
    parser.add_argument('--result-dir', type=str, default='./results/navigation/',
                        help='Directory to save the results.')

    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    if args.max_navigation_steps < 0:
        raise ValueError('--max-navigation-steps must be non-negative.')
    if args.max_questions is not None and args.max_questions < 1:
        raise ValueError('--max-questions must be positive.')
    if args.question_idxs is not None and any(idx < 0 for idx in args.question_idxs):
        raise ValueError('--question-idxs must be non-negative.')
    if args.max_actions is not None and args.max_actions < 1:
        raise ValueError('--max-actions must be positive when provided.')
    if args.n_shots < 0:
        raise ValueError('--n-shots must be non-negative.')
    if args.demo_max_actions < 1:
        raise ValueError('--demo-max-actions must be positive.')
    if args.hybrid_threshold < 0:
        raise ValueError('--hybrid-threshold must be non-negative.')
    if args.max_parse_retries < 0:
        raise ValueError('--max-parse-retries must be non-negative.')
    if args.max_parse_retries != 0 and args.structured_output:
        raise ValueError('--max-parse-retries != 0 is incompatible with --structured-output, which guarantees valid JSON output.')
    if args.prompting_approach == 'one-shot' and args.n_shots == 0:
        args.n_shots = 1
    if args.prompting_approach == 'io':
        raise NotImplementedError(
            "--prompting-approach 'io' is not implemented for iterative navigation yet. "
            'Use --prompting-approach zero-shot with --n-shots for n-shot prompting.'
        )
    prompting_label = number_to_shot_label(args.n_shots)

    data_dir = os.path.join(args.data_dir, args.dataset)
    qa_file = os.path.join(data_dir, f'qa_{args.hops}hop.csv')
    triplet_file = os.path.join(data_dir, 'triplets.txt')
    entity_file = os.path.join(data_dir, 'node_data.csv')
    relation_file = os.path.join(data_dir, 'relation_data.csv')

    entity_title, relation_title, title_mapping_status = load_title_maps(
        entity_file,
        relation_file,
    )

    all_triplets_df = load_triplets(triplet_file)
    all_triplets = set(tuple(triplet) for triplet in all_triplets_df.values)
    outgoing_index = build_outgoing_index(all_triplets) # TODO: Add an option to build bidirectional index for other datasets. For now, only outgoing edges are used for MQuAKE and kinship_v2.
    grapher = RelationEntityGrapher(all_triplets)
    relation_index = grapher.get_relation_index() if args.n_shots > 0 else {}

    qa_all_df = load_pandas(qa_file)
    train_df = qa_all_df[qa_all_df['SplitLabel'] == 'train'].copy()
    qa_df = qa_all_df[qa_all_df['SplitLabel'] == 'test'].copy() # TODO: Add an option to evaluate on validation split for hyperparameter tuning.
    if args.question_idxs is not None:
        qa_df = qa_df[qa_df['Question-Number'].isin(args.question_idxs)].copy()
        args.max_questions = len(qa_df)
    elif args.max_questions is not None:
        qa_df = qa_df.head(args.max_questions).copy()

    # Normalize list-valued answer columns used by multi-answer datasets.
    if not qa_df.empty and qa_df['Answer'].apply(
        lambda value: isinstance(value, str) and value.strip().startswith('[')
    ).all():
        qa_df['Answer'] = extract_literals(qa_df['Answer'])
        qa_df['Answer-Entity'] = extract_literals(qa_df['Answer-Entity'])

    qa_df = qa_df.reset_index(drop=False).rename(columns={'index': 'dataframe_index'})

    config_path = Path(__file__).with_name('openwebui_config.json').parent / 'configs' / 'openwebui_config.json'
    client = NavigationLLMKGQAClient(
        config_path,
        model_choice=args.llm_model,
        use_instruct=args.use_instruct,
        use_quantized=args.use_quantized,
        quantization_bits=args.quantization_bits,
        context_window=args.context_window,
        seed=args.seed,
        temperature=args.temperature,
        timeout=args.timeout,
        connect_timeout=args.connect_timeout,
        timeout_cooldown=args.timeout_cooldown,
        max_output_tokens=args.max_output_tokens,
        use_think=args.use_think,
        debug=args.debug,
    )

    demonstration_records = sample_navigation_demonstrations(
        train_df=train_df.reset_index(drop=True),
        outgoing_index=outgoing_index,
        relation_index=relation_index,
        n_shots=args.n_shots,
        seed=args.seed,
    )
    demonstration_prefix = client.format_navigation_demonstrations(
        demonstrations=demonstration_records,
        outgoing_index=outgoing_index,
        entity_title=entity_title,
        relation_title=relation_title,
        demo_history_mode=args.demo_history_mode,
        demo_max_actions=args.demo_max_actions,
        seed=args.seed,
    )
    # Demonstration construction is complete; semantic test evaluation can rebuild
    # this index lazily later if a multi-answer row actually requires it.
    grapher.clear_relation_index()

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
                max_actions_policy=args.max_actions_policy,
                navigation_approach=args.navigation_approach,
                memory_approach=args.memory_approach,
                prompting_approach=prompting_label,
                hybrid_threshold=args.hybrid_threshold,
                max_parse_retries=args.max_parse_retries,
                structured_output=args.structured_output,
                demonstration_prefix=demonstration_prefix,
                n_shots=args.n_shots,
                trace=pbar.write if args.show_navigation else None,
            )

            predicted_path = status_info.get('predicted_path', [])
            final_entity = status_info.get('final_entity')
            raw_answer_entities = row['Answer-Entity']
            valid_answer_entities = normalize_answer_entities(raw_answer_entities)
            reference_paths = normalize_reference_paths(row['Paths']) if 'Paths' in qa_df.columns else []
            relation_chain = normalize_relation_chain(row['Path-Key']) if 'Path-Key' in qa_df.columns else None
            reference_path_source = 'dataset_paths' if reference_paths else None

            # Match MINERVA's multi-answer semantics when exhaustive entity-level
            # paths are not stored: lazily enumerate all graph realizations that
            # follow the annotated relation chain exactly and end at a valid answer.
            if (
                not reference_paths
                and relation_chain is not None
                and valid_answer_entities
                and is_multi_answer_value(raw_answer_entities)
            ):
                reference_paths = grapher.find_paths_by_relation_chain(
                    start_entity=start_node,
                    relation_chain=relation_chain,
                    target_entities=valid_answer_entities,
                )
                if reference_paths:
                    reference_path_source = 'lazy_relation_chain'

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
                'gold_reference_path_source': reference_path_source,
                'gold_reference_path_count': len(reference_paths),
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
                'prompting_approach': status_info.get('prompting_approach', prompting_label),
                'n_shots': status_info.get('n_shots', args.n_shots),
                'hybrid_threshold': status_info.get('hybrid_threshold', args.hybrid_threshold),
                'max_actions': status_info.get('max_actions', args.max_actions),
                'max_actions_policy': status_info.get('max_actions_policy', args.max_actions_policy),
                'max_parse_retries': status_info.get('max_parse_retries', args.max_parse_retries),
                'structured_output': bool(status_info.get('structured_output', args.structured_output)),
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
                'max_actions_truncated': bool(status_info.get('max_actions_truncated')),
                'max_actions_truncations': compact_max_actions_truncations(
                    status_info.get('max_actions_truncations', [])
                ),
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

    # Semantic evaluation no longer needs the graph-derived relation index.
    grapher.clear_relation_index()

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

    result_path = os.path.join(args.result_dir, args.dataset, args.prompting_approach.replace('-', '_'))
    os.makedirs(result_path, exist_ok=True)
    model_name = args.llm_model
    if args.use_instruct:
        model_name += '-instruct'
        if args.use_quantized:
            model_name += f'-q{args.quantization_bits}'

    question_limit_suffix = f"_questions{len(qa_df)}" if args.max_questions is not None else ''
    hybrid_suffix = f"_hybrid{args.hybrid_threshold}" if args.navigation_approach == 'hybrid' else ''
    max_actions_suffix = (f"_maxactions{args.max_actions}_pol{args.max_actions_policy}"
                          if args.max_actions is not None else '_fullactions')
    structured_suffix = '_structured' if args.structured_output else '_unstructured'
    results_file = os.path.join(
        result_path,
        f"results_{args.hops}hop_{model_name}_{args.navigation_approach}_mem{args.memory_approach}_"
        f"steps{args.max_navigation_steps}{hybrid_suffix}"
        f"{max_actions_suffix}{structured_suffix}{question_limit_suffix}_seed{args.seed}.json",
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
            'connect_timeout': args.connect_timeout,
            'timeout_cooldown': args.timeout_cooldown,
            'max_output_tokens': args.max_output_tokens,
            'seed': args.seed,
            'dataset': args.dataset,
            'hop_split': args.hops,
            'data_dir': args.data_dir,
            'navigation_approach': args.navigation_approach,
            'memory_approach': args.memory_approach,
            'prompting_approach': prompting_label,
            'requested_prompting_approach': args.prompting_approach,
            'n_shots': args.n_shots,
            'demo_history_mode': args.demo_history_mode,
            'demo_max_actions': args.demo_max_actions,
            'demonstrations': demonstration_records,
            'max_navigation_steps': args.max_navigation_steps,
            'max_actions': args.max_actions,
            'max_actions_policy': args.max_actions_policy,
            'hybrid_threshold': args.hybrid_threshold,
            'max_parse_retries': args.max_parse_retries,
            'structured_output': args.structured_output,
            'graph_directionality': 'outgoing',
            'title_mapping': title_mapping_status,
            'max_questions': args.max_questions,
        },
        'statistics': statistics,
        'episodes': episodes,
    }
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(payload), f, indent=4)
    print(f"Results saved to {results_file}")
