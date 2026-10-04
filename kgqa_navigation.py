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
from numbers import Number
from pathlib import Path

from tqdm import tqdm

from model.navigation_llm_client import NavigationLLMKGQAClient
from model.model_config import (
    load_model_profile,
    model_result_config,
    validate_runtime_settings,
)
from utils.basic import load_pandas, load_triplets
from utils.graph_utils import Grapher, build_outgoing_index
from utils.kgqa_data_utils import (
    get_row_value,
    normalize_answer_entities,
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


def _parse_literal(value):
    """Parse a Python-literal string while preserving already-decoded values."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped[0] in '[({':
            try:
                return ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return value
    return value


def _is_triplet(value):
    """Return True when value structurally resembles one [head, relation, tail] edge."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and not any(isinstance(part, (list, tuple, dict, set)) for part in value)
    )


def _is_path(value):
    """Return True when value is one non-empty sequence of KG triplets."""
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(_is_triplet(edge) for edge in value)
    )


def flatten_reference_paths(value):
    """Flatten Paths/Multi-Paths/Graph-Multi-Paths into individual entity paths."""
    value = _parse_literal(value)
    paths = []

    def visit(obj):
        if _is_path(obj):
            paths.append([tuple(edge) for edge in obj])
            return
        if isinstance(obj, (list, tuple)):
            for child in obj:
                visit(child)

    visit(value)
    return paths


def normalize_relation_chains(value):
    """Normalize Path-Key or Multi-Paths-Key into a list of relation chains."""
    value = _parse_literal(value)
    if value is None:
        return []

    if isinstance(value, str):
        stripped = value.strip()
        return [stripped.split('->')] if stripped else []

    if not isinstance(value, (list, tuple)) or not value:
        return []

    # A flat list is one already-tokenized relation chain.
    if not any(isinstance(item, (list, tuple)) for item in value):
        # Multi-Paths-Key is commonly serialized as a list of "r1->r2" strings.
        if all(isinstance(item, str) and '->' in item for item in value):
            return [item.split('->') for item in value]
        return [list(value)]

    chains = []
    for item in value:
        item = _parse_literal(item)
        if isinstance(item, str):
            if item.strip():
                chains.append(item.strip().split('->'))
        elif isinstance(item, (list, tuple)) and item:
            chains.append(list(item))
    return chains


def get_reference_paths(row, graph_scope=False):
    """Return released or graph-expanded entity-level reference paths."""
    if graph_scope:
        column = 'Graph-Multi-Paths'
        return flatten_reference_paths(row[column]) if column in row and row[column] != '' else []

    if 'Multi-Paths' in row and row['Multi-Paths'] != '':
        return flatten_reference_paths(row['Multi-Paths'])
    if 'Paths' in row and row['Paths'] != '':
        return flatten_reference_paths(row['Paths'])
    return []


def get_reference_relation_chains(row, reference_paths=None):
    """Return all released relation-chain annotations for one question."""
    if 'Multi-Paths-Key' in row and row['Multi-Paths-Key'] != '':
        chains = normalize_relation_chains(row['Multi-Paths-Key'])
        if chains:
            return chains
    if 'Path-Key' in row and row['Path-Key'] != '':
        chains = normalize_relation_chains(row['Path-Key'])
        if chains:
            return chains

    chains = []
    seen = set()
    for path in reference_paths or []:
        chain = tuple(edge[1] for edge in path)
        if chain and chain not in seen:
            seen.add(chain)
            chains.append(list(chain))
    return chains


def score_path_against_references(predicted_path, reference_paths, relation_chains):
    """Match MINERVA multi-reference semantics for PED/F1_SG and RED/F1_REL."""
    if not reference_paths and not relation_chains:
        return None

    # Entity-level metrics are best-reference metrics over all candidate paths.
    base_chain = relation_chains[0] if relation_chains else None
    result = best_path_fidelity_score(
        predicted_path,
        reference_paths,
        base_chain,
    )

    if not relation_chains:
        relation_chains = get_reference_relation_chains({}, reference_paths)

    if relation_chains:
        relation_scores = [
            best_path_fidelity_score(predicted_path, [], chain)
            for chain in relation_chains
        ]
        red_values = [score.get('RED') for score in relation_scores if score and score.get('RED') is not None]
        f1_values = [score.get('F1_REL') for score in relation_scores if score and score.get('F1_REL') is not None]
        exact_values = [
            score.get('relation_chain_exact_match')
            for score in relation_scores
            if score and score.get('relation_chain_exact_match') is not None
        ]
        prefix_values = [
            score.get('relation_prefix_recall')
            for score in relation_scores
            if score and score.get('relation_prefix_recall') is not None
        ]
        if result is None:
            result = {}
        result = dict(result)
        result['RED'] = min(red_values) if red_values else None
        result['F1_REL'] = max(f1_values) if f1_values else None
        result['relation_chain_exact_match'] = max(exact_values) if exact_values else None
        result['relation_prefix_recall'] = max(prefix_values) if prefix_values else None

    return result


def _mean_available(values):
    """Arithmetic mean over numeric, non-None values."""
    available = [float(value) for value in values if isinstance(value, Number)]
    return sum(available) / len(available) if available else None


def _family_macro_scores(scores, family_ids, family_sizes=None):
    """Macro-average metric records over question families."""
    if not scores or not family_ids or len(scores) != len(family_ids):
        return None

    families = {}
    for score, family_id in zip(scores, family_ids):
        families.setdefault(str(family_id), []).append(score)

    metric_names = sorted({
        key
        for family_scores in families.values()
        for score in family_scores
        for key in score
    })
    result = {
        'count': len(scores),
        'family_count': len(families),
    }
    for metric_name in metric_names:
        family_values = [
            _mean_available(score.get(metric_name) for score in family_scores)
            for family_scores in families.values()
        ]
        result[metric_name] = _mean_available(family_values)
        result[f'{metric_name}_support'] = sum(value is not None for value in family_values)

    if family_sizes is not None and len(family_sizes) == len(family_ids):
        observed_counts = {}
        declared_sizes = {}
        for family_id, family_size in zip(family_ids, family_sizes):
            key = str(family_id)
            observed_counts[key] = observed_counts.get(key, 0) + 1
            try:
                size = int(family_size)
            except (TypeError, ValueError):
                size = 0
            if size > 0:
                declared_sizes[key] = size
        result['families_complete'] = bool(
            declared_sizes
            and all(
                observed_counts.get(key, 0) == declared_sizes.get(key)
                for key in observed_counts
            )
        )
    return result


def aggregate_family_answer_metrics(scores, family_ids, family_sizes=None):
    """Family-macro counterpart of aggregate_answer_metrics."""
    result = _family_macro_scores(scores, family_ids, family_sizes)
    if result is None:
        return None
    hits1 = result.get('Hits1')
    result['scored'] = len(scores)
    result['correct'] = sum(float(score.get('Hits1') or 0.0) for score in scores)
    result['accuracy'] = hits1 or 0.0
    return result


def prepare_demonstration_frame(train_df):
    """Expose flattened Multi-Paths through the legacy Paths hook used by demo sampling."""
    demo_df = train_df.copy()
    if 'Multi-Paths' in demo_df.columns:
        demo_df['Paths'] = demo_df['Multi-Paths'].apply(flatten_reference_paths)
    return demo_df


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


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative KG navigation for QA datasets")

    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path containing the dataset splits.')
    parser.add_argument('--dataset', type=str, default='mquake_single',
                        help='Name of the dataset to process.')
    parser.add_argument('--hops', type=str, default='n',
                        help='QA dataset hop split to evaluate.')
    parser.add_argument('--triplets-file', type=str, default='triplets.txt',
                        help='Optional triplet file override for the dataset.')
    parser.add_argument('--qa-file-prefix', type=str, default=None,
                        help='Optional QA file prefix for the dataset.')
    parser.add_argument('--entity-id-col', type=str, default='QID',
                        help='Column name for entity ID in node_data.csv.')
    parser.add_argument('--relation-id-col', type=str, default='Property',
                        help='Column name for relation ID in relation_data.csv.')
    parser.add_argument('--max-questions', type=int, default=None,
                        help='Process only the first N test questions (must be positive).')
    parser.add_argument('--question-idxs', type=int, nargs='+', default=None,
                        help='Process only the questions at these indices (0-based). Overrides --max-questions.')

    # LLM parameters
    parser.add_argument('--model-config', type=str, default='configs/models/gemma3.json',
                        help='Path to a validated model profile JSON file.')
    parser.add_argument('--model-id', type=str, default=None,
                        help='Optional backend model ID override for this server.')
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

    if args.qa_file_prefix is not None:
        qa_file = os.path.join(args.data_dir, args.dataset, f'{args.qa_file_prefix}_qa_{args.hops}hop.csv')
    else:
        qa_file = os.path.join(args.data_dir, args.dataset, f'qa_{args.hops}hop.csv')

    data_dir = os.path.join(args.data_dir, args.dataset)
    triplet_file = os.path.join(data_dir, args.triplets_file)
    entity_file = os.path.join(data_dir, 'node_data.csv')
    relation_file = os.path.join(data_dir, 'relation_data.csv')

    entity_title, relation_title, title_mapping_status = load_title_maps(
        entity_file,
        relation_file,
        args.entity_id_col,
        args.relation_id_col,
    )

    all_triplets_df = load_triplets(triplet_file)
    all_triplets = set(tuple(triplet) for triplet in all_triplets_df.values)
    outgoing_index = build_outgoing_index(all_triplets) # TODO: Add an option to build bidirectional index for other datasets. For now, only outgoing edges are used for MQuAKE and kinship.
    grapher = Grapher(all_triplets)
    relation_index = grapher.get_relation_index() if args.n_shots > 0 else {}

    qa_all_df = load_pandas(qa_file)
    train_df = qa_all_df[qa_all_df['SplitLabel'] == 'train'].copy()
    qa_df = qa_all_df[qa_all_df['SplitLabel'] == 'test'].copy() # TODO: Add an option to evaluate on validation split for hyperparameter tuning.
    if args.question_idxs is not None:
        qa_df = qa_df[qa_df['Question-Number'].isin(args.question_idxs)].copy()
        args.max_questions = len(qa_df)
    elif args.max_questions is not None:
        qa_df = qa_df.head(args.max_questions).copy()

    is_multi_answer = bool(
        not qa_df.empty
        and qa_df['Answer-Entity'].apply(
            lambda value: isinstance(value, str) and value.strip().startswith('[')
        ).all()
    )
    has_family_metadata = 'Question-Family-ID' in qa_df.columns
    has_family_sizes = 'Question-Family-Size' in qa_df.columns
    has_graph_answers = 'Graph-Answer-Entity' in qa_df.columns

    qa_df = qa_df.reset_index(drop=False).rename(columns={'index': 'dataframe_index'})

    model_profile = load_model_profile(args.model_config)
    validate_runtime_settings(
        model_profile,
        context_window=args.context_window,
        use_think=args.use_think,
        structured_output=args.structured_output,
    )
    config_path = Path(__file__).with_name('openwebui_config.json').parent / 'configs' / 'openwebui_config.json'
    client = NavigationLLMKGQAClient(
        config_path,
        model_profile=model_profile,
        model_id=args.model_id,
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
        train_df=prepare_demonstration_frame(train_df.reset_index(drop=True)),
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
        section: {
            'path': [],
            'answer': [],
            'graph_path': [],
            'graph_answer': [],
            'family_ids': [],
            'family_sizes': [],
            'path_family_ids': [],
            'path_family_sizes': [],
            'graph_path_family_ids': [],
            'graph_path_family_sizes': [],
        }
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

            valid_answer_entities = normalize_answer_entities(row['Answer-Entity'])
            graph_answer_entities = (
                normalize_answer_entities(row['Graph-Answer-Entity'])
                if has_graph_answers else set()
            )
            reference_paths = get_reference_paths(row, graph_scope=False)
            graph_reference_paths = get_reference_paths(row, graph_scope=True)
            relation_chains = get_reference_relation_chains(row, reference_paths)
            reference_path_source = 'dataset_paths' if reference_paths else None
            graph_reference_path_source = 'graph_dataset_paths' if graph_reference_paths else None

            # Match MINERVA's reconstruction semantics when exhaustive entity-level
            # references are absent: enumerate all graph realizations for every
            # released relation-chain annotation and retain answer-consistent paths.
            if not reference_paths and is_multi_answer and relation_chains:
                seen_paths = set()
                reconstructed = []
                for relation_chain in relation_chains:
                    for path in grapher.find_paths_by_relation_chain(
                        start_entity=start_node,
                        relation_chain=relation_chain,
                        target_entities=valid_answer_entities,
                    ):
                        key = tuple(tuple(edge) for edge in path)
                        if key not in seen_paths:
                            seen_paths.add(key)
                            reconstructed.append(path)
                reference_paths = reconstructed
                if reference_paths:
                    reference_path_source = 'lazy_relation_chain'

            if has_graph_answers and not graph_reference_paths and relation_chains:
                seen_paths = set()
                reconstructed = []
                for relation_chain in relation_chains:
                    for path in grapher.find_paths_by_relation_chain(
                        start_entity=start_node,
                        relation_chain=relation_chain,
                        target_entities=graph_answer_entities,
                    ):
                        key = tuple(tuple(edge) for edge in path)
                        if key not in seen_paths:
                            seen_paths.add(key)
                            reconstructed.append(path)
                graph_reference_paths = reconstructed
                if graph_reference_paths:
                    graph_reference_path_source = 'lazy_relation_chain'

            path_score = score_path_against_references(
                predicted_path,
                reference_paths,
                relation_chains,
            )
            graph_path_score = (
                score_path_against_references(
                    predicted_path,
                    graph_reference_paths,
                    relation_chains,
                )
                if has_graph_answers else None
            )

            missing_answer_score = {
                'Hits1': 0.0,
                'MRR': None,
                'final_entity_correct': 0.0,
            }
            answer_entity_score = (
                score_single_final_entity(final_entity, valid_answer_entities)
                if final_entity is not None else dict(missing_answer_score)
            )
            graph_answer_entity_score = (
                score_single_final_entity(final_entity, graph_answer_entities)
                if has_graph_answers and final_entity is not None
                else dict(missing_answer_score) if has_graph_answers else None
            )
            correct = bool(answer_entity_score.get('Hits1'))
            graph_correct = (
                bool(graph_answer_entity_score.get('Hits1'))
                if graph_answer_entity_score is not None else None
            )

            metric_sections = ['overall']
            if args.hops == 'n' and f'{hop}' in statistics:
                metric_sections.append(f'{hop}')
            family_id = get_row_value(row, 'Question-Family-ID') if has_family_metadata else None
            family_size = get_row_value(row, 'Question-Family-Size') if has_family_sizes else None

            for section in metric_sections:
                if path_score is not None:
                    navigation_metric_scores[section]['path'].append(path_score)
                    if has_family_metadata:
                        navigation_metric_scores[section]['path_family_ids'].append(family_id)
                        navigation_metric_scores[section]['path_family_sizes'].append(family_size)
                navigation_metric_scores[section]['answer'].append(answer_entity_score)
                if graph_path_score is not None:
                    navigation_metric_scores[section]['graph_path'].append(graph_path_score)
                    if has_family_metadata:
                        navigation_metric_scores[section]['graph_path_family_ids'].append(family_id)
                        navigation_metric_scores[section]['graph_path_family_sizes'].append(family_size)
                if graph_answer_entity_score is not None:
                    navigation_metric_scores[section]['graph_answer'].append(graph_answer_entity_score)
                if has_family_metadata:
                    navigation_metric_scores[section]['family_ids'].append(family_id)
                    navigation_metric_scores[section]['family_sizes'].append(family_size)

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
                'gold_relation_chain_count': len(relation_chains),
                'question_family_id': family_id,
                'question_family_size': family_size,
                'graph_gold_answer_entities': sorted(graph_answer_entities) if has_graph_answers else None,
                'graph_gold_answer_labels': (
                    [entity_title.get(entity, entity) for entity in sorted(graph_answer_entities)]
                    if has_graph_answers else None
                ),
                'graph_gold_answer_text': get_row_value(row, 'Graph-Answer') if has_graph_answers else None,
                'graph_reference_path_source': graph_reference_path_source if has_graph_answers else None,
                'graph_reference_path_count': len(graph_reference_paths) if has_graph_answers else None,
                'predicted_terminal_entity': final_entity,
                'predicted_terminal_label': entity_title.get(final_entity, final_entity) if final_entity is not None else None,
                'answer_correct': correct,
                'graph_answer_correct': graph_correct,
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
                'graph_path_fidelity': graph_path_score,
                'graph_final_entity_score': graph_answer_entity_score,
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
                if has_graph_answers:
                    pbar.write(f"Graph answer entities: {sorted(graph_answer_entities)}")
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
        # Backward-compatible primary metrics: released references, instance-micro.
        statistics[section]['path_fidelity'] = aggregate_single_prediction_metrics(metric_values['path'])
        statistics[section]['final_entity'] = aggregate_answer_metrics(metric_values['answer'])

        if has_family_metadata:
            statistics[section]['path_fidelity_family_macro'] = _family_macro_scores(
                metric_values['path'],
                metric_values['path_family_ids'],
                metric_values['path_family_sizes'] if has_family_sizes else None,
            )
            statistics[section]['final_entity_family_macro'] = aggregate_family_answer_metrics(
                metric_values['answer'],
                metric_values['family_ids'],
                metric_values['family_sizes'] if has_family_sizes else None,
            )

        if has_graph_answers:
            statistics[section]['graph_path_fidelity'] = aggregate_single_prediction_metrics(
                metric_values['graph_path']
            )
            statistics[section]['graph_final_entity'] = aggregate_answer_metrics(
                metric_values['graph_answer']
            )
            if has_family_metadata:
                statistics[section]['graph_path_fidelity_family_macro'] = _family_macro_scores(
                    metric_values['graph_path'],
                    metric_values['graph_path_family_ids'],
                    metric_values['graph_path_family_sizes'] if has_family_sizes else None,
                )
                statistics[section]['graph_final_entity_family_macro'] = aggregate_family_answer_metrics(
                    metric_values['graph_answer'],
                    metric_values['family_ids'],
                    metric_values['family_sizes'] if has_family_sizes else None,
                )

    statistics['overall'] = avg_dict(statistics['overall'])
    acc = statistics['overall']['accuracy']
    total = statistics['overall']['running_count']
    statistics['overall']['avg_accuracy'] = 100 * acc / total if total > 0 else 0
    print(f"\nFinal Entity Accuracy: {acc}/{total} = {statistics['overall']['avg_accuracy']:.2f}%")
    overall_path = statistics['overall']['path_fidelity']
    overall_entity = statistics['overall']['final_entity']
    print(
        'Navigation Metrics (Answer / instance-micro): '
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
    if has_family_metadata:
        family_path = statistics['overall'].get('path_fidelity_family_macro') or {}
        family_entity = statistics['overall'].get('final_entity_family_macro') or {}
        print(
            'Navigation Metrics (Answer / family-macro): '
            f"PED={family_path.get('PED')}, "
            f"RED={family_path.get('RED')}, "
            f"F1_SG={family_path.get('F1_SG')}, "
            f"F1_REL={family_path.get('F1_REL')}, "
            f"Hits1={family_entity.get('Hits1')}, "
            f"MRR={family_entity.get('MRR')}"
        )
    if has_graph_answers:
        graph_path = statistics['overall'].get('graph_path_fidelity') or {}
        graph_entity = statistics['overall'].get('graph_final_entity') or {}
        print(
            'Navigation Metrics (Graph-Answer / instance-micro): '
            f"PED={graph_path.get('PED')}, "
            f"RED={graph_path.get('RED')}, "
            f"F1_SG={graph_path.get('F1_SG')}, "
            f"F1_REL={graph_path.get('F1_REL')}, "
            f"Hits1={graph_entity.get('Hits1')}, "
            f"MRR={graph_entity.get('MRR')}"
        )
        if has_family_metadata:
            graph_family_path = statistics['overall'].get('graph_path_fidelity_family_macro') or {}
            graph_family_entity = statistics['overall'].get('graph_final_entity_family_macro') or {}
            print(
                'Navigation Metrics (Graph-Answer / family-macro): '
                f"PED={graph_family_path.get('PED')}, "
                f"RED={graph_family_path.get('RED')}, "
                f"F1_SG={graph_family_path.get('F1_SG')}, "
                f"F1_REL={graph_family_path.get('F1_REL')}, "
                f"Hits1={graph_family_entity.get('Hits1')}, "
                f"MRR={graph_family_entity.get('MRR')}"
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
    model_name = model_profile.result_name

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
            **model_result_config(
                model_profile,
                backend_model_id=client.model_choice,
            ),
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
            'has_family_metadata': has_family_metadata,
            'has_family_sizes': has_family_sizes,
            'has_graph_answers': has_graph_answers,
            'primary_reference_scope': 'Answer',
        },
        'statistics': statistics,
        'episodes': episodes,
    }
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(payload), f, indent=4)
    print(f"Results saved to {results_file}")
