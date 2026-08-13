"""
This script is designed for running Knowledge Graph Question Answering (KGQA) experiments using generic Large Language Models (LLMs) and an iterative navigation approach.
"""

import argparse
import ast
import os
from pathlib import Path

import json

from tqdm import tqdm

from model.LLM_KGQA import LLM_KGQA_Client
from model.constants import valid_models

from utils.basic import load_triplets, load_pandas, extract_literals
from utils.kgqa_utils import compare_answers, extract_final_answer
from utils.graph_utils import build_outgoing_index
from llm_navigation_metrics import (
    aggregate_answer_metrics,
    aggregate_single_prediction_metrics,
    score_path_fidelity_against_references,
    score_single_final_entity,
)

from collections import defaultdict
from typing import Dict

def parse_args():
    """
    The `parse_args` function defines and parses command-line arguments for the script. These arguments include:
    - Dataset and data directory paths
    - Graph-navigation parameters
    - LLM model selection and timeout settings
    - Debugging options
    """
    parser = argparse.ArgumentParser(description="Graph Navigation for QA Dataset")
    
    # dataset parameters
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
    # TODO: finish implementing the max-actions parameter, currently it is not used in the navigation process.
    parser.add_argument('--max-actions', type=int, default=200,
                        help='Maximum number of neighborhood can have before it is cut off for the model to consider.')
    parser.add_argument('--max-navigation-steps', type=int, default=4,
                        help='Maximum number of graph edges the model may traverse before it must stop.')
    # TODO: finish implementing the navigation approach, currently only 'tuple' is supported.
    parser.add_argument('--navigation-approach', type=str, default='tuple',
                        choices=['tuple', 'factorized', 'hybrid'],
                        help='Navigation approach for the model to use when generating actions. Tuple: (relation, entity) pairs; Factorized: relation and entity separately; Hybrid: combination of both.')
    # TODO: finish implementing the memory approach, currently only 'none' is supported.
    parser.add_argument('--memory-approach', type=str, default='none',
                        choices=['none', 'full'],
                        help='Memory approach for the model to use when generating actions. None: no memory; Full: all previous actions are remembered.')
    # TODO: finish implementing the prompting approach, currently only 'zero-shot' is supported.
    parser.add_argument('--prompting-approach', type=str, default='zero-shot',
                        choices=['io', 'zero-shot', 'one-shot'],
                        help='Prompting approach for the model to use when generating actions. Zero-shot: no examples; One-shot: one example; IO: input-output pairs.')


    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug mode with verbose output.')
    parser.add_argument('--show-navigation', '--show-actions', dest='show_navigation', action='store_true',
                        help='Show every navigation prompt, model response, and validated move.')
    
    # Result parameters
    parser.add_argument('--result-dir', type=str, default='./results',
                        help='Directory to save the results.')

    return parser.parse_args()

# Main Execution
"""
The main block of the script handles the following:
1. Argument validation and adjustments based on user input.
2. Loading datasets, triplets, and entity/relation mappings.
3. Initializing the LLM client for processing questions.
4. Iteratively navigating the graph for each question and evaluating predictions.
5. Saving the results to a JSON file.
"""
def initialize_statistics(total: int) -> Dict:
    return {
        'accuracy': 0,
        'running_count': 0,
        'total': total,
        'navigation_steps': defaultdict(int),
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

def update_stats(
    stats_dict: Dict, 
    status_info: Dict, 
    result: str, 
    full_pred: str, 
    navigation_steps: int,
) -> None:
    stats_dict['accuracy'] += int(result)
    stats_dict['running_count'] += 1
    stats_dict['navigation_steps'][navigation_steps] += 1
    
    # append if exists
    if 'prompt_tokens' in status_info: stats_dict['prompt_tokens'].append(status_info['prompt_tokens'])
    if 'response_tokens' in status_info: stats_dict['response_tokens'].append(status_info['response_tokens'])
    if 'total_tokens' in status_info: stats_dict['total_tokens'].append(status_info['total_tokens'])

    if 'response_seconds' in status_info: stats_dict['response_seconds'].append(status_info['response_seconds'])
    if 'prompt_seconds' in status_info: stats_dict['prompt_seconds'].append(status_info['prompt_seconds'])
    if 'total_seconds' in status_info: stats_dict['total_seconds'].append(status_info['total_seconds'])

    if 'prompt_tps' in status_info: stats_dict['prompt_tps'].append(status_info['prompt_tps'])
    if 'completion_tps' in status_info: stats_dict['completion_tps'].append(status_info['completion_tps'])

    stats_dict['unknown'] += int(full_pred == "UNKNOWN")
    stats_dict['timeouts'] += int(full_pred == "TIMEOUT")
    stats_dict['errors'] += int(full_pred == "ERROR")

def average(lst):
    return sum(lst) / len(lst) if lst else 0

def avg_dict(vals: Dict[str, object]) -> Dict[str, float]:
    """
    Averages the values in a dictionary. If a value is a list, it computes the average of the list.
    """
    out = {}
    for k, v in vals.items():
        if isinstance(v, list): # if the value is a list, average its elements
            out[k] = average(v)
        else:                   # otherwise, keep the value as is
            out[k] = v
    return out
    # check if dict contains lists, if it does, average them


def normalize_reference_paths(value):
    """Normalize a dataset Paths cell into a list of candidate triplet paths."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
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
        if stripped.startswith("["):
            value = ast.literal_eval(stripped)
        else:
            value = stripped.split("->")
    if isinstance(value, (list, tuple)):
        return list(value)
    return None

def normalize_answer_entities(value):
    """Normalize Answer-Entity into a set without changing entity ID types."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
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

if __name__ == '__main__':
    args = parse_args()

    if args.max_navigation_steps < 0:
        raise ValueError("--max-navigation-steps must be non-negative.")
    if args.max_questions is not None and args.max_questions < 1:
        raise ValueError("--max-questions must be positive.")

    # Define file paths
    data_dir = os.path.join(args.data_dir, args.dataset)
    qa_file = os.path.join(data_dir, f'qa_{args.hops}hop.csv')
    triplet_file = os.path.join(data_dir, 'triplets.txt')
    entity_file = os.path.join(data_dir, 'node_data.csv')
    relation_file = os.path.join(data_dir, 'relation_data.csv')

    # Load entity and relation mappings
    entity_df = load_pandas(entity_file)
    relation_df = load_pandas(relation_file)

    entity_df.set_index('QID', inplace=True)
    relation_df.set_index('Property', inplace=True)

    entity_title = entity_df['Title'].to_dict()
    relation_title = relation_df['Title'].to_dict()

    # Load all triplets and build indices
    all_triplets = load_triplets(triplet_file)
    all_triplets = set(tuple(triplet) for triplet in all_triplets.values)
    outgoing_index = build_outgoing_index(all_triplets)

    # Load QA dataset
    qa_df = load_pandas(qa_file)
    qa_df = qa_df[qa_df['SplitLabel'] == 'test']
    if args.max_questions is not None:
        qa_df = qa_df.head(args.max_questions)

    # check if answers are lists (multi-answer) or single values, and adjust accordingly
    if qa_df['Answer'].apply(lambda x: isinstance(x, str) and '[' == x[0]).all():
        qa_df['Answer'] = extract_literals(qa_df["Answer"])
        qa_df['Answer-Entity'] = extract_literals(qa_df["Answer-Entity"])

    # prepare client
    CONFIG_PATH = Path(__file__).with_name("openwebui_config.json").parent / "configs" / "openwebui_config.json"

    client = LLM_KGQA_Client(
        CONFIG_PATH,
        model_choice=args.llm_model,
        use_instruct=args.use_instruct,
        use_quantized=args.use_quantized,
        quantization_bits=args.quantization_bits,
        context_window=args.context_window,
        seed=args.seed,
        temperature=args.temperature,
        timeout=args.timeout,
        debug=args.debug
    )

    statistics = {}
    statistics['overall'] = initialize_statistics(total=len(qa_df))

    if args.hops == 'n':
        hop_size_counts = qa_df['Hops'].value_counts().to_dict()
        for hop_size, count in hop_size_counts.items():
            statistics[f'{hop_size}'] = initialize_statistics(total=count)

    navigation_metric_scores = {
        section: {"path": [], "answer": []}
        for section in statistics
    }

    total_qa = len(qa_df)
    total_batches = (total_qa + args.batch_size - 1) // args.batch_size # ceiling division

    # use tqdm to show progress bar for processing questions, go through one question at a time, no batch size
    with tqdm(range(0, total_qa), desc="Processing Questions") as pbar:
        # extract current pbar value
        i0 = pbar.n
        question = qa_df['Question'].iloc[i0]
        start_node = qa_df['Source-Entity'].iloc[i0]
        answer_node = qa_df['Answer-Entity'].iloc[i0]
        hop = qa_df['Hops'].iloc[i0]

        pred, navigation_history_txt, status_info = client.process_navigation_question(
            question=question,
            start_node=start_node,
            outgoing_index=outgoing_index,
            entity_title=entity_title,
            relation_title=relation_title,
            max_steps=args.max_navigation_steps,
            max_actions=args.max_actions,
            trace=pbar.write if args.show_navigation else None,
        )

        # TODO: Continue from here
        reference_paths = (
            normalize_reference_paths(qa_df["Paths"].iloc[i0])
            if "Paths" in qa_df.columns
            else []
        )
        relation_chain = (
            normalize_relation_chain(qa_df["Path-Key"].iloc[i0])
            if "Path-Key" in qa_df.columns
            else None
        )
        valid_answer_entities = normalize_answer_entities(
            qa_df["Answer-Entity"].iloc[i0]
        )
        path_score = best_path_fidelity_score(
            status_info["predicted_path"],
            reference_paths,
            relation_chain,
        )
        answer_entity_score = score_single_final_entity(
            status_info["final_entity"],
            valid_answer_entities,
        )
        metric_sections = ["overall"]
        if args.hops == "n":
            metric_sections.append(f"{hop}")
        for section in metric_sections:
            if path_score is not None:
                navigation_metric_scores[section]["path"].append(path_score)
            navigation_metric_scores[section]["answer"].append(answer_entity_score)

        # create a copy of the full prediction before extracting final answer
        full_pred = pred
        pred = extract_final_answer(pred, lower=False)
        result = compare_answers(pred.lower(), answer)

        update_stats(
            statistics['overall'], 
            status_info, 
            result, 
            full_pred, 
            status_info.get('navigation_steps', 0)
        )

        if args.hops == 'n':
            update_stats(
                statistics[f'{hop}'], 
                status_info, 
                result, 
                full_pred, 
                status_info.get('navigation_steps', 0)
            )

        if args.debug and not result:
            """
            When the `--debug` flag is enabled, the script provides detailed output for each question, including:
            - The question and its correct answer
            - The predicted answer and full prediction details
            - The navigation history and number of steps
            - Whether the prediction was correct
            """
            pbar.write(f"\nQuestion: {question}")
            pbar.write(f"Answer: {answer}")
            pbar.write(f"Predicted: {pred}")
            pbar.write(f"Full Prediction: {full_pred}")
            pbar.write(f"Navigation history: {navigation_history_txt}")
            pbar.write(f"Navigation steps: {status_info.get('navigation_steps', 0)}")
            pbar.write(f"Navigation status: {status_info.get('message', '')}")
            pbar.write(f"Correct: {compare_answers(pred.lower(), answer)}")
            pbar.write(f"=========")

    # Update tqdm description with current accuracy at the end of the batch
    pbar.set_description(f"Processing Batches (Accuracy: {statistics['overall']['accuracy']}/{statistics['overall']['running_count']} = {statistics['overall']['accuracy']/statistics['overall']['running_count']:.4f})")

    # Results
    """
    Calculate and display accuracy metrics for the overall dataset and individual hop sizes (if applicable).
    Results are saved in the `results/` directory with a descriptive filename.
    """
    for section, metric_values in navigation_metric_scores.items():
        statistics[section]["path_fidelity"] = aggregate_single_prediction_metrics(
            metric_values["path"]
        )
        statistics[section]["final_entity"] = aggregate_answer_metrics(
            metric_values["answer"]
        )

    statistics['overall'] = avg_dict(statistics['overall'])
    acc = statistics['overall']['accuracy']
    total = statistics['overall']['running_count']
    statistics['overall']['avg_accuracy'] = 100*acc / total if total > 0 else 0
    print(f"\nFinal Accuracy: {acc}/{total} = {statistics['overall']['avg_accuracy']:.2f}%")
    overall_path = statistics["overall"]["path_fidelity"]
    overall_entity = statistics["overall"]["final_entity"]
    print(
        "Navigation Metrics: "
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
        for hop_size in sorted(statistics.keys()):
            if hop_size == 'overall':
                continue
            statistics[hop_size] = avg_dict(statistics[hop_size])
            acc = statistics[hop_size]['accuracy']
            total = statistics[hop_size]['running_count']
            statistics[hop_size]['avg_accuracy'] = 100*acc / total if total > 0 else 0
            print(f"Hop Size {hop_size} Accuracy: {acc}/{total} = {statistics[hop_size]['avg_accuracy']:.2f}%")

    # save the results as a JSON file
    result_path = os.path.join(args.result_dir, args.dataset)
    os.makedirs(result_path, exist_ok=True)
    model_name = args.llm_model
    if args.use_instruct:
        model_name += "-instruct"
        if args.use_quantized:
            model_name += f"-q{args.quantization_bits}"
    question_limit_suffix = (
        f"_questions{len(qa_df)}" if args.max_questions is not None else ""
    )
    results_file = os.path.join(
        result_path,
        f"results_{args.hops}hop_{model_name}_navigation{args.max_navigation_steps}"
        f"{question_limit_suffix}_seed{args.seed}.json",
    )
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=4)
        print(f"Results saved to {results_file}")
