"""
This script is designed for running Knowledge Graph Question Answering (KGQA) experiments using generic Large Language Models (LLMs).
It supports subgraph sampling, evidence-based reasoning, and batch processing of questions.
This is a subgraph-at-once QA pipeline, where the entire subgraph is provided to the LLM for reasoning.
"""

import argparse
import os
from pathlib import Path

import json
import pandas as pd

from tqdm import tqdm
import warnings

from model.LLM_KGQA import LLM_KGQA_Client
from model.constants import valid_models

from utils.basic import load_triplets, load_pandas, extract_literals
from utils.kgqa_utils import compare_answers, extract_final_answer
from utils.graph_utils import (
    random_subgraph_sampling,
    neighborhood_subgraph_sampling,
    build_incidence_index,
    build_relation_index,
    neighborhood_subgraph_sampling_by_node,
    random_subgraph_sampling_by_node,
    generate_multi_answer_paths_from_source
)

from collections import defaultdict
from typing import Dict

def parse_args():
    """
    The `parse_args` function defines and parses command-line arguments for the script. These arguments include:
    - Dataset and data directory paths
    - Subgraph sampling methods and parameters
    - LLM model selection and timeout settings
    - Debugging options
    """
    parser = argparse.ArgumentParser(description="Subgraph Sampling for QA Dataset")
    
    # dataset parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path containing the dataset splits.')
    parser.add_argument('--dataset', type=str, default='mquake',
                        help='Name of the dataset to process.')
    parser.add_argument('--hops', type=str, default='n',
                        help='Number of hops for subgraph extraction.')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Number of questions to process in a batch.')

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

    # Sampling parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling.')
    parser.add_argument('--sampling-method', type=str, default='neighborhood',
                        choices=['random', 'neighborhood', 'evidence'],
                        help='Method for subgraph sampling.')
    parser.add_argument('--subgraph-size', type=int, default=50,
                        help='Number of triplets in the extracted subgraph.')
    parser.add_argument('-e','--evidence-only', action='store_true',
                        help='Whether to use evidence paths instead of the subgraph sampling.')
    parser.add_argument('--max-depth', type=int, default=3,
                        help='Maximum depth for neighborhood expansion (only for neighborhood sampling).')
    
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug mode with verbose output.')
    
    # retrieval
    parser.add_argument('-r', '--retrieve', action='store_true',
                        help='Non-oracle subgraph retrieval method.')
    
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
4. Iteratively processing batches of questions, performing subgraph sampling, and evaluating predictions.
5. Saving the results to a JSON file.
"""
def initialize_statistics(total: int) -> Dict:
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

def update_stats(
    stats_dict: Dict, 
    status_info: Dict, 
    result: str, 
    full_pred: str, 
    sub_graph_size: int,
) -> None:
    stats_dict['accuracy'] += int(result)
    stats_dict['running_count'] += 1
    stats_dict['subgraph_sizes'][sub_graph_size] += 1
    
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


if __name__ == '__main__':
    args = parse_args()

    if args.evidence_only:
        args.subgraph_size = None
        args.sampling_method = 'evidence'
        args.batch_size = 1
        warnings.warn("Using evidence paths only; overriding subgraph sampling parameters.")

    # Calculate minimum subgraph size based on batch size and hops
    if args.sampling_method != 'evidence' and args.subgraph_size is not None:
        max_hops = int(args.hops) if args.hops != 'n' else (4 if args.dataset == 'mquake' else 3)
        graph_min_size = args.batch_size * max_hops
        if args.subgraph_size < graph_min_size:
            args.batch_size = graph_min_size//max_hops
            warnings.warn(f"Subgraph size ({args.subgraph_size}) is smaller than batch_size * hops ({graph_min_size}). Reducing batch_size to {args.batch_size}.")

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
    incidence, neighbors = build_incidence_index(all_triplets)

    # Load QA dataset
    qa_df = load_pandas(qa_file)
    qa_df = qa_df[qa_df['SplitLabel'] == 'test']

    is_multi_answer = False
    # check if answers are lists (multi-answer) or single values, and adjust accordingly
    if qa_df['Answer'].apply(lambda x: isinstance(x, str) and '[' == x[0]).all():
        qa_df['Answer'] = extract_literals(qa_df["Answer"])
        qa_df['Answer-Entity'] = extract_literals(qa_df["Answer-Entity"])
        is_multi_answer = True

    # Extract triplets from paths
    is_multi_path = False
    if "Paths" in qa_df.columns:
        qa_df['Paths'] = extract_literals(qa_df["Paths"])
    elif "Path-Key" in qa_df.columns:
        relation_index = build_relation_index(all_triplets) # build relation index for evidence-based sampling
        qa_df['Path-Key'] = qa_df['Path-Key'].apply(lambda x: x.split('->')) # split the path keys into lists
        
        qa_df['Paths'] = qa_df.apply(lambda row: generate_multi_answer_paths_from_source(
            source_entity=row['Source-Entity'],
            rel_list=row['Path-Key'],
            relation_index=relation_index
        ), axis=1)

        # print(qa_df['Path-Key'].head(5))
        # print(qa_df['Paths'].head(5))
        # print(qa_df['Paths'].apply(len).head(5))
        is_multi_path = True
    else:
        raise ValueError("QA dataframe must contain either 'Paths' or 'Path-Key' column for evidence paths.")
    

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

    total_qa = len(qa_df)
    total_batches = (total_qa + args.batch_size - 1) // args.batch_size # ceiling division

    # Process QA batches with tqdm showing current accuracy
    with tqdm(range(0, total_batches), desc="Processing Batches") as pbar:
        for i0 in pbar:
            qa_batch = qa_df[i0*args.batch_size:(i0+1)*args.batch_size]         # return the last smaller batch as is, even if size < batch_size
            qa_path_batch = qa_batch['Paths']
            if is_multi_path:
                path_triplets = set(tuple(triplet) for triplet in qa_path_batch.explode().explode())
            else:
                path_triplets = set(tuple(triplet) for triplet in qa_path_batch.explode())

            # Subgraph Sampling
            """
            Three subgraph sampling methods:
            - Neighborhood Sampling: Expands the graph around specific entities up to a defined depth (includes evidence paths).
            - Random Sampling: Selects random triplets from the graph (includes evidence paths).
            - Evidence-Based Sampling: Uses predefined evidence paths.
            """
            if not args.retrieve: # oracle subgraph
                if args.sampling_method == 'neighborhood':
                    sub_graph = neighborhood_subgraph_sampling(
                        full_graph=all_triplets,
                        seeds=path_triplets,
                        incidence=incidence,
                        neighbors=neighbors,
                        target_size=args.subgraph_size,
                        max_depth=args.max_depth,
                        rng_seed=args.seed + i0,
                        fill_random_if_needed=True,
                    )
                elif args.sampling_method == 'random':
                    sub_graph = random_subgraph_sampling(
                        full_graph=all_triplets, 
                        seeds=path_triplets, 
                        target_size=args.subgraph_size,
                        rng_seed=args.seed + i0
                    )
                elif args.sampling_method == 'evidence':
                    sub_graph = path_triplets
                else:
                    raise ValueError(f"Unknown sampling method: {args.sampling_method}")

            # Q per b
            for i1 in range(len(qa_batch)):
                question = qa_batch['Question'].iloc[i1]
                start_node = qa_batch['Source-Entity'].iloc[i1]
                answer = extract_final_answer(qa_batch['Answer'].iloc[i1], lower=True)
                hop = qa_batch['Hops'].iloc[i1]

                if args.retrieve: # non-oracle subgraph retrieval
                    if args.sampling_method == 'neighborhood':
                        sub_graph = neighborhood_subgraph_sampling_by_node(
                            full_graph=all_triplets,
                            start_node=start_node,
                            incidence=incidence,
                            neighbors=neighbors,
                            target_size=args.subgraph_size,
                            max_depth=args.max_depth,
                            rng_seed=args.seed + i1,
                            fill_random_if_needed=True,
                        )
                    elif args.sampling_method == 'random':
                        sub_graph = random_subgraph_sampling_by_node(
                            full_graph=all_triplets, 
                            start_node=start_node, 
                            target_size=args.subgraph_size,
                            rng_seed=args.seed + i1
                        )
                    elif args.sampling_method == 'evidence':
                        raise ValueError("Retrieval cannot use evidence.")
                    else:
                        raise ValueError(f"Unknown sampling method: {args.sampling_method}")

                pred, sub_graph_txt, status_info = client.process_question(
                    question,
                    start_node, 
                    sub_graph,
                    entity_title, 
                    relation_title, 
                    args.seed + i0, 
                    sort_graph=not args.evidence_only
                )
                # create a copy of the full prediction before extracting final answer
                full_pred = pred
                pred = extract_final_answer(pred, lower=False)
                result = compare_answers(pred.lower(), answer)

                update_stats(
                    statistics['overall'], 
                    status_info, 
                    result, 
                    full_pred, 
                    len(sub_graph)
                )

                if args.hops == 'n':
                    update_stats(
                        statistics[f'{hop}'], 
                        status_info, 
                        result, 
                        full_pred, 
                        len(sub_graph)
                    )

                if args.debug and not result:
                    """
                    When the `--debug` flag is enabled, the script provides detailed output for each question, including:
                    - The question and its correct answer
                    - The predicted answer and full prediction details
                    - The subgraph text and size
                    - Whether the prediction was correct
                    """
                    pbar.write(f"\nQuestion: {question}")
                    pbar.write(f"Answer: {answer}")
                    pbar.write(f"Predicted: {pred}")
                    pbar.write(f"Full Prediction: {full_pred}")
                    pbar.write(f"Subgraph Text: {sub_graph_txt}")
                    pbar.write(f"Subgraph size: {len(sub_graph)} triplets")
                    pbar.write(f"Subgraph sampling method: {'retrieve' if args.retrieve else 'oracle'}, {args.sampling_method}")
                    pbar.write(f"Correct: {compare_answers(pred.lower(), answer)}")
                    pbar.write(f"=========")
            # Update tqdm description with current accuracy at the end of the batch
            pbar.set_description(f"Processing Batches (Accuracy: {statistics['overall']['accuracy']}/{statistics['overall']['running_count']} = {statistics['overall']['accuracy']/statistics['overall']['running_count']:.4f})")
    
    # Results
    """
    Calculate and display accuracy metrics for the overall dataset and individual hop sizes (if applicable).
    Results are saved in the `results/` directory with a descriptive filename.
    """
    statistics['overall'] = avg_dict(statistics['overall'])
    acc = statistics['overall']['accuracy']
    total = statistics['overall']['running_count']
    statistics['overall']['avg_accuracy'] = 100*acc / total if total > 0 else 0
    print(f"\nFinal Accuracy: {acc}/{total} = {statistics['overall']['avg_accuracy']:.2f}%")
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
    results_file = os.path.join(result_path, f"results_{args.hops}hop_{model_name}_subgraph{args.subgraph_size}_{'retrieve' if args.retrieve else 'oracle'}_{args.sampling_method}_seed{args.seed}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=4)
        print(f"Results saved to {results_file}")