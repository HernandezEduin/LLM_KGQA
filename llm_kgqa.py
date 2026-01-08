import argparse
import os
from pathlib import Path

import json
import pandas as pd

from tqdm import tqdm
import warnings

from model.LLM_KGQA import LLM_KGQA_Client

from utils.basic import load_triplets, extract_literals
from utils.kgqa_utils import extract_final_answer
from utils.graph_utils import (
    random_subgraph_sampling,
    neighborhood_subgraph_sampling,
    build_incidence_index,
)

from collections import defaultdict

def parse_args():
    """
    Parse command-line arguments for the script.
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
                        choices=['gemma3', 'llama3', 'llama3.1', 'deepseek-coder', 'qwen2.5', 'gpt-oss', 'mixtral'],
                        help='Model ID to use for the LLM API.')

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
    
    return parser.parse_args()

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
    entity_file = os.path.join(data_dir, 'vocab/entity_title.json')
    relation_file = os.path.join(data_dir, 'vocab/relation_title.json')

    # Load QA dataset
    qa_df = pd.read_csv(qa_file)
    qa_df = qa_df[qa_df['SplitLabel'] == 'test']

    # Extract triplets from paths
    df_paths = extract_literals(qa_df["Paths"])
    df_triplets = set(tuple(triplet) for triplet in df_paths.explode())
    if args.debug:
        print(f"Total unique triplets in paths: {len(df_triplets)}")

    # Load entity and relation mappings
    with open(entity_file, 'r', encoding='utf-8') as f:
        entity_title = json.load(f)

    with open(relation_file, 'r', encoding='utf-8') as f:
        relation_title = json.load(f)

    # Load all triplets and build indices
    all_triplets = load_triplets(triplet_file)
    all_triplets = set(tuple(triplet) for triplet in all_triplets.values)
    incidence, neighbors = build_incidence_index(all_triplets)

    # prepare client
    CONFIG_PATH = Path(__file__).with_name("openwebui_config.json").parent / "configs" / "openwebui_config.json"

    client = LLM_KGQA_Client(
        CONFIG_PATH,
        model_choice=args.llm_model,
        debug=args.debug
    )

    statistics = {}

    overall_stats = {
        'accuracy': 0,
        'running_count': 0,
        'total': len(qa_df),
        'subgraph_sizes': defaultdict(int),
    }
    statistics['overall'] = overall_stats

    if args.hops == 'n':
        hop_size_counts = qa_df['Hops'].value_counts().to_dict()
        for hop_size, count in hop_size_counts.items():
            statistics[f'{hop_size}'] = {
                'accuracy': 0,
                'total': count,
                'running_count': 0
            }

    total_qa = len(qa_df)
    total_batches = (total_qa + args.batch_size - 1) // args.batch_size # ceiling division

    # Process QA batches with tqdm showing current accuracy
    with tqdm(range(0, total_batches), desc="Processing Batches") as pbar:
        for i0 in pbar:
            qa_batch = qa_df[i0*args.batch_size:(i0+1)*args.batch_size]         # return the last smaller batch as is, even if size < batch_size
            qa_path_batch = df_paths[i0*args.batch_size:(i0+1)*args.batch_size]
            path_triplets = set(tuple(triplet) for triplet in qa_path_batch.explode())

            # Perform subgraph sampling
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

            for i1 in range(len(qa_batch)):
                question = qa_batch['Question'].iloc[i1]
                answer = extract_final_answer(qa_batch['Answer'].iloc[i1])
                path = qa_path_batch.iloc[i1]
                hop = qa_batch['Hops'].iloc[i1]

                pred = client.process_question(
                    question, 
                    sub_graph, 
                    entity_title, 
                    relation_title, 
                    args.seed + i0, 
                    sort_graph=not args.evidence_only
                )
                pred = extract_final_answer(pred)
                result = pred.lower() == answer.lower()
                statistics['overall']['accuracy'] += int(result)
                statistics['overall']['running_count'] += 1
                statistics['overall']['subgraph_sizes'][len(sub_graph)] += 1

                if args.hops == 'n':
                    statistics[f'{hop}']['accuracy'] += int(result)
                    statistics[f'{hop}']['running_count'] += 1

                if args.debug:
                    pbar.write(f"\nQuestion: {question}")
                    pbar.write(f"Answer: {answer}")
                    pbar.write(f"Predicted: {pred}")
                    pbar.write(f"Subgraph size: {len(sub_graph)} triplets")
                    pbar.write(f"Correct: {pred.lower() == answer.lower()}")
                    pbar.write(f"=========")
            # Update tqdm description with current accuracy at the end of the batch
            pbar.set_description(f"Processing Batches (Accuracy: {statistics['overall']['accuracy']}/{statistics['overall']['running_count']} = {statistics['overall']['accuracy']/statistics['overall']['running_count']:.4f})")

            # if args.debug:
            #     pbar.write(f"\nBatch {i0//args.batch_size + 1} completed.")

    acc = statistics['overall']['accuracy']
    total = statistics['overall']['total']
    print(f"\nFinal Accuracy: {acc}/{total} = {acc/total:.4f}")
    if args.hops == 'n':
        for hop_size in sorted(statistics.keys()):
            if hop_size == 'overall':
                continue
            acc = statistics[hop_size]['accuracy']
            total = statistics[hop_size]['total']
            print(f"Hop Size {hop_size} Accuracy: {acc}/{total} = {acc/total:.4f}")

    # save the results as a JSON file
    results_file = os.path.join('./results', f'results_{args.dataset}_{args.hops}hop_{args.llm_model}_subgraph{args.subgraph_size}_{args.sampling_method}_seed{args.seed}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=4)
        print(f"Results saved to {results_file}")