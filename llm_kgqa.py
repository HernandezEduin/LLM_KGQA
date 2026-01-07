import argparse
import os
from pathlib import Path

import json
import pandas as pd

from tqdm import tqdm

from model.LLM_KGQA import LLM_KGQA_Client

from utils.basic import load_triplets, extract_literals
from utils.path_utils import translate_path
from utils.graph_utils import (
    random_subgraph_sampling,
    neighborhood_subgraph_sampling,
    build_incidence_index,
)

from typing import List, Tuple
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
                        choices=['gemma3', 'llama3.1', 'llama3.1', 'deepseek-coder', 'qwen2.5', 'gpt-oss', 'mixtral'],
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
    parser.add_argument('--max-depth', type=int, default=2,
                        help='Maximum depth for neighborhood expansion (only for neighborhood sampling).')
    
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug mode with verbose output.')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.use_evidence_only:
        print("Using evidence paths only; overriding subgraph sampling parameters.")
        args.subgraph_size = None
        args.sampling_method = 'evidence'
        args.batch_size = 1  # process one QA at a time when using evidence paths

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

    # TODO: Ensure all the QA are evalutated (total == len(qa_df) at the end)
    # TODO: Record LLM Model vs Hop Size (2-4, n) vs Sampling Method vs Subgraph Size (10, 50, 100, 500, 1000, onwards) results
    accuracy = 0
    total = 0
    # Process QA batches with tqdm showing current accuracy
    with tqdm(range(0, len(qa_df), args.batch_size), desc="Processing Batches") as pbar:
        for i0 in pbar:
            qa_batch = qa_df[i0:i0+args.batch_size]
            qa_path_batch = df_paths[i0:i0+args.batch_size]
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
                answer = qa_batch['Answer'].iloc[i1]
                path = qa_path_batch.iloc[i1]

                pred = client.process_question(question, sub_graph, entity_title, relation_title)
                accuracy += int(pred.strip().lower() == answer.strip().lower())
                total += 1

                if args.debug:
                    pbar.write(f"\nQuestion: {question}")
                    pbar.write(f"Answer: {answer}")
                    pbar.write(f"Predicted: {pred}")
                    pbar.write(f"Subgraph size: {len(sub_graph)} triplets")
                    pbar.write(f"Correct: {pred.strip().lower() == answer.strip().lower()}")
                    pbar.write(f"=========")
            # Update tqdm description with current accuracy at the end of the batch
            pbar.set_description(f"Processing Batches (Accuracy: {accuracy}/{total} = {accuracy/total:.4f})")

            # if args.debug:
            #     pbar.write(f"\nBatch {i0//args.batch_size + 1} completed.")

    print(f"\nFinal Accuracy: {accuracy}/{total} = {accuracy/total:.4f}")