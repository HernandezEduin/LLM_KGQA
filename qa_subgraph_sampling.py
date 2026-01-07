import argparse
import os
import json
import pandas as pd

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
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path containing the dataset splits.')
    parser.add_argument('--dataset', type=str, default='mquake',
                        help='Name of the dataset to process.')
    parser.add_argument('--hops', type=str, default='n',
                        help='Number of hops for subgraph extraction.')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Number of questions to process in a batch.')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling.')
    parser.add_argument('--sampling-method', type=str, default='random',
                        choices=['random', 'neighborhood'],
                        help='Method for subgraph sampling.')
    parser.add_argument('--subgraph-size', type=int, default=50,
                        help='Number of triplets in the extracted subgraph.')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='Maximum depth for neighborhood expansion (only for neighborhood sampling).')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Define file paths
    data_dir = os.path.join(args.data_dir, args.dataset)
    qa_file = os.path.join(data_dir, f'qa_{args.hops}hop.csv')
    triplet_file = os.path.join(data_dir, 'triplets.txt')
    entity_file = os.path.join(data_dir, 'vocab/entity_title.json')
    relation_file = os.path.join(data_dir, 'vocab/relation_title.json')

    # Load QA dataset
    qa_df = pd.read_csv(qa_file)
    qa_df = qa_df[qa_df['SplitLabel'] == 'test']
    print(f"Total number of test questions in {qa_file}: {len(qa_df)}")
    print(qa_df.head(5)[['Question', 'Answer', 'Paths', 'Hops']])

    # Extract triplets from paths
    df_paths = extract_literals(qa_df["Paths"])
    df_triplets = set(tuple(triplet) for triplet in df_paths.explode())
    print(f"Total unique triplets in test questions: {len(df_triplets)}")

    # Load entity and relation mappings
    with open(entity_file, 'r', encoding='utf-8') as f:
        entity_title = json.load(f)

    with open(relation_file, 'r', encoding='utf-8') as f:
        relation_title = json.load(f)

    # Load all triplets and build indices
    all_triplets = load_triplets(triplet_file)
    all_triplets = set(tuple(triplet) for triplet in all_triplets.values)
    incidence, neighbors = build_incidence_index(all_triplets)

    # Process QA batches
    for i0 in range(0, len(qa_df), args.batch_size):
        qa_batch = qa_df[i0:i0+args.batch_size]
        qa_path_batch = df_paths[i0:i0+args.batch_size]

        question = qa_batch['Question'].iloc[0]
        answer = qa_batch['Answer'].iloc[0]
        path = qa_path_batch.iloc[0]

        path_triplets = set(tuple(triplet) for triplet in qa_path_batch.explode())

        print(f"Batch {i0//args.batch_size + 1}:")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"P: {translate_path(path, entity_title, relation_title)}")
        print(f"Path triplets in Batch ({len(path_triplets)}): {path_triplets}\n")

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
        else:
            raise ValueError(f"Unknown sampling method: {args.sampling_method}")

        print(f"Subgraph triplets in Batch ({len(sub_graph)}): {sub_graph}\n")
        print(f"Readable Subgraph triplets in Batch: {translate_path(list(sub_graph), entity_title, relation_title)}\n")
        break