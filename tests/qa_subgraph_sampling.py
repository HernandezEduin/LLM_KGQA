import argparse
import os
import json
import time
import pandas as pd

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from utils.basic import load_triplets, extract_literals
from utils.kgqa_utils import translate_path
from utils.graph_utils import (
    random_subgraph_sampling,
    neighborhood_subgraph_sampling,
    build_incidence_index,
    random_subgraph_sampling_by_node,
    neighborhood_subgraph_sampling_by_node
)

from typing import List, Tuple
from collections import defaultdict
from collections import deque

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
    parser.add_argument('--sampling-method', type=str, default='neighborhood',
                        choices=['random', 'neighborhood'],
                        help='Method for subgraph sampling.')
    parser.add_argument('--subgraph-size', type=int, default=50,
                        help='Number of triplets in the extracted subgraph.')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='Maximum depth for neighborhood expansion (only for neighborhood sampling).')
    parser.add_argument('-r', '--retrieve', action='store_true',
                         help='Non-oracle subgraph retrieval method.')
    parser.add_argument('--max_depth', type=int, default=2,
                        help='MAX Depth for expansion from starting node/triplet in neighborhood subsampmling.')
    
    return parser.parse_args()

def count_entity_distances_from_start(start_node, triplets):
    """
    Compute shortest hop distance from start_node to each entity that appears in triplets.

    Treats each triplet as an undirected edge between head and tail.
    Returns a dict of entity -> distance (0 for start_node).
    """
    # Build an undirected adjacency list from the subgraph triplets
    adj = defaultdict(set)
    for h, _, t in triplets:
        adj[h].add(t)
        adj[t].add(h)

    distances = {start_node: 0}
    queue = deque([start_node])

    while queue:
        node = queue.popleft()
        for nb in adj.get(node, set()):
            if nb not in distances:
                distances[nb] = distances[node] + 1
                queue.append(nb)

    return distances

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
    #print(qa_df.head(5)[['Question', 'Answer', 'Paths', 'Hops']])

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
        qa_batch = qa_df[i0*args.batch_size:(i0+1)*args.batch_size]         # return the last smaller batch as is, even if size < batch_size
        qa_path_batch = df_paths[i0*args.batch_size:(i0+1)*args.batch_size]
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
                print("In Oracle:neighborhood")
                t0 = time.perf_counter()
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
                print(f"Subgraph sampling wall time: {time.perf_counter() - t0:.6f}s")
            elif args.sampling_method == 'random':
                print("In oracle:random")
                t0 = time.perf_counter()
                sub_graph = random_subgraph_sampling(
                    full_graph=all_triplets, 
                    seeds=path_triplets, 
                    target_size=args.subgraph_size,
                    rng_seed=args.seed + i0
                )
                print(f"Subgraph sampling wall time: {time.perf_counter() - t0:.6f}s")
            elif args.sampling_method == 'evidence':
                sub_graph = path_triplets
            else:
                raise ValueError(f"Unknown sampling method: {args.sampling_method}")

        # Q per b
        for i1 in range(len(qa_batch)):
            question = qa_batch['Question'].iloc[i1]
            start_node = qa_batch['Source-Entity'].iloc[i1]
            #answer = extract_final_answer(qa_batch['Answer'].iloc[i1])
            path = qa_path_batch.iloc[i1]
            hop = qa_batch['Hops'].iloc[i1]
            
            if args.retrieve: # non-oracle subgraph retrieval
                if args.sampling_method == 'neighborhood':
                    print("In retrieve:neighborhood")
                    t0 = time.perf_counter()
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
                    print(f"Subgraph sampling wall time: {time.perf_counter() - t0:.6f}s")
                elif args.sampling_method == 'random':
                    print("In retrieve:random")
                    t0 = time.perf_counter()
                    sub_graph = random_subgraph_sampling_by_node(
                        full_graph=all_triplets, 
                        start_node=start_node, 
                        target_size=args.subgraph_size,
                        rng_seed=args.seed + i1
                    )
                    print(f"Subgraph sampling wall time: {time.perf_counter() - t0:.6f}s")
                elif args.sampling_method == 'evidence':
                    raise ValueError("Retrieval cannot use evidence.")
                else:
                    raise ValueError(f"Unknown sampling method: {args.sampling_method}")
            print(f"Starting Node: {start_node} -> {entity_title[start_node]}")
            print(f"Subgraph: {translate_path(sub_graph, entity_title, relation_title)}")
            distances = count_entity_distances_from_start(start_node, sub_graph)
            print(f"Distance of each entity from start node: {distances}")
            [print(f"Max depth borken at {ent}!\n") for ent,dis in distances.items() if dis > args.max_depth]
            print(f"\n\n")        

        #print(f"Subgraph triplets in Batch ({len(sub_graph)}): {sub_graph}\n")
        #print(f"Readable Subgraph triplets in Batch: {translate_path(sub_graph, entity_title, relation_title)}\n")
        break
