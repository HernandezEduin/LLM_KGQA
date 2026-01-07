from collections import defaultdict
from random import Random

def build_incidence_index(full_graph: set):
    """
    Build incidence and neighbor indices for the graph.

    Args:
        full_graph (set): The complete set of triplets in the graph.

    Returns:
        tuple: Incidence index and neighbor index.
    """
    incidence = defaultdict(list)
    neighbors = defaultdict(set)

    for h, r, t in full_graph:
        # Add triplet to incidence lists (head and tail)
        incidence[h].append((h, r, t))
        incidence[t].append((h, r, t))

        # Add undirected neighbor relations
        neighbors[h].add(t)
        neighbors[t].add(h)

    return incidence, neighbors

def random_subgraph_sampling(full_graph: set, seeds: set, target_size: int, rng_seed: int = 42) -> set:
    """
    Perform random subgraph sampling from the full graph.

    Args:
        full_graph (set): The complete set of triplets in the graph.
        seeds (set): Initial set of triplets to include in the subgraph.
        target_size (int): Desired size of the subgraph.
        rng_seed (int): Random seed for reproducibility.

    Returns:
        set: Subgraph containing the sampled triplets.
    """
    samples_space_remaining = target_size - len(seeds)
    rng = Random(rng_seed)

    if samples_space_remaining <= 0:
        return seeds

    # Exclude seeds from the sampling pool
    negative_samples = list(full_graph - seeds)

    # Randomly sample the remaining triplets
    negative_sampled_triplets = set(rng.sample(negative_samples, k=samples_space_remaining))
    return seeds.union(negative_sampled_triplets)

def neighborhood_subgraph_sampling(
    full_graph: set,
    seeds: set,
    incidence,
    neighbors,
    target_size: int,
    max_depth: int = 2,
    rng_seed: int = 0,
    fill_random_if_needed: bool = True,
):
    """
    Perform neighborhood-based subgraph sampling.

    Args:
        full_graph (set): The complete set of triplets in the graph.
        seeds (set): Initial set of triplets to include in the subgraph.
        incidence (dict): Incidence index mapping entities to triplets.
        neighbors (dict): Neighbor index mapping entities to neighbors.
        target_size (int): Desired size of the subgraph.
        max_depth (int): Maximum depth for neighborhood expansion.
        rng_seed (int): Random seed for reproducibility.
        fill_random_if_needed (bool): Whether to fill the subgraph randomly if under target size.

    Returns:
        set: Subgraph containing the sampled triplets.
    """
    if target_size <= 0:
        return set()

    seeds = set(seeds)
    if len(seeds) >= target_size:
        return seeds

    rng = Random(rng_seed)
    subgraph = set(seeds)

    # Initialize the frontier with entities from the seeds
    start_nodes = {node for triplet in seeds for node in (triplet[0], triplet[2])}
    visited_nodes = set(start_nodes)
    frontier = set(start_nodes)

    # Perform BFS-like expansion
    for depth in range(max_depth):
        if len(subgraph) >= target_size or not frontier:
            break

        candidate_triplets = []
        next_frontier_nodes = set()

        for node in frontier:
            # Collect incident triplets for the current node
            candidate_triplets.extend(trip for trip in incidence.get(node, []) if trip not in subgraph)

            # Collect neighbors for the next frontier
            next_frontier_nodes.update(nb for nb in neighbors.get(node, set()) if nb not in visited_nodes)

        # Shuffle and add triplets to the subgraph
        if candidate_triplets:
            rng.shuffle(candidate_triplets)
            subgraph.update(candidate_triplets[:target_size - len(subgraph)])

        visited_nodes.update(next_frontier_nodes)
        frontier = next_frontier_nodes

    # Fill the subgraph randomly if needed
    if fill_random_if_needed and len(subgraph) < target_size:
        remaining = target_size - len(subgraph)
        pool = list(full_graph - subgraph)
        if pool:
            subgraph.update(rng.sample(pool, k=min(remaining, len(pool))))

    return subgraph