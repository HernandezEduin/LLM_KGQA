from collections import defaultdict
from random import Random
from typing import Tuple

from utils.kgqa_types import (
    EntityId,
    IncidenceIndex,
    NeighborIndex,
    OutgoingIndex,
    Path,
    PathList,
    RelationChain,
    RelationIndex,
    TripletCollection,
    TripletList,
    TripletSet,
)

def build_incidence_index(full_graph: TripletCollection) -> Tuple[IncidenceIndex, NeighborIndex]:
    """
    Build incidence and neighbor indices for the graph.

    Args:
        full_graph (TripletCollection): The complete set of triplets in the graph.

    Returns:
        Tuple[IncidenceIndex, NeighborIndex]: Incidence and neighbor indices.
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

def build_outgoing_index(
    full_graph: TripletCollection,
) -> OutgoingIndex:
    """Index directed triplets by their head entity.

    The returned lists are sorted so that action indices remain stable across
    runs. Only edges directed from head to tail are included for each entity.
    """
    outgoing = defaultdict(list)

    for head, relation, tail in full_graph:
        outgoing[head].append((head, relation, tail))

    for head in outgoing:
        outgoing[head].sort(key=lambda triplet: (triplet[1], triplet[2]))

    return outgoing

def get_outgoing_edges(
    current_entity: EntityId,
    outgoing_index: OutgoingIndex,
) -> TripletList:
    """Return the valid directed actions from ``current_entity``."""
    return outgoing_index.get(current_entity, [])

def build_relation_index(
    full_graph: TripletCollection,
) -> RelationIndex:
    """
    Builds an index for quick lookup of triplets based on their relation and head entity.

    Args:
        full_graph (TripletCollection): The complete set of triplets in the graph.
    Returns:
        RelationIndex: A nested dictionary where the first key is the relation, 
            the second key is the head entity, and the value is a list of triplets that match the relation and head.
    """
    relation_index: RelationIndex = {}

    for h, r, t in full_graph:
        relation_index.setdefault(r, {}).setdefault(h, []).append((h, r, t))

    return relation_index

def random_subgraph_sampling(
    full_graph: TripletSet,
    seeds: TripletSet,
    target_size: int,
    rng_seed: int = 42,
) -> TripletList:
    """
    Perform random subgraph sampling from the full graph.

    Args:
        full_graph (TripletSet): The complete set of triplets in the graph.
        seeds (TripletSet): Initial set of triplets to include in the subgraph.
        target_size (int): Desired size of the subgraph.
        rng_seed (int): Random seed for reproducibility.

    Returns:
        TripletList: Subgraph containing the sampled triplets.
    """
    samples_space_remaining = target_size - len(seeds)
    rng = Random(rng_seed)

    if samples_space_remaining <= 0:
        # Sort for deterministic ordering before shuffling
        sorted_seeds = sorted(list(seeds))
        rng.shuffle(sorted_seeds)
        return sorted_seeds

    # Exclude seeds from the sampling pool. Sorting first for reproducibility.
    negative_samples = sorted(list(full_graph - seeds))

    # Randomly sample the remaining triplets
    negative_sampled_triplets = set(rng.sample(negative_samples, k=samples_space_remaining))
    set_triplets =  seeds.union(negative_sampled_triplets)

    # Convert to list, sort for deterministic initial state, then shuffle
    shuffled_triplets = sorted(list(set_triplets))
    rng.shuffle(shuffled_triplets)
    return shuffled_triplets

def neighborhood_subgraph_sampling(
    full_graph: TripletSet,
    seeds: TripletSet,
    incidence: IncidenceIndex,
    neighbors: NeighborIndex,
    target_size: int,
    max_depth: int = 2,
    rng_seed: int = 0,
    fill_random_if_needed: bool = True,
) -> TripletList:
    """
    Perform neighborhood-based subgraph sampling.

    Args:
        full_graph (TripletSet): The complete set of triplets in the graph.
        seeds (TripletSet): Initial set of triplets to include in the subgraph.
        incidence (IncidenceIndex): Incidence index mapping entities to triplets.
        neighbors (NeighborIndex): Neighbor index mapping entities to neighbors.
        target_size (int): Desired size of the subgraph.
        max_depth (int): Maximum depth for neighborhood expansion.
        rng_seed (int): Random seed for reproducibility.
        fill_random_if_needed (bool): Whether to fill the subgraph randomly if under target size.

    Returns:
        TripletList: Subgraph containing the sampled triplets.
    """
    if target_size <= 0:
        return []

    rng = Random(rng_seed)
    seeds = set(seeds)
    if len(seeds) >= target_size:
        # Sort for deterministic ordering before shuffling
        sorted_seeds = sorted(list(seeds))

        rng.shuffle(sorted_seeds)
        return sorted_seeds

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
            # sort for reproducibility 
            candidate_triplets = sorted(candidate_triplets)
            rng.shuffle(candidate_triplets)
            subgraph.update(candidate_triplets[:target_size - len(subgraph)])

        visited_nodes.update(next_frontier_nodes)
        frontier = next_frontier_nodes

    # Fill the subgraph randomly if needed
    if fill_random_if_needed and len(subgraph) < target_size:
        remaining = target_size - len(subgraph)
        pool = sorted(list(full_graph - subgraph)) # sort for reproducibility
        if pool:
            subgraph.update(rng.sample(pool, k=min(remaining, len(pool))))

    # Convert to list, sort for deterministic initial state, then shuffle
    shuffled_subgraph = sorted(list(subgraph))
    rng.shuffle(shuffled_subgraph)
    return shuffled_subgraph

def neighborhood_subgraph_sampling_by_node(
    full_graph: TripletSet,
    start_node: EntityId,
    incidence: IncidenceIndex,
    neighbors: NeighborIndex,
    target_size: int,
    max_depth: int = 2,
    rng_seed: int = 0,
    fill_random_if_needed: bool = True,
) -> TripletList:
    """
    Perform neighborhood-based subgraph sampling starting from a single node entity.

    Args:
        full_graph (TripletSet): The complete set of triplets in the graph.
        start_node (EntityId): Initial node entity to seed the subgraph expansion.
        incidence (IncidenceIndex): Incidence index mapping entities to triplets.
        neighbors (NeighborIndex): Neighbor index mapping entities to neighbors.
        target_size (int): Desired size of the subgraph.
        max_depth (int): Maximum depth for neighborhood expansion.
        rng_seed (int): Random seed for reproducibility.
        fill_random_if_needed (bool): Whether to fill the subgraph randomly if under target size.

    Returns:
        TripletList: Subgraph containing the sampled triplets.
    """
    
    if target_size <= 0:
        return []

    rng = Random(rng_seed)
    subgraph = set()

    visited_nodes = {start_node}
    frontier = {start_node}

    # depth = 0 means "start_node layer"
    for depth in range(max_depth + 1):
        if len(subgraph) >= target_size or not frontier:
            break

        candidate_triplets = []
        next_frontier = set()

        # gather candidate edges from this layer
        for node in frontier:
            for tr in incidence.get(node, []):
                if tr not in subgraph:
                    candidate_triplets.append(tr)

        # capped randomized; to prevent node domination (one high degree node) ... need k_neighbors
        # for node in frontier:
        #     nb_list = [nb for nb in neighbors.get(node, set()) if nb not in visited_nodes]
        #     nb_list.sort()
        #     rng.shuffle(nb_list)

        #     for nb in nb_list[:k_neighbors]:
        #         next_frontier.add(nb)

        # sample edges from candidates
        if candidate_triplets:
            candidate_triplets = sorted(candidate_triplets)  # deterministic base
            rng.shuffle(candidate_triplets)
            need = target_size - len(subgraph)
            subgraph.update(candidate_triplets[:need])

        # prepare next layer (unless we've hit max depth)
        if depth == max_depth:
            break

        for node in frontier:
            for nb in neighbors.get(node, set()):
                if nb not in visited_nodes:
                    next_frontier.add(nb)

        visited_nodes.update(next_frontier)
        frontier = next_frontier

    # optional fill from anywhere if not max size
    if fill_random_if_needed and len(subgraph) < target_size:
        remaining = target_size - len(subgraph)
        pool = sorted(list(full_graph - subgraph))
        if pool:
            subgraph.update(rng.sample(pool, k=min(remaining, len(pool))))

    sorted_subgraph = sorted(list(subgraph))
    rng.shuffle(sorted_subgraph)
    return sorted_subgraph

def random_subgraph_sampling_by_node(
    full_graph: TripletSet,
    start_node: EntityId,
    target_size: int,
    rng_seed: int = 42,
) -> TripletList:
    """
    Perform random subgraph sampling from single node entity over the entire graph. 

    Args:
        full_graph (TripletSet): The complete set of triplets in the graph.
        start_node (EntityId): Initial node entity.
        target_size (int): Desired size of the subgraph.
        rng_seed (int): Random seed for reproducibility.

    Returns:
        TripletList: Subgraph containing the sampled triplets.
    """
    if target_size <= 0:
        return []
    
    rng = Random(rng_seed)

    random_samples = set(rng.sample(sorted(full_graph), k=target_size))

    shuffled_samples = sorted(list(random_samples))
    rng.shuffle(shuffled_samples)

    return shuffled_samples

def generate_multi_answer_paths_from_source(
    source_entity: EntityId,
    rel_list: RelationChain,
    relation_index: RelationIndex,
) -> PathList:
    """
    Generate continuous paths that start from the same source entity, follow the same
    relation sequence, and collectively yield multiple distinct final answers.

    Args:
        source_entity (EntityId): The entity to start the path from.
        rel_list (RelationChain): A list of relations that defines the required path pattern.
        relation_index (RelationIndex): A nested dictionary for quick
            lookup of triplets by relation and head entity.

    Returns:
        PathList: A list of continuous paths that begin at the given source entity
            and produce at least two distinct answers for the same relation sequence.
    """
    if not rel_list:
        return []

    candidate_paths: PathList = []

    def dfs(step_idx: int, current_entity: EntityId, current_path: Path) -> None:
        if step_idx == len(rel_list):
            candidate_paths.append(list(current_path))
            return

        next_relation = rel_list[step_idx]
        next_triplets = relation_index.get(next_relation, {}).get(current_entity, [])

        for triplet in next_triplets:
            current_path.append(triplet)
            dfs(step_idx + 1, triplet[2], current_path)
            current_path.pop()

    first_relation = rel_list[0]
    first_triplets = relation_index.get(first_relation, {}).get(source_entity, [])

    for first_triplet in first_triplets:
        dfs(1, first_triplet[2], [first_triplet])

    return candidate_paths