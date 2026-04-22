from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from utils.rag_utils import (
    Triplet,
    build_text_metadata,
    compute_idf,
    dedupe_triplets,
    flatten_unique_triplets,
    normalize_phrase,
    normalize_question_template,
    tokenize,
    weighted_overlap_score,
)


@dataclass(frozen=True)
class RetrievedPath:
    triplets: Tuple[Triplet, ...]
    node_sequence: Tuple[str, ...]
    score: float
    endpoints: Tuple[str, str]


@dataclass(frozen=True)
class PathRAGRetrievalResult:
    grouped_triplets: Tuple[Tuple[Triplet, ...], ...]
    flat_triplets: Tuple[Triplet, ...]
    retrieved_nodes: Tuple[str, ...]
    path_scores: Tuple[float, ...]


class OfflinePathRAGRetriever:
    """
    Offline adaptation of PathRAG for a local KG.

    It follows the paper's main stages:
    1. retrieve query-relevant nodes,
    2. retrieve key relational paths between those nodes with flow-based pruning,
    3. return reliability-ordered paths for prompting.

    Because this repo uses an offline symbolic KG rather than a text-indexed graph with
    embedding search, node retrieval is approximated with metadata overlap over entity
    and relation descriptions.
    """

    def __init__(
        self,
        triplets: Iterable[Triplet],
        entity_df: pd.DataFrame,
        relation_df: pd.DataFrame,
    ):
        self.triplets = dedupe_triplets(triplets)

        metadata = build_text_metadata(entity_df=entity_df, relation_df=relation_df)
        self.entity_title = metadata.entity_title
        self.entity_description = metadata.entity_description
        self.entity_alias = metadata.entity_alias
        self.relation_title = metadata.relation_title
        self.relation_description = metadata.relation_description
        self.relation_alias = metadata.relation_alias
        self.entity_tokens = metadata.entity_tokens
        self.relation_tokens = metadata.relation_tokens

        self.outgoing: Dict[str, List[Triplet]] = defaultdict(list)
        self.incoming: Dict[str, List[Triplet]] = defaultdict(list)
        self.adjacent_relations: Dict[str, set[str]] = defaultdict(set)
        for head, relation, tail in self.triplets:
            self.outgoing[head].append((head, relation, tail))
            self.incoming[tail].append((head, relation, tail))
            self.adjacent_relations[head].add(relation)
            self.adjacent_relations[tail].add(relation)

        for node in self.outgoing:
            self.outgoing[node] = sorted(self.outgoing[node])
        for node in self.incoming:
            self.incoming[node] = sorted(self.incoming[node])

        self.entity_context_tokens: Dict[str, Tuple[str, ...]] = {}
        self.entity_context_inverted: Dict[str, set[str]] = defaultdict(set)
        for entity_id in self.entity_title:
            tokens = list(self.entity_tokens.get(entity_id, ()))
            for relation_id in sorted(self.adjacent_relations.get(entity_id, ())):
                tokens.extend(self.relation_tokens.get(relation_id, ()))
            unique_tokens = tuple(sorted(set(tokens)))
            self.entity_context_tokens[entity_id] = unique_tokens
            for token in unique_tokens:
                self.entity_context_inverted[token].add(entity_id)

        text_documents: List[Sequence[str]] = list(self.entity_context_tokens.values()) + list(self.relation_tokens.values())
        self.text_idf = compute_idf(text_documents)

    def retrieve(
        self,
        question: str,
        start_node: str,
        max_hops: int = 4,
        top_nodes: int = 8,
        top_paths: int = 8,
        alpha: float = 0.8,
        threshold: float = 0.3,
        max_paths_per_pair: int = 32,
        max_branching: int = 16,
    ) -> PathRAGRetrievalResult:
        max_hops = max(1, int(max_hops))
        top_nodes = max(2, int(top_nodes))
        top_paths = max(1, int(top_paths))
        max_paths_per_pair = max(1, int(max_paths_per_pair))
        max_branching = max(1, int(max_branching))

        source_title = self.entity_title.get(start_node, start_node)
        normalized_question = normalize_question_template(question, source_title, start_node)
        query_tokens = tokenize(normalized_question)

        retrieved_nodes = self._retrieve_nodes(
            query_text=normalized_question,
            query_tokens=query_tokens,
            start_node=start_node,
            max_hops=max_hops,
            top_nodes=top_nodes,
        )

        retrieved_paths: List[RetrievedPath] = []
        for target in retrieved_nodes:
            if target == start_node:
                continue
            candidate_paths = self._enumerate_triplet_paths(
                source=start_node,
                target=target,
                query_tokens=query_tokens,
                max_hops=max_hops,
                max_paths=max_paths_per_pair,
                max_branching=max_branching,
            )
            if not candidate_paths:
                continue
            retrieved_paths.extend(
                self._score_paths_for_pair(
                    source=start_node,
                    target=target,
                    candidate_paths=candidate_paths,
                    query_tokens=query_tokens,
                    alpha=float(alpha),
                    threshold=float(threshold),
                    )
                )

        retrieved_paths.extend(
            self._fallback_paths_from_start(
                start_node=start_node,
                query_tokens=query_tokens,
                max_hops=max_hops,
                top_paths=max(top_paths * 4, 16),
                max_branching=max_branching,
            )
        )

        ranked_paths = self._dedupe_and_rank_paths(retrieved_paths, top_paths=top_paths)
        prompt_paths = sorted(
            ranked_paths,
            key=lambda path: (path.score, len(path.triplets), path.endpoints, path.triplets),
        )
        grouped_triplets = tuple(path.triplets for path in prompt_paths)
        flat_triplets = flatten_unique_triplets(grouped_triplets)

        return PathRAGRetrievalResult(
            grouped_triplets=grouped_triplets,
            flat_triplets=flat_triplets,
            retrieved_nodes=tuple(retrieved_nodes),
            path_scores=tuple(path.score for path in prompt_paths),
        )

    def _retrieve_nodes(
        self,
        query_text: str,
        query_tokens: Sequence[str],
        start_node: str,
        max_hops: int,
        top_nodes: int,
    ) -> List[str]:
        reachable_distances = self._reachable_distances(start_node, max_hops)
        candidate_nodes = set(reachable_distances.keys())
        candidate_nodes.add(start_node)

        scored_nodes = []
        for node in candidate_nodes:
            context_score = weighted_overlap_score(
                query_tokens,
                self.entity_context_tokens.get(node, ()),
                self.text_idf,
            )
            entity_score = weighted_overlap_score(
                query_tokens,
                self.entity_tokens.get(node, ()),
                self.text_idf,
            )
            distance_bonus = 0.0
            if node == start_node:
                distance_bonus = 2.0
            elif node in reachable_distances:
                distance_bonus = 0.75 / reachable_distances[node]

            phrase_bonus = self._entity_phrase_bonus(query_text, node)
            total_score = (1.5 * context_score) + (0.5 * entity_score) + distance_bonus + phrase_bonus
            if total_score <= 0 and node != start_node:
                continue
            scored_nodes.append((total_score, node))

        ranked_nodes = sorted(scored_nodes, key=lambda item: (-item[0], item[1]))
        selected_nodes = [start_node]
        selected_set = {start_node}

        for _, node in ranked_nodes:
            if node in selected_set:
                continue
            selected_nodes.append(node)
            selected_set.add(node)
            if len(selected_nodes) >= top_nodes:
                return selected_nodes

        fallback_nodes = sorted(
            (
                (distance, node)
                for node, distance in reachable_distances.items()
                if node not in selected_set
            ),
            key=lambda item: (item[0], item[1]),
        )
        for _, node in fallback_nodes:
            selected_nodes.append(node)
            if len(selected_nodes) >= top_nodes:
                break
        return selected_nodes

    def _reachable_distances(self, start_node: str, max_hops: int) -> Dict[str, int]:
        distances = {start_node: 0}
        queue = deque([(start_node, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for _, _, neighbor in self.outgoing.get(node, ()):
                if neighbor in distances:
                    continue
                distances[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))
        return distances

    def _enumerate_triplet_paths(
        self,
        source: str,
        target: str,
        query_tokens: Sequence[str],
        max_hops: int,
        max_paths: int,
        max_branching: int,
    ) -> List[Tuple[Triplet, ...]]:
        queue = deque([(source, tuple(), {source})])
        results: List[Tuple[Triplet, ...]] = []

        while queue and len(results) < max_paths:
            current_node, current_path, visited = queue.popleft()
            if len(current_path) >= max_hops:
                continue

            candidate_edges = []
            for triplet in self.outgoing.get(current_node, ()):
                next_node = triplet[2]
                if next_node in visited:
                    continue
                score = self._score_triplet(triplet, query_tokens)
                if next_node == target:
                    score += 1.0
                candidate_edges.append((score, triplet))

            candidate_edges.sort(key=lambda item: (-item[0], item[1]))
            for _, triplet in candidate_edges[:max_branching]:
                next_path = current_path + (triplet,)
                next_node = triplet[2]
                if next_node == target:
                    results.append(next_path)
                    if len(results) >= max_paths:
                        break
                else:
                    queue.append((next_node, next_path, visited | {next_node}))

        return results

    def _score_paths_for_pair(
        self,
        source: str,
        target: str,
        candidate_paths: Sequence[Tuple[Triplet, ...]],
        query_tokens: Sequence[str],
        alpha: float,
        threshold: float,
    ) -> List[RetrievedPath]:
        follow_dict: Dict[str, set[str]] = defaultdict(set)
        node_sequences = {}
        for triplet_path in candidate_paths:
            node_sequence = self._triplets_to_nodes(triplet_path)
            node_sequences[triplet_path] = node_sequence
            for current, nxt in zip(node_sequence[:-1], node_sequence[1:]):
                follow_dict[current].add(nxt)

        node_resources = {source: 1.0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            next_nodes = sorted(follow_dict.get(node, ()))
            if not next_nodes:
                continue

            propagated = alpha * node_resources[node] / len(next_nodes)
            if propagated < threshold:
                continue

            for next_node in next_nodes:
                if next_node in node_resources:
                    continue
                node_resources[next_node] = propagated
                queue.append(next_node)

        scored_paths = []
        for triplet_path in candidate_paths:
            node_sequence = node_sequences[triplet_path]
            if node_sequence[-1] != target:
                continue

            propagated_resources = []
            for node in node_sequence[1:]:
                resource = node_resources.get(node)
                if resource is None:
                    propagated_resources = []
                    break
                propagated_resources.append(resource)

            if not propagated_resources:
                continue

            flow_score = sum(propagated_resources) / len(propagated_resources)
            semantic_tiebreak = 0.001 * self._score_path_semantics(triplet_path, query_tokens)
            scored_paths.append(
                RetrievedPath(
                    triplets=triplet_path,
                    node_sequence=node_sequence,
                    score=flow_score + semantic_tiebreak,
                    endpoints=(source, target),
                )
            )
        return scored_paths

    def _fallback_paths_from_start(
        self,
        start_node: str,
        query_tokens: Sequence[str],
        max_hops: int,
        top_paths: int,
        max_branching: int,
    ) -> List[RetrievedPath]:
        queue = deque([(start_node, tuple(), {start_node})])
        candidate_paths: List[RetrievedPath] = []

        while queue:
            current_node, current_path, visited = queue.popleft()
            if len(current_path) >= max_hops:
                continue

            scored_edges = []
            for triplet in self.outgoing.get(current_node, ()):
                if triplet[2] in visited:
                    continue
                scored_edges.append((self._score_triplet(triplet, query_tokens), triplet))

            scored_edges.sort(key=lambda item: (-item[0], item[1]))
            for _, triplet in scored_edges[:max_branching]:
                next_path = current_path + (triplet,)
                next_node = triplet[2]
                path_score = self._score_path_semantics(next_path, query_tokens)
                if path_score > 0:
                    candidate_paths.append(
                        RetrievedPath(
                            triplets=next_path,
                            node_sequence=self._triplets_to_nodes(next_path),
                            score=path_score,
                            endpoints=(start_node, next_node),
                        )
                    )
                queue.append((next_node, next_path, visited | {next_node}))

        candidate_paths.sort(
            key=lambda path: (-path.score, len(path.triplets), path.endpoints, path.triplets),
        )
        return candidate_paths[:top_paths]

    def _score_path_semantics(
        self,
        triplet_path: Sequence[Triplet],
        query_tokens: Sequence[str],
    ) -> float:
        if not triplet_path:
            return 0.0

        triplet_scores = [self._score_triplet(triplet, query_tokens) for triplet in triplet_path]
        average_triplet_score = sum(triplet_scores) / len(triplet_scores)
        first_hop_score = triplet_scores[0]

        relation_tokens = []
        for _, relation, _ in triplet_path:
            relation_tokens.extend(self.relation_tokens.get(relation, ()))
        relation_coverage = weighted_overlap_score(query_tokens, relation_tokens, self.text_idf)

        return average_triplet_score + (2.0 * first_hop_score) + (0.35 * relation_coverage)

    def _score_triplet(self, triplet: Triplet, query_tokens: Sequence[str]) -> float:
        head, relation, tail = triplet
        relation_score = weighted_overlap_score(query_tokens, self.relation_tokens.get(relation, ()), self.text_idf)
        head_score = weighted_overlap_score(query_tokens, self.entity_tokens.get(head, ()), self.text_idf)
        tail_score = weighted_overlap_score(query_tokens, self.entity_tokens.get(tail, ()), self.text_idf)
        return (2.5 * relation_score) + (0.25 * head_score) + (0.75 * tail_score)

    def _entity_phrase_bonus(self, query_text: str, entity_id: str) -> float:
        normalized_query = f" {normalize_phrase(query_text)} "
        if normalized_query == "  ":
            return 0.0

        phrases = [self.entity_title.get(entity_id, "")]
        alias_text = self.entity_alias.get(entity_id, "")
        if alias_text:
            phrases.extend(alias_text.split("|"))

        total_bonus = 0.0
        for phrase in phrases:
            normalized_phrase = normalize_phrase(phrase)
            if not normalized_phrase:
                continue
            if f" {normalized_phrase} " not in normalized_query:
                continue
            token_count = len(normalized_phrase.split())
            total_bonus = max(total_bonus, 1.25 if token_count >= 2 else 0.35)
        return total_bonus

    @staticmethod
    def _triplets_to_nodes(triplet_path: Sequence[Triplet]) -> Tuple[str, ...]:
        if not triplet_path:
            return tuple()
        nodes = [triplet_path[0][0]]
        nodes.extend(triplet[2] for triplet in triplet_path)
        return tuple(nodes)

    @staticmethod
    def _dedupe_and_rank_paths(
        paths: Sequence[RetrievedPath],
        top_paths: int,
    ) -> List[RetrievedPath]:
        deduped: Dict[Tuple[Triplet, ...], RetrievedPath] = {}
        for path in paths:
            current = deduped.get(path.triplets)
            if current is None or path.score > current.score:
                deduped[path.triplets] = path

        ranked = sorted(
            deduped.values(),
            key=lambda path: (-path.score, len(path.triplets), path.endpoints, path.triplets),
        )
        return ranked[:top_paths]
