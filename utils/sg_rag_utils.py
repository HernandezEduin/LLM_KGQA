from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

Triplet = Tuple[str, str, str]

TOKEN_RE = re.compile(r"[a-z0-9]+")
QID_RE = re.compile(r"\bq\d+\b", flags=re.IGNORECASE)
PARENS_QID_RE = re.compile(r"\(\s*Q\d+\s*\)", flags=re.IGNORECASE)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "does",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "name",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}


@dataclass(frozen=True)
class QueryPattern:
    relations: Tuple[str, ...]
    score: float


@dataclass(frozen=True)
class RetrievedSubgraph:
    triplets: Tuple[Triplet, ...]
    query_pattern: Tuple[str, ...]
    score: float
    source: str


@dataclass(frozen=True)
class SGRAGRetrievalResult:
    grouped_triplets: Tuple[Tuple[Triplet, ...], ...]
    flat_triplets: Tuple[Triplet, ...]
    candidate_query_patterns: Tuple[Tuple[str, ...], ...]


def _split_aliases(alias: str) -> str:
    if not alias:
        return ""
    return alias.replace("|", " ")


def _combine_metadata_text(title: str, description: str, alias: str) -> str:
    parts = [title or "", description or "", _split_aliases(alias)]
    return " ".join(part.strip() for part in parts if part and str(part).strip())


def _tokenize(text: str) -> Tuple[str, ...]:
    tokens = [
        token
        for token in TOKEN_RE.findall((text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    ]
    return tuple(tokens)


def _compute_idf(documents: Iterable[Sequence[str]]) -> Dict[str, float]:
    documents = [set(doc) for doc in documents if doc]
    total_docs = len(documents)
    if total_docs == 0:
        return {}

    doc_freq: Counter[str] = Counter()
    for doc in documents:
        doc_freq.update(doc)

    return {
        token: math.log((1 + total_docs) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }


def _weighted_overlap_score(
    query_tokens: Sequence[str],
    doc_tokens: Sequence[str],
    idf: Dict[str, float],
) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    overlap = query_set.intersection(doc_set)
    if not overlap:
        return 0.0

    numerator = sum(idf.get(token, 1.0) for token in overlap)
    denominator = math.sqrt(sum(idf.get(token, 1.0) for token in doc_set))
    if denominator == 0:
        return numerator
    return numerator / denominator


def _normalize_question_template(
    question: str,
    source_title: Optional[str] = None,
    source_id: Optional[str] = None,
) -> str:
    normalized = (question or "").strip()
    normalized = PARENS_QID_RE.sub(" ", normalized)
    normalized = QID_RE.sub(" ", normalized)

    if source_title:
        normalized = re.sub(
            re.escape(source_title),
            " SOURCE ",
            normalized,
            flags=re.IGNORECASE,
        )

    if source_id:
        normalized = re.sub(
            re.escape(source_id),
            " SOURCE_ID ",
            normalized,
            flags=re.IGNORECASE,
        )

    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _normalize_phrase(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class OfflineSGRAGRetriever:
    """
    Offline approximation of SG-RAG.

    The retriever:
    1. infers candidate graph query patterns from the question and KG schema metadata,
    2. executes each pattern over the offline KG,
    3. returns the matched records as grouped subgraphs.

    This follows the SG-RAG core idea of query -> matched subgraphs -> grouped triplets,
    while remaining executable without a live Cypher backend.
    """

    def __init__(
        self,
        triplets: Iterable[Triplet],
        entity_df: pd.DataFrame,
        relation_df: pd.DataFrame,
    ):
        self.triplets = tuple(sorted({tuple(triplet) for triplet in triplets}))

        entity_frame = entity_df.copy()
        relation_frame = relation_df.copy()

        if "QID" not in entity_frame.columns:
            entity_frame = entity_frame.reset_index()
        if "Property" not in relation_frame.columns:
            relation_frame = relation_frame.reset_index()

        entity_frame = entity_frame.fillna("")
        relation_frame = relation_frame.fillna("")

        self.entity_title = dict(zip(entity_frame["QID"], entity_frame["Title"]))
        self.entity_description = dict(zip(entity_frame["QID"], entity_frame["Description"]))
        self.entity_alias = dict(zip(entity_frame["QID"], entity_frame["Alias"]))

        self.relation_title = dict(zip(relation_frame["Property"], relation_frame["Title"]))
        self.relation_description = dict(zip(relation_frame["Property"], relation_frame["Description"]))
        self.relation_alias = dict(zip(relation_frame["Property"], relation_frame["Alias"]))

        self.entity_tokens = {
            entity_id: _tokenize(
                _combine_metadata_text(
                    self.entity_title.get(entity_id, entity_id),
                    self.entity_description.get(entity_id, ""),
                    self.entity_alias.get(entity_id, ""),
                )
            )
            for entity_id in self.entity_title
        }
        self.relation_tokens = {
            relation_id: _tokenize(
                _combine_metadata_text(
                    self.relation_title.get(relation_id, relation_id),
                    self.relation_description.get(relation_id, ""),
                    self.relation_alias.get(relation_id, ""),
                )
            )
            for relation_id in self.relation_title
        }

        text_documents: List[Sequence[str]] = list(self.entity_tokens.values()) + list(self.relation_tokens.values())
        self.text_idf = _compute_idf(text_documents)

        self.outgoing: Dict[str, List[Triplet]] = defaultdict(list)
        self.outgoing_by_relation: Dict[str, Dict[str, List[Triplet]]] = defaultdict(lambda: defaultdict(list))
        for head, relation, tail in self.triplets:
            self.outgoing[head].append((head, relation, tail))
            self.outgoing_by_relation[head][relation].append((head, relation, tail))

        for head in self.outgoing:
            self.outgoing[head] = sorted(self.outgoing[head])
            for relation in self.outgoing_by_relation[head]:
                self.outgoing_by_relation[head][relation] = sorted(self.outgoing_by_relation[head][relation])

    def retrieve(
        self,
        question: str,
        start_node: str,
        max_hops: int,
        min_hops: int = 1,
        top_query_patterns: int = 5,
        top_subgraphs: int = 8,
        beam_width: int = 24,
        max_branching: int = 16,
    ) -> SGRAGRetrievalResult:
        max_hops = max(1, int(max_hops))
        min_hops = max(1, min(int(min_hops), max_hops))

        source_title = self.entity_title.get(start_node, start_node)
        normalized_question = _normalize_question_template(question, source_title, start_node)
        query_tokens = _tokenize(normalized_question)

        candidate_query_patterns = self._infer_query_patterns(
            query_text=normalized_question,
            query_tokens=query_tokens,
            start_node=start_node,
            min_hops=min_hops,
            max_hops=max_hops,
            top_query_patterns=top_query_patterns,
            beam_width=beam_width,
            max_branching=max_branching,
        )

        max_matches_per_pattern = max(top_subgraphs * 8, 32)
        retrieved_subgraphs: List[RetrievedSubgraph] = []
        for query_pattern in candidate_query_patterns:
            retrieved_subgraphs.extend(
                self._execute_query_pattern(
                    start_node=start_node,
                    query_pattern=query_pattern,
                    query_tokens=query_tokens,
                    max_matches=max_matches_per_pattern,
                )
            )

        deduped_subgraphs = self._dedupe_and_rank_subgraphs(
            subgraphs=retrieved_subgraphs,
            top_subgraphs=top_subgraphs,
        )
        grouped_triplets = tuple(subgraph.triplets for subgraph in deduped_subgraphs)
        flat_triplets = self._flatten_unique(grouped_triplets)
        candidate_patterns = tuple(pattern.relations for pattern in candidate_query_patterns)

        return SGRAGRetrievalResult(
            grouped_triplets=grouped_triplets,
            flat_triplets=flat_triplets,
            candidate_query_patterns=candidate_patterns,
        )

    def _infer_query_patterns(
        self,
        query_text: str,
        query_tokens: Sequence[str],
        start_node: str,
        min_hops: int,
        max_hops: int,
        top_query_patterns: int,
        beam_width: int,
        max_branching: int,
    ) -> List[QueryPattern]:
        pattern_scores: Dict[Tuple[str, ...], float] = defaultdict(float)

        for hop_count in range(min_hops, max_hops + 1):
            patterns = self._search_query_patterns(
                start_node=start_node,
                hop_budget=hop_count,
                query_text=query_text,
                query_tokens=query_tokens,
                beam_width=max(beam_width, top_query_patterns * 4),
                max_branching=max_branching,
            )
            for pattern in patterns[:top_query_patterns]:
                pattern_scores[pattern.relations] = max(pattern_scores[pattern.relations], pattern.score)

        ranked_patterns = sorted(
            (
                QueryPattern(relations=relations, score=score)
                for relations, score in pattern_scores.items()
            ),
            key=lambda pattern: (-pattern.score, len(pattern.relations), pattern.relations),
        )
        return ranked_patterns[:top_query_patterns]

    def _search_query_patterns(
        self,
        start_node: str,
        hop_budget: int,
        query_text: str,
        query_tokens: Sequence[str],
        beam_width: int,
        max_branching: int,
    ) -> List[QueryPattern]:
        if hop_budget <= 0:
            return []

        # Search over relation sequences, but treat them as candidate graph query patterns.
        states: List[Tuple[Tuple[str, ...], str, float]] = [((), start_node, 0.0)]

        for _ in range(hop_budget):
            next_states: List[Tuple[Tuple[str, ...], str, float]] = []
            for relation_pattern, current_entity, score in states:
                candidate_edges = self.outgoing.get(current_entity, [])
                scored_edges = []
                for triplet in candidate_edges:
                    edge_score = self._score_triplet(triplet, query_tokens)
                    scored_edges.append((score + edge_score, triplet))

                scored_edges.sort(key=lambda item: (-item[0], item[1]))
                for next_score, triplet in scored_edges[:max_branching]:
                    next_states.append(
                        (
                            relation_pattern + (triplet[1],),
                            triplet[2],
                            next_score,
                        )
                    )

            next_states.sort(key=lambda item: (-item[2], item[0], item[1]))
            states = next_states[:beam_width]
            if not states:
                break

        ranked_patterns: Dict[Tuple[str, ...], float] = defaultdict(float)
        for relation_pattern, _, score in states:
            pattern_score = self._score_query_pattern(
                query_text=query_text,
                query_tokens=query_tokens,
                relation_pattern=relation_pattern,
                traversal_score=score,
            )
            ranked_patterns[relation_pattern] = max(ranked_patterns[relation_pattern], pattern_score)

        return sorted(
            (
                QueryPattern(relations=relations, score=score)
                for relations, score in ranked_patterns.items()
            ),
            key=lambda pattern: (-pattern.score, pattern.relations),
        )

    def _execute_query_pattern(
        self,
        start_node: str,
        query_pattern: QueryPattern,
        query_tokens: Sequence[str],
        max_matches: int,
    ) -> List[RetrievedSubgraph]:
        relations = query_pattern.relations
        if not relations:
            return []

        states: List[Tuple[Tuple[Triplet, ...], str, float]] = [((), start_node, 0.0)]

        for relation in relations:
            next_states: List[Tuple[Tuple[Triplet, ...], str, float]] = []
            for matched_triplets, current_entity, partial_score in states:
                candidate_edges = self.outgoing_by_relation.get(current_entity, {}).get(relation, [])
                for triplet in candidate_edges:
                    edge_score = self._score_triplet(triplet, query_tokens)
                    next_states.append(
                        (
                            matched_triplets + (triplet,),
                            triplet[2],
                            partial_score + edge_score,
                        )
                    )

            if not next_states:
                return []

            # Keep the execution broad enough to resemble query matching, while
            # still bounded for offline use.
            next_states.sort(
                key=lambda state: (
                    -self._score_partial_record(state[0], query_tokens),
                    state[0],
                    state[1],
                )
            )
            states = next_states[:max_matches]

        matched_subgraphs = []
        for matched_triplets, _, partial_score in states:
            if len(matched_triplets) != len(relations):
                continue
            subgraph_triplets = self._materialize_subgraph_record(matched_triplets)
            final_score = query_pattern.score + partial_score / max(len(relations), 1)
            final_score += self._score_subgraph_record(subgraph_triplets, query_tokens)
            matched_subgraphs.append(
                RetrievedSubgraph(
                    triplets=subgraph_triplets,
                    query_pattern=relations,
                    score=final_score,
                    source="query_pattern",
                )
            )

        return matched_subgraphs

    def _score_query_pattern(
        self,
        query_text: str,
        query_tokens: Sequence[str],
        relation_pattern: Sequence[str],
        traversal_score: float,
    ) -> float:
        pattern_tokens = self._relation_pattern_tokens(relation_pattern)
        relation_coverage = _weighted_overlap_score(query_tokens, pattern_tokens, self.text_idf)
        phrase_bonus = self._relation_pattern_phrase_bonus(query_text, relation_pattern)
        return (traversal_score / max(len(relation_pattern), 1)) + (1.5 * relation_coverage) + phrase_bonus

    def _score_partial_record(
        self,
        partial_triplets: Sequence[Triplet],
        query_tokens: Sequence[str],
    ) -> float:
        if not partial_triplets:
            return 0.0

        relation_tokens: List[str] = []
        entity_tokens: List[str] = []
        for head, relation, tail in partial_triplets:
            relation_tokens.extend(self.relation_tokens.get(relation, ()))
            entity_tokens.extend(self.entity_tokens.get(head, ()))
            entity_tokens.extend(self.entity_tokens.get(tail, ()))

        relation_score = _weighted_overlap_score(query_tokens, relation_tokens, self.text_idf)
        entity_score = _weighted_overlap_score(query_tokens, entity_tokens, self.text_idf)
        return relation_score + (0.5 * entity_score)

    def _score_subgraph_record(
        self,
        triplets: Sequence[Triplet],
        query_tokens: Sequence[str],
    ) -> float:
        relation_tokens: List[str] = []
        entity_tokens: List[str] = []
        nodes = set()

        for head, relation, tail in triplets:
            relation_tokens.extend(self.relation_tokens.get(relation, ()))
            entity_tokens.extend(self.entity_tokens.get(head, ()))
            entity_tokens.extend(self.entity_tokens.get(tail, ()))
            nodes.update((head, tail))

        relation_coverage = _weighted_overlap_score(query_tokens, relation_tokens, self.text_idf)
        entity_coverage = _weighted_overlap_score(query_tokens, entity_tokens, self.text_idf)
        connectivity_bonus = 0.05 * len(nodes)
        return (1.5 * relation_coverage) + (0.5 * entity_coverage) + connectivity_bonus

    def _score_triplet(self, triplet: Triplet, query_tokens: Sequence[str]) -> float:
        head, relation, tail = triplet
        relation_score = _weighted_overlap_score(query_tokens, self.relation_tokens.get(relation, ()), self.text_idf)
        head_score = _weighted_overlap_score(query_tokens, self.entity_tokens.get(head, ()), self.text_idf)
        tail_score = _weighted_overlap_score(query_tokens, self.entity_tokens.get(tail, ()), self.text_idf)
        return (2.5 * relation_score) + (0.25 * head_score) + (0.75 * tail_score)

    def _relation_pattern_tokens(self, relation_pattern: Sequence[str]) -> Tuple[str, ...]:
        tokens: List[str] = []
        for relation in relation_pattern:
            tokens.extend(self.relation_tokens.get(relation, ()))
        return tuple(tokens)

    def _relation_pattern_phrase_bonus(self, query_text: str, relation_pattern: Sequence[str]) -> float:
        normalized_query = f" {_normalize_phrase(query_text)} "
        if normalized_query == "  ":
            return 0.0

        total_bonus = 0.0
        for relation in relation_pattern:
            phrases = [self.relation_title.get(relation, "")]
            alias_text = self.relation_alias.get(relation, "")
            if alias_text:
                phrases.extend(alias_text.split("|"))

            relation_bonus = 0.0
            for phrase in phrases:
                normalized_phrase = _normalize_phrase(phrase)
                if not normalized_phrase:
                    continue
                if f" {normalized_phrase} " not in normalized_query:
                    continue
                token_count = len(normalized_phrase.split())
                relation_bonus = max(relation_bonus, 1.25 if token_count >= 2 else 0.35)

            total_bonus += relation_bonus

        return total_bonus

    @staticmethod
    def _materialize_subgraph_record(matched_triplets: Sequence[Triplet]) -> Tuple[Triplet, ...]:
        # Each record returned by executing a query pattern is a connected subgraph.
        unique_triplets = []
        seen = set()
        for triplet in matched_triplets:
            if triplet in seen:
                continue
            seen.add(triplet)
            unique_triplets.append(triplet)
        return tuple(unique_triplets)

    @staticmethod
    def _flatten_unique(grouped_triplets: Sequence[Sequence[Triplet]]) -> Tuple[Triplet, ...]:
        flattened: List[Triplet] = []
        seen = set()
        for subgraph in grouped_triplets:
            for triplet in subgraph:
                if triplet in seen:
                    continue
                seen.add(triplet)
                flattened.append(triplet)
        return tuple(flattened)

    @staticmethod
    def _dedupe_and_rank_subgraphs(
        subgraphs: Sequence[RetrievedSubgraph],
        top_subgraphs: int,
    ) -> List[RetrievedSubgraph]:
        deduped: Dict[Tuple[Triplet, ...], RetrievedSubgraph] = {}
        for subgraph in subgraphs:
            key = tuple(subgraph.triplets)
            current = deduped.get(key)
            if current is None or subgraph.score > current.score:
                deduped[key] = subgraph

        ranked = sorted(
            deduped.values(),
            key=lambda subgraph: (
                -subgraph.score,
                len(subgraph.triplets),
                subgraph.query_pattern,
                subgraph.triplets,
            ),
        )
        return ranked[:top_subgraphs]
