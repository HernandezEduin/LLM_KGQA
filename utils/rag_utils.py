from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

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
class KGTextMetadata:
    entity_title: Dict[str, str]
    entity_description: Dict[str, str]
    entity_alias: Dict[str, str]
    relation_title: Dict[str, str]
    relation_description: Dict[str, str]
    relation_alias: Dict[str, str]
    entity_tokens: Dict[str, Tuple[str, ...]]
    relation_tokens: Dict[str, Tuple[str, ...]]


def dedupe_triplets(triplets: Iterable[Triplet]) -> Tuple[Triplet, ...]:
    return tuple(sorted({tuple(triplet) for triplet in triplets}))


def split_aliases(alias: str) -> str:
    if not alias:
        return ""
    return alias.replace("|", " ")


def combine_metadata_text(title: str, description: str, alias: str) -> str:
    parts = [title or "", description or "", split_aliases(alias)]
    return " ".join(part.strip() for part in parts if part and str(part).strip())


def tokenize(text: str) -> Tuple[str, ...]:
    tokens = [
        token
        for token in TOKEN_RE.findall((text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    ]
    return tuple(tokens)


def compute_idf(documents: Iterable[Sequence[str]]) -> Dict[str, float]:
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


def weighted_overlap_score(
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


def normalize_question_template(
    question: str,
    source_title: str | None = None,
    source_id: str | None = None,
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


def normalize_phrase(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def dedupe_preserve_order(triplets: Sequence[Triplet]) -> Tuple[Triplet, ...]:
    unique_triplets = []
    seen = set()
    for triplet in triplets:
        if triplet in seen:
            continue
        seen.add(triplet)
        unique_triplets.append(triplet)
    return tuple(unique_triplets)


def flatten_unique_triplets(grouped_triplets: Sequence[Sequence[Triplet]]) -> Tuple[Triplet, ...]:
    flattened = []
    seen = set()
    for group in grouped_triplets:
        for triplet in group:
            if triplet in seen:
                continue
            seen.add(triplet)
            flattened.append(triplet)
    return tuple(flattened)


def prepare_metadata_frames(
    entity_df: pd.DataFrame,
    relation_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    entity_frame = entity_df.copy()
    relation_frame = relation_df.copy()

    if "QID" not in entity_frame.columns:
        entity_frame = entity_frame.reset_index()
    if "Property" not in relation_frame.columns:
        relation_frame = relation_frame.reset_index()

    entity_frame = entity_frame.fillna("")
    relation_frame = relation_frame.fillna("")
    return entity_frame, relation_frame


def build_text_metadata(
    entity_df: pd.DataFrame,
    relation_df: pd.DataFrame,
) -> KGTextMetadata:
    entity_frame, relation_frame = prepare_metadata_frames(entity_df, relation_df)

    entity_title = dict(zip(entity_frame["QID"], entity_frame["Title"]))
    entity_description = dict(zip(entity_frame["QID"], entity_frame["Description"]))
    entity_alias = dict(zip(entity_frame["QID"], entity_frame["Alias"]))

    relation_title = dict(zip(relation_frame["Property"], relation_frame["Title"]))
    relation_description = dict(zip(relation_frame["Property"], relation_frame["Description"]))
    relation_alias = dict(zip(relation_frame["Property"], relation_frame["Alias"]))

    entity_tokens = {
        entity_id: tokenize(
            combine_metadata_text(
                entity_title.get(entity_id, entity_id),
                entity_description.get(entity_id, ""),
                entity_alias.get(entity_id, ""),
            )
        )
        for entity_id in entity_title
    }
    relation_tokens = {
        relation_id: tokenize(
            combine_metadata_text(
                relation_title.get(relation_id, relation_id),
                relation_description.get(relation_id, ""),
                relation_alias.get(relation_id, ""),
            )
        )
        for relation_id in relation_title
    }

    return KGTextMetadata(
        entity_title=entity_title,
        entity_description=entity_description,
        entity_alias=entity_alias,
        relation_title=relation_title,
        relation_description=relation_description,
        relation_alias=relation_alias,
        entity_tokens=entity_tokens,
        relation_tokens=relation_tokens,
    )
