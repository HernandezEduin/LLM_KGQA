"""Shared KGQA type aliases for clients, scripts, and graph utilities."""

from typing import Any, Callable, Dict, List, Set, Tuple


# Entity and relation identifiers
EntityId = str
RelationId = str


# Knowledge-graph triplets and paths
Triplet = Tuple[EntityId, RelationId, EntityId]
ReadableTriplet = Tuple[str, str, str]
ReadableTripletList = List[ReadableTriplet]
TripletList = List[Triplet]
TripletSet = Set[Triplet]
TripletCollection = TripletList | TripletSet
Path = List[Triplet]
PathList = List[Path]
RelationChain = List[RelationId]


# Entity and relation title maps
EntityTitleMap = Dict[EntityId, str]
RelationTitleMap = Dict[RelationId, str]


# Graph lookup indices
OutgoingIndex = Dict[EntityId, TripletList]
IncidenceIndex = Dict[EntityId, TripletList]
NeighborIndex = Dict[EntityId, Set[EntityId]]
RelationIndex = Dict[RelationId, Dict[EntityId, TripletList]]
RelationGroup = Tuple[RelationId, TripletList]
RelationGroups = List[RelationGroup]


# API, status, and statistics records
APIResponse = Dict[str, Any]
StatusInfo = Dict[str, Any]
Statistics = Dict[str, Any]


# Metric score records
MetricValue = float | None
MetricScores = Dict[str, MetricValue]
AggregateMetricScores = Dict[str, MetricValue | int]


# Prompt and client result tuples
PromptParts = Tuple[str, str]
NavigationDecision = Dict[str, Any]
NavigationStatus = Dict[str, Any]
NavigationResult = Tuple[str, str, NavigationStatus]
SubgraphResult = Tuple[str, str, StatusInfo]


# Navigation demonstrations
NavigationDemonstration = Dict[str, Any]
NavigationDemonstrationList = List[NavigationDemonstration]


# Navigation callbacks and parser helpers
TraceFn = Callable[[str], None]
StageParser = Callable[[str], NavigationDecision]
StageCallResult = Tuple[NavigationDecision | None, str | None, NavigationStatus, Exception | None]
