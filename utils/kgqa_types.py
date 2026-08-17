"""Shared KGQA type aliases for clients, scripts, and graph utilities."""

from typing import Any, Callable, Dict, List, Set, Tuple


EntityId = str
RelationId = str
Triplet = Tuple[EntityId, RelationId, EntityId]
ReadableTriplet = Tuple[str, str, str]
ReadableTripletList = List[ReadableTriplet]
TripletList = List[Triplet]
TripletSet = Set[Triplet]
TripletCollection = TripletList | TripletSet
Path = List[Triplet]
PathList = List[Path]
RelationChain = List[RelationId]
EntityTitleMap = Dict[EntityId, str]
RelationTitleMap = Dict[RelationId, str]
OutgoingIndex = Dict[EntityId, TripletList]
IncidenceIndex = Dict[EntityId, TripletList]
NeighborIndex = Dict[EntityId, Set[EntityId]]
RelationIndex = Dict[RelationId, Dict[EntityId, TripletList]]
RelationGroup = Tuple[RelationId, TripletList]
RelationGroups = List[RelationGroup]
APIResponse = Dict[str, Any]
StatusInfo = Dict[str, Any]
Statistics = Dict[str, Any]
PromptParts = Tuple[str, str]
NavigationDecision = Dict[str, Any]
NavigationStatus = Dict[str, Any]
NavigationResult = Tuple[str, str, NavigationStatus]
TraceFn = Callable[[str], None]
StageParser = Callable[[str], NavigationDecision]
StageCallResult = Tuple[NavigationDecision | None, str | None, NavigationStatus, Exception | None]
SubgraphResult = Tuple[str, str, StatusInfo]
