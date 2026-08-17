"""Compatibility imports for KGQA LLM clients.

Prefer importing explicit clients from:
- model.llm_kgqa_subgraph.SubgraphLLMKGQAClient
- model.llm_kgqa_navigation.NavigationLLMKGQAClient
"""

from model.llm_kgqa_base import BaseLLMKGQAClient
from model.llm_kgqa_navigation import NavigationLLMKGQAClient
from model.llm_kgqa_subgraph import SubgraphLLMKGQAClient

# Backward compatibility for older scripts that imported the monolithic client.
LLM_KGQA_Client = NavigationLLMKGQAClient

__all__ = [
    "BaseLLMKGQAClient",
    "SubgraphLLMKGQAClient",
    "NavigationLLMKGQAClient",
    "LLM_KGQA_Client",
]
