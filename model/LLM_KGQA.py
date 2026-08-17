"""Compatibility imports for KGQA LLM clients.

Prefer importing explicit clients from:
- model.subgraph_llm_client.SubgraphLLMKGQAClient
- model.navigation_llm_client.NavigationLLMKGQAClient
"""

from model.base_llm_client import BaseLLMKGQAClient
from model.navigation_llm_client import NavigationLLMKGQAClient
from model.subgraph_llm_client import SubgraphLLMKGQAClient

# Backward compatibility for older scripts that imported the monolithic client.
LLM_KGQA_Client = NavigationLLMKGQAClient

__all__ = [
    "BaseLLMKGQAClient",
    "SubgraphLLMKGQAClient",
    "NavigationLLMKGQAClient",
    "LLM_KGQA_Client",
]
