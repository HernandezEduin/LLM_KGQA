import random

from model.base_llm_client import BaseLLMKGQAClient
from utils.kgqa_types import (
    EntityId,
    EntityTitleMap,
    PromptParts,
    RelationTitleMap,
    SubgraphResult,
    TripletCollection,
    TripletList,
)


class SubgraphLLMKGQAClient(BaseLLMKGQAClient):
    """LLM client for subgraph-at-once KGQA experiments."""

    def prepare_prompt(
        self,
        question: str,
        start_node: EntityId,
        triplets: TripletList,
        entity_title: EntityTitleMap,
        relation_title: RelationTitleMap,
    ) -> PromptParts:
        """
        Prepare the prompt for the LLM based on the question and triplets.

        Args:
            question (str): The natural-language question.
            start_node (EntityId): The starting node for the subgraph.
            triplets (TripletList): Knowledge-graph triplets.
            entity_title (EntityTitleMap): Mapping of entity IDs to titles.
            relation_title (RelationTitleMap): Mapping of relation IDs to titles.

        Returns:
            PromptParts: Prompt text and the formatted triplet block used in it.
        """
        start_node_str = self._format_entity_reference(start_node, entity_title)
        triplets_str = "{\n" + "\n".join(
            f"\t{self._format_triplet(triplet, entity_title, relation_title)}"
            for triplet in triplets
        ) + "\n}"
        template = (
            "You will be given a natural-language question, a starting node, and a set of knowledge-graph triplets.\n"
            "Answer the question using ONLY the information supported by the provided triplets.\n"
            "Each question contains a unique answer.\n"
            "Return only the final answer (no explanation, no reasoning, no extra text).\n"
            "Double-check the spelling of your answer.\n\n"
            f"Question: {question}\n"
            f"Starting Node: {start_node_str}\n"
            "Triplets (head, relation, tail):\n"
            f"{triplets_str}\n\n"
        )
        return template, triplets_str

    def process_question(
        self,
        question: str,
        start_node: EntityId,
        sub_graph: TripletCollection,
        entity_title: EntityTitleMap,
        relation_title: RelationTitleMap,
        random_seed: int = 42,
        sort_graph: bool = True,
    ) -> SubgraphResult:
        """
        Process a single question by preparing the prompt, sending it to the API, and extracting the prediction.

        Args:
            question (str): The natural-language question.
            start_node (EntityId): The starting node for the subgraph.
            sub_graph (TripletCollection): The subgraph of triplets to use for the question.
            entity_title (EntityTitleMap): Mapping of entity IDs to titles.
            relation_title (RelationTitleMap): Mapping of relation IDs to titles.
            random_seed (int): Seed for random operations to ensure reproducibility.
            sort_graph (bool): Whether to randomly shuffle the subgraph triplets.

        Returns:
            SubgraphResult: Prediction, formatted subgraph text, and call status metadata.
        """
        # randomly shuffle the subgraph triplets to avoid any ordering bias
        sub_graph = list(sub_graph)
        if sort_graph:
            random.Random(random_seed).shuffle(sub_graph)
        template, triplets_str = self.prepare_prompt(
            question=question,
            start_node=start_node,
            triplets=sub_graph,
            entity_title=entity_title,
            relation_title=relation_title,
        )
        out, status_info = self.chat(user_text=template)
        status_info.update(self.normalize_usage(out))

        if self.debug and status_info["status"] != "success":
            print(
                f"LLM response status: {status_info['status']}, "
                f"message: {status_info.get('message', '')}"
            )

        if status_info["status"] == "timeout":
            return "TIMEOUT", triplets_str, status_info
        if status_info["status"] != "success":
            return "ERROR", triplets_str, status_info

        if out is None:
            return "UNKNOWN", triplets_str, status_info

        if type(out) != dict or "message" not in out or "content" not in out["message"]:
            return "UNKNOWN", triplets_str, status_info
        return out["message"]["content"], triplets_str, status_info
