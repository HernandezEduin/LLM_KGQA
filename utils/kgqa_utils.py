import re
from typing import List, Tuple

def translate_path(triplet_path: List[Tuple[str, str, str]], entity_title: dict, relation_title: dict) -> List[Tuple[str, str, str]]:
    """
    Translate triplet paths into human-readable format using entity and relation titles.

    Args:
        triplet_path (List[Tuple[str, str, str]]): List of triplets (head, relation, tail).
        entity_title (dict): Mapping of entity IDs to titles.
        relation_title (dict): Mapping of relation IDs to titles.

    Returns:
        List[Tuple[str, str, str]]: Human-readable triplet paths.
    """
    readable_path = []
    for head, relation, tail in triplet_path:
        head_name = entity_title.get(str(head), str(head))
        relation_name = relation_title.get(str(relation), str(relation))
        tail_name = entity_title.get(str(tail), str(tail))
        readable_path.append((head_name, relation_name, tail_name))
    return readable_path

def extract_final_answer(output: str) -> str:
    """
    Extract the final answer from the LLM output.

    Args:
        output (str): The raw output from the LLM.
    Returns:
        str: The cleaned final answer.
    """
    # Remove leading phrases like "Answer:", "The answer is", etc.
    cleaned = re.sub(r'(?i)(^.*?(answer is|final answer|output|response)[:,\s]*)', '', output)
    # Take the first line or token until punctuation
    cleaned = cleaned.strip().split("\n")[0] #.split(".")[0] # TODO: verify if "." is a good idea, i.e., "U.S.A."
    # Optional: remove quotes, trailing punctuation
    cleaned = cleaned.strip(' "\'.')
    # remove parentheses
    cleaned = re.sub(r'[\(\)]', '', cleaned)
    return cleaned