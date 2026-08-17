import re
from typing import List, Union

from utils.kgqa_types import (
    EntityTitleMap,
    ReadableTriplet,
    RelationTitleMap,
    TripletList,
)

def translate_path(
    triplet_path: TripletList,
    entity_title: EntityTitleMap,
    relation_title: RelationTitleMap,
) -> list[ReadableTriplet]:
    """
    Translate triplet paths into human-readable format using entity and relation titles.

    Args:
        triplet_path (TripletList): List of triplets (head, relation, tail).
        entity_title (EntityTitleMap): Mapping of entity IDs to titles.
        relation_title (RelationTitleMap): Mapping of relation IDs to titles.

    Returns:
        list[ReadableTriplet]: Human-readable triplet paths.
    """
    readable_path = []
    for head, relation, tail in triplet_path:
        head_name = entity_title.get(str(head), str(head))
        relation_name = relation_title.get(str(relation), str(relation))
        tail_name = entity_title.get(str(tail), str(tail))
        readable_path.append((head_name, relation_name, tail_name))
    return readable_path

def extract_final_answer(output: Union[str, List[str]], lower: bool = False) -> Union[str, List[str]]:
    """
    Extract the final answer from the model's output, which may contain additional text.

    Args:
        output (Union[str, List[str]]): The raw output from the model, which can be a string or a list of strings.
        lower (bool): Whether to convert the extracted answer(s) to lowercase.

    Returns:
        Union[str, List[str]]: The extracted final answer(s).
    """
    if isinstance(output, list):
        res = [_clearn_answer(ans) for ans in output]
    else:
        res = _clearn_answer(output)

    if lower:
        if isinstance(output, list):
            res = [ans.lower() for ans in res]
        else:
            res = res.lower()
    return res

def _clearn_answer(answer: str) -> str:
    """
    Clean the answer string by removing leading phrases and extraneous characters.

    Args:
        answer (str): The raw answer string.
    
    Returns:
        str: The cleaned answer string.
    """
    # Remove leading phrases like "Answer:", "The answer is", etc.
    cleaned = re.sub(r'(?i)(^.*?(answer is|final answer|output|response)[:,\s]*)', '', answer)
    # Take the first line or token until punctuation
    cleaned = cleaned.strip().split("\n")[0] #.split(".")[0] # TODO: verify if "." is a good idea, i.e., "U.S.A."
    # Optional: remove quotes, trailing punctuation
    cleaned = cleaned.strip(' "\'.')
    # remove parentheses
    cleaned = re.sub(r'[\(\)]', '', cleaned)
    return cleaned

def compare_answers(pred: str, gold: Union[str, List[str]]) -> bool:
    """
    Compare the predicted answer with the gold answer, ignoring case and whitespace.

    Args:
        pred (str): The predicted answer.
        gold (str): The gold answer.

    Returns:
        bool: True if the answers match, False otherwise.
    """
    if isinstance(gold, list):
        return pred in gold
    else:
        return pred == gold