import os
import re
from typing import List, Tuple, Union

from utils.basic import load_pandas
from utils.kgqa_types import (
    EntityTitleMap,
    ReadableTripletList,
    RelationTitleMap,
    StatusInfo,
    TripletList,
)

def translate_path(
    triplet_path: TripletList,
    entity_title: EntityTitleMap,
    relation_title: RelationTitleMap,
) -> ReadableTripletList:
    """
    Translate triplet paths into human-readable format using entity and relation titles.

    Args:
        triplet_path (TripletList): List of triplets (head, relation, tail).
        entity_title (EntityTitleMap): Mapping of entity IDs to titles.
        relation_title (RelationTitleMap): Mapping of relation IDs to titles.

    Returns:
        ReadableTripletList: Human-readable triplet paths.
    """
    readable_path = []
    for head, relation, tail in triplet_path:
        head_name = entity_title.get(str(head), str(head))
        relation_name = relation_title.get(str(relation), str(relation))
        tail_name = entity_title.get(str(tail), str(tail))
        readable_path.append((head_name, relation_name, tail_name))
    return readable_path


def load_title_maps(
    entity_file: str,
    relation_file: str,
) -> Tuple[EntityTitleMap, RelationTitleMap, StatusInfo]:
    # TODO: Allow for different column names for QID/Property and Title, e.g., "EID"/"RID" and "Title" in MetaQA.
    """Load optional entity/relation label maps, falling back to identity labels.

    Encoded datasets such as MQuAKE provide ``node_data.csv`` and
    ``relation_data.csv``. Unencoded datasets such as kinship_v2 omit those
    files; in that case raw node and relation strings are already readable,
    so empty maps intentionally trigger identity formatting.

    Args:
        entity_file (str): Path to optional node_data.csv with QID and Title columns.
        relation_file (str): Path to optional relation_data.csv with Property and Title columns.

    Returns:
        Tuple[EntityTitleMap, RelationTitleMap, StatusInfo]: Entity title map,
            relation title map, and metadata describing each mapping source.

    Raises:
        ValueError: If a provided mapping file is missing required columns.
    """
    mapping_status: StatusInfo = {
        'entity_title_source': 'identity',
        'relation_title_source': 'identity',
        'entity_title_count': 0,
        'relation_title_count': 0,
        'entity_title_file': entity_file,
        'relation_title_file': relation_file,
    }

    entity_title: EntityTitleMap = {}
    if os.path.exists(entity_file):
        entity_df = load_pandas(entity_file)
        required_columns = {'QID', 'Title'}
        if not required_columns.issubset(entity_df.columns):
            raise ValueError(
                f"Entity mapping file {entity_file} must contain columns {sorted(required_columns)}."
            )
        entity_df.set_index('QID', inplace=True)
        entity_title = entity_df['Title'].to_dict()
        mapping_status['entity_title_source'] = 'file'
        mapping_status['entity_title_count'] = len(entity_title)

    relation_title: RelationTitleMap = {}
    if os.path.exists(relation_file):
        relation_df = load_pandas(relation_file)
        required_columns = {'Property', 'Title'}
        if not required_columns.issubset(relation_df.columns):
            raise ValueError(
                f"Relation mapping file {relation_file} must contain columns {sorted(required_columns)}."
            )
        relation_df.set_index('Property', inplace=True)
        relation_title = relation_df['Title'].to_dict()
        mapping_status['relation_title_source'] = 'file'
        mapping_status['relation_title_count'] = len(relation_title)

    return entity_title, relation_title, mapping_status


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