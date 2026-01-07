def translate_path(triplet_path: list, entity_title: dict, relation_title: dict) -> list:
    """
    Translate triplet paths into human-readable format using entity and relation titles.

    Args:
        triplet_path (list): List of triplets (head, relation, tail).
        entity_title (dict): Mapping of entity IDs to titles.
        relation_title (dict): Mapping of relation IDs to titles.

    Returns:
        list: Human-readable triplet paths.
    """
    readable_path = []
    for head, relation, tail in triplet_path:
        head_name = entity_title.get(str(head), str(head))
        relation_name = relation_title.get(str(relation), str(relation))
        tail_name = entity_title.get(str(tail), str(tail))
        readable_path.append((head_name, relation_name, tail_name))
    return readable_path