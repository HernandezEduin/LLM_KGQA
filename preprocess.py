import argparse
import os
import json
import pandas as pd

from utils.basic import load_triplets, save_triplets, sort_qid_list

def parse_args():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path containing the dataset splits.')
    parser.add_argument('--dataset', type=str, default='kinship',
                        help='Name of the dataset to process.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    data_dir = os.path.join(args.data_dir, args.dataset)
    
    # --------------------------------------------------------------------------
    'Load/Create full triplets'

    # if triplet files do not exist, use train, valid, test files to create them
    if not os.path.exists(os.path.join(data_dir, 'triplets.txt')):
        print(f"Triplet file not found. Creating from train, valid, and test splits in {data_dir}...")
        
        # Load triplets from train, valid, and test splits
        train_triplets = load_triplets(os.path.join(data_dir, 'train.txt'))
        valid_triplets = load_triplets(os.path.join(data_dir, 'valid.txt'))
        test_triplets = load_triplets(os.path.join(data_dir, 'test.txt'))

        all_triplets = pd.concat([train_triplets, valid_triplets, test_triplets], ignore_index=True)
        print(f"Total number of triplets: {len(all_triplets)}")
        all_triplets.drop_duplicates(inplace=True)
        print(f"Total number of unique triplets: {len(all_triplets)}")

        save_triplets(os.path.join(data_dir, 'triplets.txt'), all_triplets)
    else:
        print(f"Triplet file found at {os.path.join(data_dir, 'triplets.txt')}. No need to create from splits.")

        all_triplets = load_triplets(os.path.join(data_dir, 'triplets.txt'))
        print(f"Total number of triplets: {len(all_triplets)}")
        all_triplets.drop_duplicates(inplace=True)
        print(f"Total number of unique triplets: {len(all_triplets)}")
    

    # Load or create indexed triplets
    indexed_triplets_path = os.path.join(data_dir, 'triplets_indexed.csv')

    if not os.path.exists(indexed_triplets_path):
        print(f"Indexed triplet file not found. Creating from triplets in {data_dir}...")

        # Add an index column to the triplets for easier processing later
        all_triplets.reset_index(inplace=True)
        all_triplets.rename(columns={'index': 'id'}, inplace=True)

        # Make 'id' column the index
        all_triplets.set_index('id', inplace=True)

        # Save the indexed triplets
        all_triplets.to_csv(indexed_triplets_path, index=True)
    else:
        print(f"Indexed triplet file found at {indexed_triplets_path}. Loading...")

        # Load with indexed triplets
        # all_triplets = pd.read_csv(indexed_triplets_path)
        # all_triplets.set_index('id', inplace=True)
        all_triplets = pd.read_csv(indexed_triplets_path, index_col='id')

    print(f"Total number of triplets (indexed): {len(all_triplets)}")
    print(all_triplets.head(5))
    print("")

    # --------------------------------------------------------------------------
    'Load/Create vocabularies'

    vocab_dir = os.path.join(data_dir, 'vocab')
    os.makedirs(vocab_dir, exist_ok=True)

    # check if the vocabulary files exist    
    if not os.path.exists(os.path.join(vocab_dir, 'entity_vocab.json')) or not os.path.exists(os.path.join(vocab_dir, 'relation_vocab.json')):
        print(f"Vocabulary files not found. Creating entity_vocab.json and relation_vocab.json in {vocab_dir}...")
        
        entities = set()
        relations = set()

        using_qid = True
        using_pid = True

        for head, relation, tail in all_triplets.itertuples(index=False):
            entities.add(head)
            entities.add(tail)
            relations.add(relation)

            if head.startswith('Q') is False or tail.startswith('Q') is False:
                using_qid = False
            
            if relation.startswith('P') is False:
                using_pid = False

        entities_sorted = sort_qid_list(entities) if using_qid else sorted(entities)
        relations_sorted = sort_qid_list(relations) if using_pid else sorted(relations)

        # enumerate and create vocabularies
        entity_vocab = {entity: idx for idx, entity in enumerate(entities_sorted)}
        relation_vocab = {relation: idx for idx, relation in enumerate(relations_sorted)}
        
        with open(os.path.join(vocab_dir, 'entity_vocab.json'), 'w') as fout:
            json.dump(entity_vocab, fout, indent=4)

        with open(os.path.join(vocab_dir, 'relation_vocab.json'), 'w') as fout:
            json.dump(relation_vocab, fout, indent=4)
        
        print(f"Entities and relations vocabulary files created.")
    else:
        # load existing vocabularies

        entity_vocab = {}
        relation_vocab = {}
        with open(os.path.join(vocab_dir, 'entity_vocab.json'), 'r') as fin:
            entity_vocab = json.load(fin)
            
        with open(os.path.join(vocab_dir, 'relation_vocab.json'), 'r') as fin:
            relation_vocab = json.load(fin)

        print(f"Vocabulary files already exist in {data_dir}. No need to create them.")

    if not os.path.exists(os.path.join(vocab_dir, 'entity_title.json')) or not os.path.exists(os.path.join(vocab_dir, 'relation_title.json')):
        print(f"Creating entity_title.json and relation_title.json in {vocab_dir}...")
        
        # create title mappings (here simply using the entity/relation as title)
        entity_title = {entity: entity for entity in entity_vocab.keys()}
        relation_title = {relation: relation for relation in relation_vocab.keys()}

        with open(os.path.join(vocab_dir, 'entity_title.json'), 'w') as fout:
            json.dump(entity_title, fout, indent=4)

        with open(os.path.join(vocab_dir, 'relation_title.json'), 'w') as fout:
            json.dump(relation_title, fout, indent=4)

        print(f"Entity and relation title files created.")