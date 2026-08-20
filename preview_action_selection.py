"""Preview max-action policies on real dataset questions without calling an LLM."""
import argparse
import os
from utils.action_selection import _score, _tokens, select_options
from utils.basic import load_pandas, load_triplets
from utils.graph_utils import build_outgoing_index
from utils.kgqa_utils import load_title_maps

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--dataset", default="mquake_single")
    p.add_argument("--hops", default="n")
    p.add_argument("--split", default="test", choices=["train", "test", "validation"])
    p.add_argument("--question-index", type=int, default=0, help="Zero-based index within the split.")
    p.add_argument("--num-questions", type=int, default=1)
    p.add_argument("--max-actions", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show-all", action="store_true")
    return p.parse_args()

def readable(action, entity_title, relation_title):
    h, r, t = action
    return entity_title.get(h, h), relation_title.get(r, r), entity_title.get(t, t)

def main():
    args = parse_args()
    if args.question_index < 0 or args.num_questions < 1 or args.max_actions < 1:
        raise ValueError("Indexes must be non-negative and counts must be positive.")
    root = os.path.join(args.data_dir, args.dataset)
    qa = load_pandas(os.path.join(root, f"qa_{args.hops}hop.csv"))
    if "SplitLabel" in qa.columns:
        qa = qa[qa["SplitLabel"] == args.split]
    qa = qa.reset_index(drop=True)
    rows = qa.iloc[args.question_index:args.question_index + args.num_questions]
    if rows.empty:
        raise IndexError(f"No questions at index {args.question_index} in the {args.split} split.")
    triplets = load_triplets(os.path.join(root, "triplets.txt"))
    outgoing = build_outgoing_index(set(map(tuple, triplets.values)))
    entity_title, relation_title, _ = load_title_maps(
        os.path.join(root, "node_data.csv"), os.path.join(root, "relation_data.csv")
    )
    for index, row in rows.iterrows():
        question, source = row["Question"], row["Source-Entity"]
        actions = outgoing.get(source, [])
        common = dict(question=question, seed=args.seed, step=1, option_kind="tuple_action",
                      current_entity=source, entity_title=entity_title, relation_title=relation_title)
        q_tokens = _tokens(question)
        print(f"\n{'=' * 80}\nQuestion index: {index}")
        print(f"Question: {question}")
        print(f"Source: {entity_title.get(source, source)} ({source})")
        print(f"Outgoing actions: {len(actions)}; max shown: {args.max_actions}")
        if args.show_all:
            print("\nAll outgoing actions:")
            for original_id, action in enumerate(actions):
                score = _score(action, "tuple_action", q_tokens, entity_title, relation_title)
                print(f"  original={original_id:<4} score={score:<2} {readable(action, entity_title, relation_title)}")
        for policy in ("first", "random", "question-aware"):
            selected = select_options(actions, args.max_actions, policy, **common)
            print(f"\n{policy}:")
            for prompt_id, (original_id, action) in enumerate(selected):
                score = _score(action, "tuple_action", q_tokens, entity_title, relation_title)
                print(f"  prompt={prompt_id:<2} original={original_id:<4} score={score:<2} "
                      f"{readable(action, entity_title, relation_title)}")

if __name__ == "__main__":
    main()
