# Evaluation on Kinship with evidence paths only
# python ./kgqa_subgraph.py --dataset kinship --hops n --llm-model gemma3 -e
python ./kgqa_subgraph.py --dataset kinship --hops n --llm-model llama3 -e
python ./kgqa_subgraph.py --dataset kinship --hops n --llm-model llama3.1 -e
python ./kgqa_subgraph.py --dataset kinship --hops n --llm-model deepseek-coder -e
# python ./kgqa_subgraph.py --dataset kinship --hops n --llm-model qwen2.5 -e
python ./kgqa_subgraph.py --dataset kinship --hops n --llm-model gpt-oss -e
python ./kgqa_subgraph.py --dataset kinship --hops n --llm-model mixtral -e

# Evaluation on MQuAKE with evidence paths only
# python ./kgqa_subgraph.py --dataset mquake --hops n --llm-model gemma3 -e
python ./kgqa_subgraph.py --dataset mquake --hops n --llm-model llama3 -e
python ./kgqa_subgraph.py --dataset mquake --hops n --llm-model llama3.1 -e
python ./kgqa_subgraph.py --dataset mquake --hops n --llm-model deepseek-coder -e
python ./kgqa_subgraph.py --dataset mquake --hops n --llm-model qwen2.5 -e
python ./kgqa_subgraph.py --dataset mquake --hops n --llm-model gpt-oss -e
python ./kgqa_subgraph.py --dataset mquake --hops n --llm-model mixtral -e