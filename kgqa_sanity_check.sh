# Evaluation on Kinship with evidence paths only
# python ./llm_kgqa.py --dataset kinship --hops n --llm-model gemma3 -e
python ./llm_kgqa.py --dataset kinship --hops n --llm-model llama3 -e
python ./llm_kgqa.py --dataset kinship --hops n --llm-model llama3.1 -e
python ./llm_kgqa.py --dataset kinship --hops n --llm-model deepseek-coder -e
# python ./llm_kgqa.py --dataset kinship --hops n --llm-model qwen2.5 -e
python ./llm_kgqa.py --dataset kinship --hops n --llm-model gpt-oss -e
python ./llm_kgqa.py --dataset kinship --hops n --llm-model mixtral -e

# Evaluation on MQuAKE with evidence paths only
# python ./llm_kgqa.py --dataset mquake --hops n --llm-model gemma3 -e
python ./llm_kgqa.py --dataset mquake --hops n --llm-model llama3 -e
python ./llm_kgqa.py --dataset mquake --hops n --llm-model llama3.1 -e
python ./llm_kgqa.py --dataset mquake --hops n --llm-model deepseek-coder -e
python ./llm_kgqa.py --dataset mquake --hops n --llm-model qwen2.5 -e
python ./llm_kgqa.py --dataset mquake --hops n --llm-model gpt-oss -e
python ./llm_kgqa.py --dataset mquake --hops n --llm-model mixtral -e